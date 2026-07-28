"""Formal, distributed Qwen generation for Janus-TS evaluation.

The two ZeRO-3 ranks generate disjoint, equally-sized contiguous shards while
participating in the same number of ``generate`` calls.  Every call uses
``synced_gpus=True`` so one rank cannot leave the decoding loop while the
other still needs partitioned model weights.  Fragments and the merged JSONL
are installed atomically and carry all three content identities needed to
reject stale predictions.

This module deliberately preserves the ten *raw* beams in rank order,
including duplicates.  The only token-level normalization removes generation
padding after the first ``<|im_end|>``; the EOS token itself remains in the
decoded string for the strict TS parser to validate.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from .artifacts import sha256_bytes, sha256_file
from .constants import (
    FORMAL_EVAL_K,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)
from .evaluation import (
    EvaluationReport,
    ReactionEvaluation,
    aggregate_evaluations,
    evaluate_reaction,
)
from .representation import serialize_input
from .schema import ReactionRecord
from .tokenization import (
    TokenizationContractError,
    coerce_reaction_record,
    render_nonthinking_prompt,
)

GENERATION_SCHEMA_VERSION = "janus-ts-formal-generation-jsonl-v1"
SELECTION_SCHEMA_VERSION = "janus-ts-checkpoint-selection-v1"
EVALUATION_SCHEMA_VERSION = "janus-ts-formal-evaluation-v1"
TEST_COMPLETION_SCHEMA_VERSION = "janus-ts-test-evaluation-complete-v1"
EXPECTED_FORMAL_SPLIT_COUNTS = {"val": 994, "test": 996}
FORMAL_WORLD_SIZE = 2
FORMAL_NUM_BEAMS = 10
FormalTestRole = Literal["selected_checkpoint", "frozen_zero_shot"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GenerationContractError(RuntimeError):
    """Raised when formal generation would deviate from the frozen protocol."""


class TestAlreadyEvaluatedError(GenerationContractError):
    """Raised before touching the model when the selected test run is complete."""

    __test__ = False


class DistributedContext(Protocol):
    """Minimal distributed surface, intentionally easy to fake in tests."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def barrier(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TorchDistributedContext:
    """Adapter around an initialized :mod:`torch.distributed` process group."""

    @property
    def rank(self) -> int:
        self._require_initialized()
        return int(torch.distributed.get_rank())

    @property
    def world_size(self) -> int:
        self._require_initialized()
        return int(torch.distributed.get_world_size())

    def barrier(self) -> None:
        self._require_initialized()
        torch.distributed.barrier()

    @staticmethod
    def _require_initialized() -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise GenerationContractError(
                "formal generation requires an initialized torch.distributed group"
            )


def _validate_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GenerationContractError(f"{name} must be a lowercase SHA256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class GenerationIdentity:
    """Content identity shared by every prediction and evaluation artifact."""

    split: str
    data_fingerprint: str
    run_fingerprint: str
    checkpoint_fingerprint: str

    def __post_init__(self) -> None:
        if self.split not in EXPECTED_FORMAL_SPLIT_COUNTS:
            raise GenerationContractError(f"formal split must be val or test, got {self.split!r}")
        _validate_sha256(self.data_fingerprint, name="data_fingerprint")
        _validate_sha256(self.run_fingerprint, name="run_fingerprint")
        _validate_sha256(self.checkpoint_fingerprint, name="checkpoint_fingerprint")

    def to_json_dict(self) -> dict[str, str]:
        return {
            "split": self.split,
            "data_fingerprint": self.data_fingerprint,
            "run_fingerprint": self.run_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PromptEncoding:
    reaction_id: str
    atom_count: int
    prompt_text: str
    prompt_sha256: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GenerationRow:
    """One reaction's ten undecuplicated beam strings."""

    reaction_id: str
    ordinal: int
    atom_count: int
    prompt_sha256: str
    raw_beams: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reaction_id, str) or not self.reaction_id:
            raise GenerationContractError("reaction_id must be a non-empty string")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise GenerationContractError("prediction ordinal must be a non-negative integer")
        if (
            isinstance(self.atom_count, bool)
            or not isinstance(self.atom_count, int)
            or self.atom_count <= 0
        ):
            raise GenerationContractError("atom_count must be a positive integer")
        _validate_sha256(self.prompt_sha256, name="prompt_sha256")
        if len(self.raw_beams) != FORMAL_NUM_BEAMS:
            raise GenerationContractError(
                f"each reaction requires exactly {FORMAL_NUM_BEAMS} raw beams"
            )
        if any(not isinstance(beam, str) for beam in self.raw_beams):
            raise GenerationContractError("raw beams must all be strings")

    def to_json_dict(
        self,
        identity: GenerationIdentity,
        *,
        rank: int,
        world_size: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_SCHEMA_VERSION,
            **identity.to_json_dict(),
            "rank": rank,
            "world_size": world_size,
            "ordinal": self.ordinal,
            "reaction_id": self.reaction_id,
            "atom_count": self.atom_count,
            "prompt_sha256": self.prompt_sha256,
            "raw_beams": list(self.raw_beams),
        }


@dataclass(frozen=True, slots=True)
class FormalEvaluationResult:
    predictions_path: Path
    metrics_path: Path
    evaluations: tuple[ReactionEvaluation, ...]
    report: EvaluationReport
    eval_loss: float | None


def formal_generation_kwargs(*, synchronized: bool = True) -> dict[str, Any]:
    """Return the complete immutable decoding contract for Qwen."""

    return {
        "num_beams": FORMAL_NUM_BEAMS,
        "num_return_sequences": FORMAL_NUM_BEAMS,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "length_penalty": 1.0,
        "early_stopping": False,
        "eos_token_id": QWEN_IM_END_TOKEN_ID,
        "pad_token_id": QWEN_PAD_TOKEN_ID,
        "use_cache": True,
        "synced_gpus": bool(synchronized),
    }


def _as_unbatched_ids(value: Any, *, context: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, (list, tuple)) for item in value
    ):
        raise GenerationContractError(f"{context} did not produce one token-ID sequence")
    try:
        ids = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise GenerationContractError(f"{context} contains non-integral token IDs") from exc
    if not ids or any(item < 0 for item in ids):
        raise GenerationContractError(f"{context} is empty or contains a negative token ID")
    return ids


def encode_formal_prompt(
    tokenizer: Any,
    record: ReactionRecord | Mapping[str, Any],
) -> PromptEncoding:
    """Build the canonical evaluation prompt; truncation is never permitted."""

    reaction = coerce_reaction_record(record)
    user_content = serialize_input(reaction, training=False, epoch=0)
    prompt = render_nonthinking_prompt(tokenizer, user_content)
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        ids = _as_unbatched_ids(
            encoded["input_ids"], context=f"{reaction.reaction_id} formal prompt"
        )
    except (KeyError, TypeError, TokenizationContractError) as exc:
        if isinstance(exc, GenerationContractError):  # pragma: no cover - type narrowing
            raise
        raise GenerationContractError(
            f"cannot tokenize formal prompt for {reaction.reaction_id!r}: {exc}"
        ) from exc
    if len(ids) > MAX_SEQUENCE_LENGTH:
        raise GenerationContractError(
            f"{reaction.reaction_id}: formal prompt length {len(ids)} exceeds "
            f"the hard limit {MAX_SEQUENCE_LENGTH}; truncation is forbidden"
        )
    return PromptEncoding(
        reaction_id=reaction.reaction_id,
        atom_count=reaction.atom_count,
        prompt_text=prompt,
        prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
        input_ids=ids,
    )


def _model_device(model: Any) -> torch.device:
    value = getattr(model, "device", None)
    if value is None and getattr(model, "module", None) is not None:
        value = getattr(model.module, "device", None)
    if value is not None:
        return torch.device(value)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _logical_continuation(token_ids: Sequence[int]) -> list[int]:
    """Keep the first EOS and reject anything except padding after it."""

    values = [int(item) for item in token_ids]
    try:
        eos_index = values.index(QWEN_IM_END_TOKEN_ID)
    except ValueError:
        return values
    trailing = values[eos_index + 1 :]
    if any(token_id != QWEN_PAD_TOKEN_ID for token_id in trailing):
        raise GenerationContractError("generated sequence contains non-padding tokens after EOS")
    return values[: eos_index + 1]


def generate_reaction(
    model: Any,
    tokenizer: Any,
    record: ReactionRecord | Mapping[str, Any],
    *,
    ordinal: int,
    synchronized: bool = True,
) -> GenerationRow:
    """Generate exactly ten ordered raw beams for one canonical input."""

    prompt = encode_formal_prompt(tokenizer, record)
    device = _model_device(model)
    input_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **formal_generation_kwargs(synchronized=synchronized),
    )
    sequences = getattr(outputs, "sequences", outputs)
    if not isinstance(sequences, torch.Tensor):
        try:
            sequences = torch.as_tensor(sequences)
        except (TypeError, ValueError) as exc:
            raise GenerationContractError("model.generate did not return token sequences") from exc
    sequences = sequences.detach().cpu()
    if sequences.ndim != 2 or sequences.shape[0] != FORMAL_NUM_BEAMS:
        raise GenerationContractError(
            "model.generate must return shape "
            f"({FORMAL_NUM_BEAMS}, prompt+continuation), got {tuple(sequences.shape)}"
        )
    prompt_length = len(prompt.input_ids)
    if sequences.shape[1] < prompt_length:
        raise GenerationContractError("generated sequence is shorter than its prompt")
    expected_prefix = torch.tensor(prompt.input_ids, dtype=sequences.dtype)
    prompt_prefixes = sequences[:, :prompt_length]
    if not torch.equal(prompt_prefixes, expected_prefix.expand_as(prompt_prefixes)):
        raise GenerationContractError("model.generate changed the decoder-only prompt prefix")
    continuations = [_logical_continuation(row[prompt_length:].tolist()) for row in sequences]
    try:
        decoded = tokenizer.batch_decode(
            continuations,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise GenerationContractError(
            f"cannot decode beams for {prompt.reaction_id!r}: {exc}"
        ) from exc
    if not isinstance(decoded, (list, tuple)) or len(decoded) != FORMAL_NUM_BEAMS:
        raise GenerationContractError("tokenizer did not decode exactly ten beam strings")
    return GenerationRow(
        reaction_id=prompt.reaction_id,
        ordinal=ordinal,
        atom_count=prompt.atom_count,
        prompt_sha256=prompt.prompt_sha256,
        raw_beams=tuple(decoded),
    )


def rank_shard_bounds(
    total: int,
    *,
    rank: int,
    world_size: int = FORMAL_WORLD_SIZE,
) -> tuple[int, int]:
    """Return a contiguous shard, refusing unequal rank workloads."""

    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise GenerationContractError("dataset size must be a positive integer")
    if world_size != FORMAL_WORLD_SIZE:
        raise GenerationContractError(
            f"formal generation requires exactly {FORMAL_WORLD_SIZE} ranks"
        )
    if rank not in range(world_size):
        raise GenerationContractError(f"invalid distributed rank {rank}")
    if total % world_size:
        raise GenerationContractError(
            f"{total} examples cannot be split into equal halves across two ranks"
        )
    per_rank = total // world_size
    return rank * per_rank, (rank + 1) * per_rank


def fragment_path(
    output_dir: str | Path,
    identity: GenerationIdentity,
    *,
    rank: int,
    world_size: int = FORMAL_WORLD_SIZE,
) -> Path:
    return Path(output_dir) / (
        f"{identity.split}.{identity.checkpoint_fingerprint}."
        f"rank-{rank:05d}-of-{world_size:05d}.jsonl"
    )


def merged_predictions_path(output_dir: str | Path, identity: GenerationIdentity) -> Path:
    return Path(output_dir) / (
        f"{identity.split}.{identity.checkpoint_fingerprint}.predictions.jsonl"
    )


def metrics_path(output_dir: str | Path, identity: GenerationIdentity) -> Path:
    return Path(output_dir) / f"{identity.split}.{identity.checkpoint_fingerprint}.metrics.json"


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(path) from None
            temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl_payload(values: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for value in values
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_generation_fragment(
    output_dir: str | Path,
    identity: GenerationIdentity,
    *,
    rank: int,
    rows: Sequence[GenerationRow],
    world_size: int = FORMAL_WORLD_SIZE,
) -> Path:
    """Atomically write one non-empty, rank-owned prediction fragment."""

    if not rows:
        raise GenerationContractError("a formal prediction fragment cannot be empty")
    # The global total is not inferable from a single fragment, so ownership
    # is checked exactly during merge.  Here we only freeze rank metadata.
    if world_size != FORMAL_WORLD_SIZE or rank not in range(world_size):
        raise GenerationContractError("fragment rank metadata violates the two-rank contract")
    ordinals = [row.ordinal for row in rows]
    reaction_ids = [row.reaction_id for row in rows]
    if len(set(ordinals)) != len(ordinals) or len(set(reaction_ids)) != len(reaction_ids):
        raise GenerationContractError("fragment contains duplicate ordinals or reaction IDs")
    if ordinals != sorted(ordinals):
        raise GenerationContractError("fragment rows must be ordered by global ordinal")
    payload = _jsonl_payload(
        [row.to_json_dict(identity, rank=rank, world_size=world_size) for row in rows]
    )
    destination = fragment_path(output_dir, identity, rank=rank, world_size=world_size)
    _atomic_write_bytes(destination, payload, overwrite=True)
    return destination


_GENERATION_KEYS = {
    "schema_version",
    "split",
    "data_fingerprint",
    "run_fingerprint",
    "checkpoint_fingerprint",
    "rank",
    "world_size",
    "ordinal",
    "reaction_id",
    "atom_count",
    "prompt_sha256",
    "raw_beams",
}


def _row_from_payload(
    payload: Any,
    identity: GenerationIdentity,
    *,
    expected_rank: int | None,
    world_size: int,
) -> tuple[int, GenerationRow]:
    if not isinstance(payload, dict) or set(payload) != _GENERATION_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise GenerationContractError(f"prediction row has the wrong schema keys: {keys}")
    expected_identity = identity.to_json_dict()
    if payload["schema_version"] != GENERATION_SCHEMA_VERSION or any(
        payload[name] != value for name, value in expected_identity.items()
    ):
        raise GenerationContractError("prediction row schema or fingerprint mismatch")
    rank = payload["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(world_size):
        raise GenerationContractError(f"prediction row has invalid rank {rank!r}")
    if expected_rank is not None and rank != expected_rank:
        raise GenerationContractError(
            f"prediction row rank {rank} does not match fragment rank {expected_rank}"
        )
    if payload["world_size"] != world_size:
        raise GenerationContractError("prediction row world_size mismatch")
    beams = payload["raw_beams"]
    if not isinstance(beams, list):
        raise GenerationContractError("prediction raw_beams must be a JSON array")
    return rank, GenerationRow(
        reaction_id=payload["reaction_id"],
        ordinal=payload["ordinal"],
        atom_count=payload["atom_count"],
        prompt_sha256=payload["prompt_sha256"],
        raw_beams=tuple(beams),
    )


def _read_jsonl_rows(
    path: Path,
    identity: GenerationIdentity,
    *,
    expected_rank: int | None,
    world_size: int,
) -> list[tuple[int, GenerationRow]]:
    result: list[tuple[int, GenerationRow]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise GenerationContractError(
                        f"{path} line {line_number} lacks its terminal newline"
                    )
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GenerationContractError(
                        f"invalid JSON in {path} line {line_number}: {exc}"
                    ) from exc
                result.append(
                    _row_from_payload(
                        payload,
                        identity,
                        expected_rank=expected_rank,
                        world_size=world_size,
                    )
                )
    except OSError as exc:
        raise GenerationContractError(f"cannot read prediction artifact {path}: {exc}") from exc
    if not result:
        raise GenerationContractError(f"prediction artifact is empty: {path}")
    return result


def merge_generation_fragments(
    output_dir: str | Path,
    identity: GenerationIdentity,
    *,
    expected_reaction_ids: Sequence[str],
    world_size: int = FORMAL_WORLD_SIZE,
) -> tuple[Path, tuple[GenerationRow, ...]]:
    """Validate both fragments and merge strictly in dataset ID order."""

    expected_ids = tuple(expected_reaction_ids)
    if not expected_ids or any(not isinstance(item, str) or not item for item in expected_ids):
        raise GenerationContractError("expected reaction IDs must be non-empty strings")
    if len(set(expected_ids)) != len(expected_ids):
        raise GenerationContractError("expected reaction IDs are not unique")
    rank_shard_bounds(len(expected_ids), rank=0, world_size=world_size)
    all_rows: list[tuple[int, GenerationRow]] = []
    for rank in range(world_size):
        path = fragment_path(output_dir, identity, rank=rank, world_size=world_size)
        rank_rows = _read_jsonl_rows(
            path,
            identity,
            expected_rank=rank,
            world_size=world_size,
        )
        start, stop = rank_shard_bounds(len(expected_ids), rank=rank, world_size=world_size)
        if [row.ordinal for _, row in rank_rows] != list(range(start, stop)):
            raise GenerationContractError(
                f"rank {rank} fragment does not contain exactly ordinals [{start}, {stop})"
            )
        all_rows.extend(rank_rows)
    all_rows.sort(key=lambda item: item[1].ordinal)
    rows = tuple(row for _, row in all_rows)
    if tuple(row.reaction_id for row in rows) != expected_ids:
        raise GenerationContractError("merged reaction IDs differ from the expected dataset order")
    payload = _jsonl_payload(
        [row.to_json_dict(identity, rank=rank, world_size=world_size) for rank, row in all_rows]
    )
    destination = merged_predictions_path(output_dir, identity)
    _atomic_write_bytes(destination, payload, overwrite=True)
    return destination, rows


def load_merged_predictions(
    path: str | Path,
    identity: GenerationIdentity,
    *,
    expected_reaction_ids: Sequence[str],
    world_size: int = FORMAL_WORLD_SIZE,
) -> tuple[GenerationRow, ...]:
    """Strictly reload a merged JSONL, including shard-ownership checks."""

    values = _read_jsonl_rows(Path(path), identity, expected_rank=None, world_size=world_size)
    values.sort(key=lambda item: item[1].ordinal)
    rows = tuple(row for _, row in values)
    expected_ids = tuple(expected_reaction_ids)
    if [row.ordinal for row in rows] != list(range(len(expected_ids))):
        raise GenerationContractError("merged predictions have missing or duplicate ordinals")
    if tuple(row.reaction_id for row in rows) != expected_ids:
        raise GenerationContractError("merged predictions do not match expected reaction IDs")
    for rank, row in values:
        start, stop = rank_shard_bounds(len(expected_ids), rank=rank, world_size=world_size)
        if row.ordinal not in range(start, stop):
            raise GenerationContractError("merged prediction has invalid original rank ownership")
    return rows


def evaluate_generation_rows(
    records: Sequence[ReactionRecord | Mapping[str, Any]],
    rows: Sequence[GenerationRow],
) -> tuple[tuple[ReactionEvaluation, ...], EvaluationReport]:
    """Call the shared strict parser/scorer and exact dataset aggregator."""

    reactions = tuple(coerce_reaction_record(record) for record in records)
    predictions = tuple(rows)
    if len(reactions) != len(predictions):
        raise GenerationContractError("record and prediction counts differ")
    evaluations: list[ReactionEvaluation] = []
    for reaction, row in zip(reactions, predictions, strict=True):
        if reaction.reaction_id != row.reaction_id or reaction.atom_count != row.atom_count:
            raise GenerationContractError(
                f"prediction identity mismatch at ordinal {row.ordinal}: {row.reaction_id!r}"
            )
        evaluations.append(
            evaluate_reaction(
                reaction.reaction_id,
                row.raw_beams,
                reaction.ts_edges,
                atom_count=reaction.atom_count,
                report_k=FORMAL_EVAL_K,
            )
        )
    return tuple(evaluations), aggregate_evaluations(evaluations, report_k=FORMAL_EVAL_K)


EvalLossSource = float | Callable[[], float]


def resolve_eval_loss(source: EvalLossSource) -> float:
    """Resolve an already-aggregated loss or a distributed evaluation hook."""

    try:
        value = source() if callable(source) else source
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GenerationContractError(f"eval_loss hook returned an invalid value: {exc}") from exc
    if not math.isfinite(result) or result < 0.0:
        raise GenerationContractError("eval_loss must be finite and non-negative")
    return result


def trainer_eval_loss_hook(trainer: Any, *, eval_dataset: Any | None = None) -> Callable[[], float]:
    """Adapt ``Trainer.evaluate`` to the formal generation loss hook."""

    def evaluate() -> float:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if not isinstance(metrics, Mapping) or "eval_loss" not in metrics:
            raise GenerationContractError("Trainer.evaluate did not return eval_loss")
        return resolve_eval_loss(metrics["eval_loss"])

    return evaluate


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class SelectionProof:
    data_fingerprint: str
    run_fingerprint: str
    selected_checkpoint_fingerprint: str
    selected_epoch: int

    def __post_init__(self) -> None:
        _validate_sha256(self.data_fingerprint, name="data_fingerprint")
        _validate_sha256(self.run_fingerprint, name="run_fingerprint")
        _validate_sha256(
            self.selected_checkpoint_fingerprint,
            name="selected_checkpoint_fingerprint",
        )
        if (
            isinstance(self.selected_epoch, bool)
            or not isinstance(self.selected_epoch, int)
            or self.selected_epoch <= 0
        ):
            raise GenerationContractError("selected_epoch must be a positive integer")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "selection_locked": True,
            "data_fingerprint": self.data_fingerprint,
            "run_fingerprint": self.run_fingerprint,
            "selected_checkpoint_fingerprint": self.selected_checkpoint_fingerprint,
            "selected_epoch": self.selected_epoch,
        }


_SELECTION_KEYS = {
    "schema_version",
    "selection_locked",
    "data_fingerprint",
    "run_fingerprint",
    "selected_checkpoint_fingerprint",
    "selected_epoch",
}


def write_selection_proof(path: str | Path, proof: SelectionProof) -> Path:
    """Persist the immutable receipt that unlocks the single test evaluation."""

    destination = Path(path)
    _atomic_write_bytes(
        destination,
        _canonical_json_bytes(proof.to_json_dict()),
        overwrite=False,
    )
    return destination


def load_selection_proof(path: str | Path, identity: GenerationIdentity) -> SelectionProof:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationContractError(f"cannot load checkpoint selection proof: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _SELECTION_KEYS:
        raise GenerationContractError("checkpoint selection proof has the wrong schema")
    if (
        payload["schema_version"] != SELECTION_SCHEMA_VERSION
        or payload["selection_locked"] is not True
    ):
        raise GenerationContractError("checkpoint selection proof is not locked")
    proof = SelectionProof(
        data_fingerprint=payload["data_fingerprint"],
        run_fingerprint=payload["run_fingerprint"],
        selected_checkpoint_fingerprint=payload["selected_checkpoint_fingerprint"],
        selected_epoch=payload["selected_epoch"],
    )
    if identity.split != "test":
        raise GenerationContractError("a selection proof may only authorize test evaluation")
    if (
        proof.data_fingerprint != identity.data_fingerprint
        or proof.run_fingerprint != identity.run_fingerprint
        or proof.selected_checkpoint_fingerprint != identity.checkpoint_fingerprint
    ):
        raise GenerationContractError("test identity does not match the locked selection proof")
    return proof


def _process_start_ticks(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
    except OSError:
        return None
    return fields[21] if len(fields) > 21 else None


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return "unavailable"


class TestEvaluationLease:
    """Exclusive, reboot-safe claim for the selected checkpoint's one test run."""

    __test__ = False

    def __init__(self, output_dir: str | Path, identity: GenerationIdentity) -> None:
        if identity.split != "test":
            raise GenerationContractError("test lease requires split='test'")
        root = Path(output_dir)
        stem = f"test.{identity.run_fingerprint}.{identity.checkpoint_fingerprint}"
        self.claim_path = root / f"{stem}.claim.json"
        self.completion_path = root / f"{stem}.complete.json"
        self.identity = identity
        self._owner = {
            "hostname": socket.gethostname(),
            "boot_id": _boot_id(),
            "pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(os.getpid()),
        }
        self._acquired = False

    def _claim_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "janus-ts-test-evaluation-claim-v1",
            **self.identity.to_json_dict(),
            **self._owner,
        }

    def _remove_stale_local_claim(self) -> bool:
        try:
            payload = json.loads(self.claim_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("hostname") != self._owner["hostname"]:
            return False
        alive = payload.get("boot_id") == self._owner["boot_id"] and _process_start_ticks(
            payload.get("pid", -1)
        ) == payload.get("process_start_ticks")
        if alive:
            return False
        self.claim_path.unlink(missing_ok=True)
        return True

    def acquire(self) -> None:
        if self.completion_path.exists():
            raise TestAlreadyEvaluatedError(
                f"selected checkpoint test evaluation is already complete: {self.completion_path}"
            )
        payload = _canonical_json_bytes(self._claim_payload())
        try:
            _atomic_write_bytes(self.claim_path, payload, overwrite=False)
        except FileExistsError:
            if not self._remove_stale_local_claim():
                raise GenerationContractError(
                    f"another test evaluation owns {self.claim_path}"
                ) from None
            _atomic_write_bytes(self.claim_path, payload, overwrite=False)
        self._acquired = True

    def complete(self, *, predictions_path: Path, metrics_path_value: Path) -> None:
        if not self._acquired:
            raise GenerationContractError("cannot complete a test lease that is not acquired")
        payload = {
            "schema_version": TEST_COMPLETION_SCHEMA_VERSION,
            **self.identity.to_json_dict(),
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path_value),
        }
        _atomic_write_bytes(
            self.completion_path,
            _canonical_json_bytes(payload),
            overwrite=False,
        )
        self.release()

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            payload = json.loads(self.claim_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and all(
            payload.get(name) == value for name, value in self._owner.items()
        ):
            self.claim_path.unlink(missing_ok=True)
        self._acquired = False


@contextmanager
def _preserve_model_mode(model: Any):
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _validate_formal_records(
    records: Sequence[ReactionRecord | Mapping[str, Any]],
    identity: GenerationIdentity,
) -> tuple[ReactionRecord, ...]:
    reactions = tuple(coerce_reaction_record(record) for record in records)
    expected_count = EXPECTED_FORMAL_SPLIT_COUNTS[identity.split]
    if len(reactions) != expected_count:
        raise GenerationContractError(
            f"formal {identity.split} requires {expected_count} records, got {len(reactions)}"
        )
    ids = [reaction.reaction_id for reaction in reactions]
    if len(set(ids)) != len(ids):
        raise GenerationContractError("formal dataset contains duplicate reaction IDs")
    wrong_splits = [
        reaction.reaction_id for reaction in reactions if reaction.split != identity.split
    ]
    if wrong_splits:
        raise GenerationContractError(
            f"formal dataset has records outside split {identity.split}: {wrong_splits[:3]}"
        )
    return reactions


def _write_evaluation_metrics(
    destination: Path,
    identity: GenerationIdentity,
    *,
    predictions_path: Path,
    report: EvaluationReport,
    eval_loss: float | None,
) -> None:
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        **identity.to_json_dict(),
        "predictions_sha256": sha256_file(predictions_path),
        "eval_loss": eval_loss,
        "evaluation": report.to_json_dict(),
    }
    _atomic_write_bytes(destination, _canonical_json_bytes(payload), overwrite=True)


def run_formal_generation(
    model: Any,
    tokenizer: Any,
    records: Sequence[ReactionRecord | Mapping[str, Any]],
    identity: GenerationIdentity,
    *,
    output_dir: str | Path,
    eval_loss: EvalLossSource | None = None,
    selection_proof_path: str | Path | None = None,
    test_role: FormalTestRole = "selected_checkpoint",
    distributed: DistributedContext | None = None,
) -> FormalEvaluationResult | None:
    """Generate/evaluate val or the selected checkpoint's one test pass.

    All ranks call this function.  Rank zero returns the CPU evaluation result;
    the other rank returns ``None`` after its fragment is durable.  Validation
    loss hooks are invoked on *both* ranks before generation, allowing a
    distributed ``Trainer.evaluate`` implementation to perform its collectives.
    """

    context = distributed or TorchDistributedContext()
    if context.world_size != FORMAL_WORLD_SIZE:
        raise GenerationContractError(
            f"formal generation requires exactly {FORMAL_WORLD_SIZE} ranks"
        )
    reactions = _validate_formal_records(records, identity)
    rank = context.rank
    rank_shard_bounds(len(reactions), rank=rank, world_size=context.world_size)

    if test_role not in ("selected_checkpoint", "frozen_zero_shot"):
        raise GenerationContractError(f"invalid formal test role: {test_role!r}")
    if identity.split == "val":
        if test_role != "selected_checkpoint":
            raise GenerationContractError("validation cannot use a test-only model role")
        if eval_loss is None:
            raise GenerationContractError("formal validation requires eval_loss")
        resolved_eval_loss = resolve_eval_loss(eval_loss)
        proof = None
    else:
        if eval_loss is not None:
            raise GenerationContractError("test evaluation must not compute checkpoint eval_loss")
        if test_role == "selected_checkpoint":
            if selection_proof_path is None:
                raise GenerationContractError(
                    "selected-checkpoint test evaluation requires the locked selection proof"
                )
            proof = load_selection_proof(selection_proof_path, identity)
        else:
            if selection_proof_path is not None:
                raise GenerationContractError(
                    "frozen zero-shot test evaluation must not receive a selection proof"
                )
            proof = None
        resolved_eval_loss = None

    lease: TestEvaluationLease | None = None
    if identity.split == "test":
        # Every rank performs the read-only completion preflight, so a repeated
        # torchrun fails on both ranks before either reaches a collective.
        completion_probe = TestEvaluationLease(output_dir, identity)
        if completion_probe.completion_path.exists():
            raise TestAlreadyEvaluatedError(
                f"selected checkpoint test evaluation is already complete: "
                f"{completion_probe.completion_path}"
            )
        if rank == 0:
            lease = completion_probe
            lease.acquire()
    context.barrier()

    start, stop = rank_shard_bounds(len(reactions), rank=rank, world_size=context.world_size)
    rows: list[GenerationRow] = []
    try:
        with _preserve_model_mode(model):
            for ordinal in range(start, stop):
                rows.append(
                    generate_reaction(
                        model,
                        tokenizer,
                        reactions[ordinal],
                        ordinal=ordinal,
                        synchronized=True,
                    )
                )
        write_generation_fragment(
            output_dir,
            identity,
            rank=rank,
            rows=rows,
            world_size=context.world_size,
        )
        context.barrier()
        if rank != 0:
            return None

        expected_ids = tuple(reaction.reaction_id for reaction in reactions)
        predictions, merged_rows = merge_generation_fragments(
            output_dir,
            identity,
            expected_reaction_ids=expected_ids,
            world_size=context.world_size,
        )
        reloaded = load_merged_predictions(
            predictions,
            identity,
            expected_reaction_ids=expected_ids,
            world_size=context.world_size,
        )
        if reloaded != merged_rows:
            raise GenerationContractError("merged prediction round-trip changed content")
        evaluations, report = evaluate_generation_rows(reactions, merged_rows)
        metrics_destination = metrics_path(output_dir, identity)
        _write_evaluation_metrics(
            metrics_destination,
            identity,
            predictions_path=predictions,
            report=report,
            eval_loss=resolved_eval_loss,
        )
        result = FormalEvaluationResult(
            predictions_path=predictions,
            metrics_path=metrics_destination,
            evaluations=evaluations,
            report=report,
            eval_loss=resolved_eval_loss,
        )
        if lease is not None:
            # A trained checkpoint needs its locked selection proof.  The
            # separately fingerprinted zero-shot model is fixed before any
            # test prediction and therefore has no checkpoint-selection proof.
            if test_role == "selected_checkpoint" and proof is None:
                raise GenerationContractError("test selection proof was not retained")
            lease.complete(
                predictions_path=predictions,
                metrics_path_value=metrics_destination,
            )
        return result
    except BaseException:
        if lease is not None:
            lease.release()
        raise


__all__ = [
    "EXPECTED_FORMAL_SPLIT_COUNTS",
    "FORMAL_NUM_BEAMS",
    "FORMAL_WORLD_SIZE",
    "FormalTestRole",
    "GenerationContractError",
    "GenerationIdentity",
    "GenerationRow",
    "FormalEvaluationResult",
    "PromptEncoding",
    "SelectionProof",
    "TestAlreadyEvaluatedError",
    "TestEvaluationLease",
    "TorchDistributedContext",
    "encode_formal_prompt",
    "evaluate_generation_rows",
    "formal_generation_kwargs",
    "fragment_path",
    "generate_reaction",
    "load_merged_predictions",
    "load_selection_proof",
    "merge_generation_fragments",
    "merged_predictions_path",
    "metrics_path",
    "rank_shard_bounds",
    "resolve_eval_loss",
    "run_formal_generation",
    "trainer_eval_loss_hook",
    "write_generation_fragment",
    "write_selection_proof",
]
