"""Isolated, exploratory thinking-mode inference for Janus-TS.

This entrypoint intentionally cannot participate in formal validation, test
evaluation, or checkpoint selection.  It restores the selected durable
checkpoint exactly like the formal runtime (original Qwen base plus the
ordinary rank-64 portable adapter under two-rank ZeRO-3), but renders Qwen's
thinking chat prompt and uses the separately frozen sampling profile.

Example::

    torchrun --standalone --nproc-per-node=2 -m janus_ts.thinking_inference \
      --config configs/transition1x.yaml \
      --processed-path artifacts/data/processed/transition1x/... \
      --checkpoint-dir artifacts/.../epoch-00005 \
      --output-dir artifacts/exploration --split val --limit 2

All files are written below a content-addressed ``thinking-exploratory``
subdirectory.  No formal metric, selection proof, test lease, or run-state
file is read or written here.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .artifacts import (
    ArtifactError,
    mark_complete,
    read_complete_manifest,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from .config import ExperimentConfig, GenerationConfig, load_config
from .constants import (
    MAX_SEQUENCE_LENGTH,
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
    SYSTEM_PROMPT,
)
from .device_map_runtime import load_device_map_portable_model
from .distributed import (
    TorchrunContext,
    assert_initialized_two_rank_job,
    configure_reproducibility,
    preflight_torchrun_environment,
)
from .formal_eval_runtime import (
    DeepSpeedGenerationProxy,
    DurableCheckpoint,
    PreparedFormalData,
    build_formal_eval_arguments,
    build_formal_trainer,
    generation_module_from_trainer,
    initialize_inference_engine,
    inspect_durable_checkpoint,
    load_zero3_portable_model,
)
from .preprocessing import (
    load_pinned_tokenizer,
    load_processed_dataset,
    reaction_record_from_row,
)
from .representation import serialize_input
from .runtime import install_frozen_environment
from .schema import ReactionRecord
from .tokenization import EMPTY_THINK_PREFIX, CausalLMCollator

THINKING_EXPLORATION_SCHEMA_VERSION = "janus-ts-thinking-exploration-v3"
THINKING_PREDICTION_SCHEMA_VERSION = "janus-ts-thinking-prediction-v2"
THINKING_REQUEST_SCHEMA_VERSION = "janus-ts-thinking-request-v3"
THINKING_BATCH_PROGRESS_SCHEMA_VERSION = "janus-ts-thinking-batch-progress-v1"
ARTIFACT_CLASS = "exploratory-non-formal"
THINKING_PROMPT_SUFFIX = "<think>\n"
DEFAULT_EXPLORATION_LIMIT = 2
WORLD_SIZE = 2
THINKING_TEST_BATCH_SIZE = 2
Split = Literal["val", "test"]


class ThinkingInferenceError(RuntimeError):
    """An exploratory thinking-inference invariant was violated."""


@dataclass(frozen=True, slots=True)
class ThinkingPromptEncoding:
    """One canonical MoleCode input rendered with thinking enabled."""

    reaction_id: str
    atom_count: int
    prompt_text: str
    prompt_sha256: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ThinkingPrediction:
    """Ordered raw sampled continuations; they are deliberately not scored here."""

    reaction_id: str
    ordinal: int
    atom_count: int
    prompt_sha256: str
    prompt_tokens: int
    raw_responses: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.raw_responses or any(
            not isinstance(response, str) for response in self.raw_responses
        ):
            raise ThinkingInferenceError(
                "thinking prediction requires one or more raw string responses"
            )

    def to_json_dict(
        self,
        *,
        split: Split,
        rank: int,
        request_sha256: str,
        checkpoint_fingerprint: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": THINKING_PREDICTION_SCHEMA_VERSION,
            "artifact_class": ARTIFACT_CLASS,
            "formal_eligible": False,
            "split": split,
            "rank": rank,
            "world_size": WORLD_SIZE,
            "request_sha256": request_sha256,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "ordinal": self.ordinal,
            "reaction_id": self.reaction_id,
            "atom_count": self.atom_count,
            "prompt_sha256": self.prompt_sha256,
            "prompt_tokens": self.prompt_tokens,
            "sample_count": len(self.raw_responses),
            "raw_responses": list(self.raw_responses),
        }


@dataclass(frozen=True, slots=True)
class ScheduledRecord:
    """A real output or a synchronized dummy call assigned to one rank."""

    record: ReactionRecord
    ordinal: int
    is_dummy: bool


@dataclass(frozen=True, slots=True)
class ThinkingExplorationReceipt:
    """Content-verified receipt for one isolated exploratory request."""

    path: Path
    payload: Mapping[str, Any]


_EXPECTED_PROFILE: dict[str, Any] = {
    "enable_thinking": True,
    "num_beams": 1,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 8192,
    "length_penalty": None,
    "early_stopping": None,
    "report_k": None,
}


def validate_thinking_profile(profile: GenerationConfig) -> None:
    """Reject any deviation from the user-confirmed exploratory profile."""

    actual = profile.model_dump(mode="python")
    mismatches = [
        f"{name}={actual.get(name)!r}, expected {expected!r}"
        for name, expected in _EXPECTED_PROFILE.items()
        if actual.get(name) != expected
    ]
    if mismatches:
        raise ThinkingInferenceError(
            "thinking generation profile mismatch: " + "; ".join(mismatches)
        )


def thinking_generation_kwargs(
    profile: GenerationConfig,
    *,
    sample_count: int | None = None,
    synchronized: bool = True,
) -> dict[str, Any]:
    """Return only kwargs supported by Transformers' generation API.

    ``presence_penalty=0`` is retained in the artifact manifest for provenance,
    but is not forwarded: Transformers does not expose that OpenAI-style
    argument here, and zero would be the identity operation in any case.
    """

    validate_thinking_profile(profile)
    effective_sample_count = profile.num_return_sequences if sample_count is None else sample_count
    if (
        isinstance(effective_sample_count, bool)
        or not isinstance(effective_sample_count, int)
        or effective_sample_count <= 0
    ):
        raise ThinkingInferenceError("thinking sample_count must be a positive integer")
    return {
        "num_beams": profile.num_beams,
        "num_return_sequences": effective_sample_count,
        "do_sample": profile.do_sample,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "min_p": profile.min_p,
        "repetition_penalty": profile.repetition_penalty,
        "max_new_tokens": profile.max_new_tokens,
        "eos_token_id": QWEN_IM_END_TOKEN_ID,
        "pad_token_id": QWEN_PAD_TOKEN_ID,
        "use_cache": True,
        "synced_gpus": bool(synchronized),
    }


def render_thinking_prompt(
    tokenizer: Any,
    user_content: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Render the same two context turns while opening a non-empty think block."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception as exc:  # tokenizer/Jinja exceptions vary by version
        raise ThinkingInferenceError(
            "Qwen chat template could not render enable_thinking=True"
        ) from exc
    if not isinstance(prompt, str):
        raise ThinkingInferenceError("Qwen thinking chat template did not return text")
    if EMPTY_THINK_PREFIX in prompt:
        raise ThinkingInferenceError(
            "thinking prompt contains the non-thinking empty <think> prefix"
        )
    if not prompt.endswith(THINKING_PROMPT_SUFFIX):
        raise ThinkingInferenceError(
            "thinking prompt must end at Qwen's open '<think>\\n' generation boundary"
        )
    return prompt


def _as_unbatched_ids(value: Any, *, context: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, (list, tuple)) for item in value
    ):
        raise ThinkingInferenceError(f"{context} did not produce one token-ID sequence")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ThinkingInferenceError(f"{context} contains non-integral token IDs") from exc
    if not result or any(item < 0 for item in result):
        raise ThinkingInferenceError(f"{context} is empty or contains a negative token ID")
    return result


def encode_thinking_prompt(
    tokenizer: Any,
    record: ReactionRecord,
    *,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> ThinkingPromptEncoding:
    """Encode canonical evaluation-order MoleCode without truncation."""

    user_content = serialize_input(record, training=False, epoch=0)
    prompt = render_thinking_prompt(tokenizer, user_content)
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        ids = _as_unbatched_ids(
            encoded["input_ids"], context=f"{record.reaction_id} thinking prompt"
        )
    except (KeyError, TypeError) as exc:
        raise ThinkingInferenceError(
            f"cannot tokenize thinking prompt for {record.reaction_id!r}: {exc}"
        ) from exc
    if len(ids) > max_length:
        raise ThinkingInferenceError(
            f"{record.reaction_id}: thinking prompt length {len(ids)} exceeds "
            f"the hard input limit {max_length}; truncation is forbidden"
        )
    return ThinkingPromptEncoding(
        reaction_id=record.reaction_id,
        atom_count=record.atom_count,
        prompt_text=prompt,
        prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
        input_ids=ids,
    )


def _model_device(model: Any) -> Any:
    import torch

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
    values = [int(item) for item in token_ids]
    try:
        eos_index = values.index(QWEN_IM_END_TOKEN_ID)
    except ValueError:
        return values
    if any(item != QWEN_PAD_TOKEN_ID for item in values[eos_index + 1 :]):
        raise ThinkingInferenceError(
            "sampled response contains non-padding tokens after the first im_end"
        )
    return values[: eos_index + 1]


def generate_thinking_reaction(
    model: Any,
    tokenizer: Any,
    record: ReactionRecord,
    *,
    ordinal: int,
    profile: GenerationConfig,
    sample_count: int | None = None,
    max_input_length: int = MAX_SEQUENCE_LENGTH,
    synchronized: bool = True,
) -> ThinkingPrediction:
    """Sample ordered unparsed thinking responses in one synchronized call."""

    return generate_thinking_reactions(
        model,
        tokenizer,
        (record,),
        ordinals=(ordinal,),
        profile=profile,
        sample_count=sample_count,
        max_input_length=max_input_length,
        synchronized=synchronized,
    )[0]


def generate_thinking_reactions(
    model: Any,
    tokenizer: Any,
    records: Sequence[ReactionRecord],
    *,
    ordinals: Sequence[int],
    profile: GenerationConfig,
    sample_count: int | None = None,
    max_input_length: int = MAX_SEQUENCE_LENGTH,
    synchronized: bool = True,
) -> tuple[ThinkingPrediction, ...]:
    """Sample ordered responses for a left-padded batch of thinking prompts."""

    import torch

    if not records or len(records) != len(ordinals):
        raise ThinkingInferenceError("thinking batch requires equally sized records and ordinals")
    prompts = tuple(
        encode_thinking_prompt(tokenizer, record, max_length=max_input_length) for record in records
    )
    prompt_width = max(len(prompt.input_ids) for prompt in prompts)
    device = _model_device(model)
    input_ids = torch.full(
        (len(prompts), prompt_width),
        QWEN_PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, prompt in enumerate(prompts):
        prompt_length = len(prompt.input_ids)
        input_ids[index, -prompt_length:] = torch.tensor(
            prompt.input_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, -prompt_length:] = 1
    generation_kwargs = thinking_generation_kwargs(
        profile,
        sample_count=sample_count,
        synchronized=synchronized,
    )
    effective_sample_count = int(generation_kwargs["num_return_sequences"])
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    sequences = getattr(output, "sequences", output)
    if not isinstance(sequences, torch.Tensor):
        try:
            sequences = torch.as_tensor(sequences)
        except (TypeError, ValueError) as exc:
            raise ThinkingInferenceError("model.generate did not return token sequences") from exc
    sequences = sequences.detach().cpu()
    expected_sequence_count = len(prompts) * effective_sample_count
    if sequences.ndim != 2 or sequences.shape[0] != expected_sequence_count:
        raise ThinkingInferenceError(
            f"thinking generate must return exactly {expected_sequence_count} sequences, got "
            f"shape {tuple(sequences.shape)}"
        )
    if sequences.shape[1] < prompt_width:
        raise ThinkingInferenceError("sampled sequence is shorter than its prompt")

    predictions: list[ThinkingPrediction] = []
    cpu_input_ids = input_ids.detach().cpu()
    for batch_index, (prompt, ordinal) in enumerate(zip(prompts, ordinals, strict=True)):
        start = batch_index * effective_sample_count
        stop = start + effective_sample_count
        prompt_prefixes = sequences[start:stop, :prompt_width]
        expected_prefix = cpu_input_ids[batch_index].expand_as(prompt_prefixes)
        if not torch.equal(prompt_prefixes, expected_prefix):
            raise ThinkingInferenceError("thinking generation changed the decoder-only prompt")
        continuations = [
            _logical_continuation(sequence[prompt_width:].tolist())
            for sequence in sequences[start:stop]
        ]
        try:
            decoded = tokenizer.batch_decode(
                continuations,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise ThinkingInferenceError(
                f"cannot decode thinking response for {prompt.reaction_id!r}: {exc}"
            ) from exc
        if not isinstance(decoded, (list, tuple)) or len(decoded) != effective_sample_count:
            raise ThinkingInferenceError(
                f"tokenizer did not decode exactly {effective_sample_count} thinking responses"
            )
        predictions.append(
            ThinkingPrediction(
                reaction_id=prompt.reaction_id,
                ordinal=ordinal,
                atom_count=prompt.atom_count,
                prompt_sha256=prompt.prompt_sha256,
                prompt_tokens=len(prompt.input_ids),
                raw_responses=tuple(map(str, decoded)),
            )
        )
    return tuple(predictions)


def load_exploration_records(
    config: ExperimentConfig,
    processed_path: str | Path,
    split: Split,
    *,
    dataset_loader: Callable[[str | Path], Any] = load_processed_dataset,
) -> tuple[ReactionRecord, ...]:
    """Load and verify one complete frozen split without creating loss examples."""

    datasets = dataset_loader(processed_path)
    if split not in datasets:
        raise ThinkingInferenceError(f"processed DatasetDict has no {split!r} split")
    dataset = datasets[split]
    expected = config.data.expected_retained_counts[split]
    if len(dataset) != expected:
        raise ThinkingInferenceError(
            f"exploratory {split} requires the complete {expected}-row split, found {len(dataset)}"
        )
    records = tuple(reaction_record_from_row(dataset[index]) for index in range(len(dataset)))
    if any(record.split != split for record in records):
        raise ThinkingInferenceError(f"exploratory {split} data contains a foreign split")
    return records


def select_exploration_records(
    records: Sequence[ReactionRecord],
    *,
    reaction_ids: Sequence[str] = (),
    limit: int | None = None,
) -> tuple[ReactionRecord, ...]:
    """Select deterministically, defaulting to the first two dataset rows."""

    if not records:
        raise ThinkingInferenceError("cannot explore an empty record collection")
    by_id: dict[str, ReactionRecord] = {}
    for record in records:
        if record.reaction_id in by_id:
            raise ThinkingInferenceError(
                f"processed split contains duplicate reaction ID {record.reaction_id!r}"
            )
        by_id[record.reaction_id] = record

    requested = tuple(reaction_ids)
    if any(not isinstance(item, str) or not item for item in requested):
        raise ThinkingInferenceError("reaction IDs must be non-empty strings")
    if len(set(requested)) != len(requested):
        raise ThinkingInferenceError("requested reaction IDs contain duplicates")
    missing = [item for item in requested if item not in by_id]
    if missing:
        raise ThinkingInferenceError(f"requested reaction IDs are absent: {missing!r}")
    candidates = tuple(by_id[item] for item in requested) if requested else tuple(records)

    effective_limit = limit
    if effective_limit is None and not requested:
        effective_limit = DEFAULT_EXPLORATION_LIMIT
    if effective_limit is not None:
        if (
            isinstance(effective_limit, bool)
            or not isinstance(effective_limit, int)
            or effective_limit <= 0
        ):
            raise ThinkingInferenceError("--limit must be a positive integer")
        candidates = candidates[:effective_limit]
    if not candidates:
        raise ThinkingInferenceError("exploratory selection is empty")
    return candidates


def schedule_equal_rank_calls(
    records: Sequence[ReactionRecord],
) -> tuple[ScheduledRecord, ...]:
    """Append one deterministic dummy when needed so both ranks call generate equally."""

    if not records:
        raise ThinkingInferenceError("cannot schedule an empty exploratory selection")
    scheduled = [
        ScheduledRecord(record=record, ordinal=ordinal, is_dummy=False)
        for ordinal, record in enumerate(records)
    ]
    if len(scheduled) % WORLD_SIZE:
        last = scheduled[-1]
        scheduled.append(ScheduledRecord(record=last.record, ordinal=last.ordinal, is_dummy=True))
    return tuple(scheduled)


def _profile_manifest(profile: GenerationConfig) -> dict[str, Any]:
    validate_thinking_profile(profile)
    payload = profile.model_dump(mode="json")
    payload["presence_penalty_forwarded_to_transformers"] = False
    payload["presence_penalty_note"] = (
        "recorded only: unsupported by this Transformers generate path and zero is identity"
    )
    return payload


def _request_payload(
    config: ExperimentConfig,
    checkpoint: DurableCheckpoint,
    *,
    split: Split,
    records: Sequence[ReactionRecord],
    sample_count_per_reaction: int | None = None,
    reaction_batch_size: int = 1,
) -> dict[str, Any]:
    scheduled = schedule_equal_rank_calls(records)
    sample_count = (
        config.thinking_generation.num_return_sequences
        if sample_count_per_reaction is None
        else sample_count_per_reaction
    )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ThinkingInferenceError("thinking sample_count_per_reaction must be positive")
    if (
        isinstance(reaction_batch_size, bool)
        or not isinstance(reaction_batch_size, int)
        or reaction_batch_size <= 0
    ):
        raise ThinkingInferenceError("thinking reaction_batch_size must be positive")
    per_rank = len(scheduled) // WORLD_SIZE
    calls_per_rank = (per_rank + reaction_batch_size - 1) // reaction_batch_size
    return {
        "schema_version": THINKING_REQUEST_SCHEMA_VERSION,
        "artifact_class": ARTIFACT_CLASS,
        "formal_eligible": False,
        "affects_checkpoint_selection": False,
        "metrics_computed": False,
        "writes_formal_test_lease": False,
        "writes_run_state": False,
        "split": split,
        "selected_reaction_ids": [record.reaction_id for record in records],
        "selection_count": len(records),
        "sample_count_per_reaction": sample_count,
        "scheduled_generate_calls": calls_per_rank * WORLD_SIZE,
        "dummy_generate_calls": len(scheduled) - len(records),
        "scheduled_sample_count": len(scheduled) * sample_count,
        "dummy_sample_count": (len(scheduled) - len(records)) * sample_count,
        "world_size": WORLD_SIZE,
        "seed": config.seed,
        "config_fingerprint": config.sha256,
        "data_fingerprint": checkpoint.data_fingerprint,
        "run_fingerprint": checkpoint.run_fingerprint,
        "model_fingerprint": checkpoint.model_fingerprint,
        "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_global_step": checkpoint.global_step,
        "checkpoint_path": str(checkpoint.path),
        "prompt": {
            "representation_mode": "canonical-evaluation-order",
            "enable_thinking": True,
            "empty_think_prefix_allowed": False,
        },
        "generation_profile": _profile_manifest(config.thinking_generation),
        "generation_execution": {
            "num_return_sequences": sample_count,
            "candidates_batched_in_one_generate_call": True,
            "reaction_batch_size": reaction_batch_size,
        },
    }


def exploration_run_path(
    output_dir: str | Path,
    request_payload: Mapping[str, Any],
) -> tuple[Path, str]:
    """Return an isolated content-addressed location and its request digest."""

    request_sha256 = sha256_json(request_payload)
    checkpoint_fingerprint = str(request_payload["checkpoint_fingerprint"])
    split = str(request_payload["split"])
    path = (
        Path(output_dir) / "thinking-exploratory" / checkpoint_fingerprint / split / request_sha256
    )
    return path, request_sha256


def _atomic_write(path: Path, payload: bytes) -> None:
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
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in rows
        )
        + "\n"
    ).encode("utf-8")


def exploration_fragment_path(root: str | Path, *, rank: int) -> Path:
    if rank not in range(WORLD_SIZE):
        raise ThinkingInferenceError(f"invalid exploratory rank {rank}")
    return Path(root) / f"rank-{rank:05d}-of-{WORLD_SIZE:05d}.jsonl"


def write_exploration_fragment(
    root: str | Path,
    *,
    rank: int,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    destination = exploration_fragment_path(root, rank=rank)
    _atomic_write(destination, _jsonl_bytes(rows))
    return destination


def _batch_persistence_payload() -> dict[str, Any]:
    return {
        "schema_version": THINKING_BATCH_PROGRESS_SCHEMA_VERSION,
        "write_granularity": "one-generate-call",
        "write_mode": "atomic-jsonl-and-fsync",
        "seed_schedule": "base-seed-plus-global-batch-index-v1",
    }


def exploration_batch_path(
    root: str | Path,
    *,
    rank: int,
    batch_index: int,
) -> Path:
    if rank not in range(WORLD_SIZE):
        raise ThinkingInferenceError(f"invalid exploratory rank {rank}")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ThinkingInferenceError(f"invalid exploratory batch index {batch_index!r}")
    return (
        Path(root)
        / "batches"
        / f"rank-{rank:05d}-of-{WORLD_SIZE:05d}"
        / f"batch-{batch_index:06d}.jsonl"
    )


def write_exploration_batch(
    root: str | Path,
    *,
    rank: int,
    batch_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically persist one completed generate call without overwriting evidence."""

    destination = exploration_batch_path(root, rank=rank, batch_index=batch_index)
    payload = _jsonl_bytes(rows)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ThinkingInferenceError(f"invalid existing thinking batch: {destination}")
        if destination.read_bytes() != payload:
            raise ThinkingInferenceError(f"existing thinking batch differs: {destination}")
        return destination
    _atomic_write(destination, payload)
    return destination


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ThinkingInferenceError(f"cannot read exploratory fragment {path}: {exc}") from exc
    if raw and not raw.endswith("\n"):
        raise ThinkingInferenceError(f"exploratory fragment lacks final newline: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ThinkingInferenceError(
                f"invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ThinkingInferenceError(
                f"exploratory row is not an object in {path} line {line_number}"
            )
        rows.append(value)
    return rows


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    request_sha256: str,
    checkpoint_fingerprint: str,
    sample_count: int,
    expected_items: Sequence[ScheduledRecord] | None = None,
) -> None:
    expected_metadata = {
        "schema_version": THINKING_PREDICTION_SCHEMA_VERSION,
        "artifact_class": ARTIFACT_CLASS,
        "formal_eligible": False,
        "rank": rank,
        "world_size": WORLD_SIZE,
        "request_sha256": request_sha256,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "sample_count": sample_count,
    }
    for row in rows:
        raw_responses = row.get("raw_responses")
        if (
            any(row.get(name) != value for name, value in expected_metadata.items())
            or not isinstance(raw_responses, list)
            or len(raw_responses) != sample_count
            or any(not isinstance(response, str) for response in raw_responses)
        ):
            raise ThinkingInferenceError(
                f"rank {rank} exploratory fragment has stale or formal-eligible metadata"
            )
    if expected_items is None:
        return
    expected_rows = tuple(item for item in expected_items if not item.is_dummy)
    if [row.get("ordinal") for row in rows] != [item.ordinal for item in expected_rows] or [
        row.get("reaction_id") for row in rows
    ] != [item.record.reaction_id for item in expected_rows]:
        raise ThinkingInferenceError(
            f"rank {rank} persisted thinking batch has the wrong reaction identities"
        )


def _batch_inventory_paths(
    root: Path,
    request_payload: Mapping[str, Any],
) -> tuple[Path, ...]:
    persistence = request_payload.get("batch_persistence")
    if persistence is None:
        return ()
    if persistence != _batch_persistence_payload():
        raise ThinkingInferenceError("thinking batch persistence protocol mismatch")
    call_count = request_payload.get("scheduled_generate_calls")
    if (
        isinstance(call_count, bool)
        or not isinstance(call_count, int)
        or call_count <= 0
        or call_count % WORLD_SIZE
    ):
        raise ThinkingInferenceError("thinking request has an invalid generate-call count")
    calls_per_rank = call_count // WORLD_SIZE
    return tuple(
        exploration_batch_path(root, rank=rank, batch_index=batch_index)
        for rank in range(WORLD_SIZE)
        for batch_index in range(calls_per_rank)
    )


def _inventory(paths: Sequence[Path], *, root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def finalize_exploration_artifact(
    root: str | Path,
    request_payload: Mapping[str, Any],
    *,
    request_sha256: str,
) -> ThinkingExplorationReceipt:
    """Merge rank fragments and mark an immutable non-formal artifact complete."""

    destination = Path(root)
    selected_ids = tuple(str(item) for item in request_payload["selected_reaction_ids"])
    checkpoint_fingerprint = str(request_payload["checkpoint_fingerprint"])
    sample_count = int(request_payload["sample_count_per_reaction"])
    fragments = [exploration_fragment_path(destination, rank=rank) for rank in range(WORLD_SIZE)]
    rows: list[dict[str, Any]] = []
    rows_by_rank: list[list[dict[str, Any]]] = []
    for rank, fragment in enumerate(fragments):
        rank_rows = _read_jsonl(fragment)
        _validate_prediction_rows(
            rank_rows,
            rank=rank,
            request_sha256=request_sha256,
            checkpoint_fingerprint=checkpoint_fingerprint,
            sample_count=sample_count,
        )
        rows_by_rank.append(rank_rows)
        rows.extend(rank_rows)

    batch_paths = _batch_inventory_paths(destination, request_payload)
    if batch_paths:
        calls_per_rank = len(batch_paths) // WORLD_SIZE
        for rank in range(WORLD_SIZE):
            persisted_rows = [
                row
                for batch_index in range(calls_per_rank)
                for row in _read_jsonl(
                    exploration_batch_path(
                        destination,
                        rank=rank,
                        batch_index=batch_index,
                    )
                )
            ]
            if persisted_rows != rows_by_rank[rank]:
                raise ThinkingInferenceError(
                    f"rank {rank} final fragment differs from its persisted thinking batches"
                )
    rows.sort(key=lambda row: int(row.get("ordinal", -1)))
    if [row.get("ordinal") for row in rows] != list(range(len(selected_ids))):
        raise ThinkingInferenceError("exploratory output ordinals are incomplete or duplicated")
    if [row.get("reaction_id") for row in rows] != list(selected_ids):
        raise ThinkingInferenceError("exploratory output reaction IDs differ from the request")

    predictions = destination / "predictions.jsonl"
    _atomic_write(predictions, _jsonl_bytes(rows))
    payload_paths = [*batch_paths, *fragments, predictions]
    manifest = {
        **dict(request_payload),
        "schema_version": THINKING_EXPLORATION_SCHEMA_VERSION,
        "request_schema_version": THINKING_REQUEST_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "payload_inventory": _inventory(payload_paths, root=destination),
    }
    mark_complete(destination, manifest)
    return validate_exploration_artifact(destination, request_sha256=request_sha256)


def validate_exploration_artifact(
    root: str | Path,
    *,
    request_sha256: str | None = None,
) -> ThinkingExplorationReceipt:
    """Validate classification, request identity, and every payload digest."""

    destination = Path(root)
    try:
        manifest = read_complete_manifest(destination)
    except ArtifactError as exc:
        raise ThinkingInferenceError(str(exc)) from exc
    required = {
        "schema_version": THINKING_EXPLORATION_SCHEMA_VERSION,
        "artifact_class": ARTIFACT_CLASS,
        "formal_eligible": False,
        "affects_checkpoint_selection": False,
        "metrics_computed": False,
        "writes_formal_test_lease": False,
        "writes_run_state": False,
        "world_size": WORLD_SIZE,
    }
    if any(manifest.get(name) != value for name, value in required.items()):
        raise ThinkingInferenceError("artifact is not an isolated non-formal thinking exploration")
    sample_count = manifest.get("sample_count_per_reaction")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ThinkingInferenceError("exploratory artifact has an invalid sample count")
    actual_request = manifest.get("request_sha256")
    if request_sha256 is not None and actual_request != request_sha256:
        raise ThinkingInferenceError("exploratory artifact request fingerprint mismatch")
    inventory = manifest.get("payload_inventory")
    expected_names = {
        "rank-00000-of-00002.jsonl",
        "rank-00001-of-00002.jsonl",
        "predictions.jsonl",
    }
    expected_names.update(
        path.relative_to(destination).as_posix()
        for path in _batch_inventory_paths(destination, manifest)
    )
    if not isinstance(inventory, Mapping) or set(inventory) != expected_names:
        raise ThinkingInferenceError("exploratory artifact has the wrong payload inventory")
    for relative, raw_entry in inventory.items():
        if not isinstance(raw_entry, Mapping):
            raise ThinkingInferenceError(f"invalid inventory entry for {relative}")
        path = destination / relative
        size = raw_entry.get("size_bytes")
        digest = raw_entry.get("sha256")
        if (
            path.is_symlink()
            or not path.is_file()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or path.stat().st_size != size
            or not isinstance(digest, str)
            or sha256_file(path) != digest
        ):
            raise ThinkingInferenceError(f"exploratory payload failed verification: {relative}")
    return ThinkingExplorationReceipt(destination, manifest)


def _rank_rows(
    model: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    checkpoint: DurableCheckpoint,
    *,
    context: TorchrunContext,
    split: Split,
    scheduled: Sequence[ScheduledRecord],
    request_sha256: str,
    sample_count_per_reaction: int,
    generator: Callable[..., ThinkingPrediction],
) -> list[dict[str, Any]]:
    per_rank = len(scheduled) // WORLD_SIZE
    start = context.rank * per_rank
    stop = start + per_rank
    rows: list[dict[str, Any]] = []
    for item in scheduled[start:stop]:
        prediction = generator(
            model,
            tokenizer,
            item.record,
            ordinal=item.ordinal,
            profile=config.thinking_generation,
            sample_count=sample_count_per_reaction,
            max_input_length=config.model.max_sequence_length,
        )
        if len(prediction.raw_responses) != sample_count_per_reaction:
            raise ThinkingInferenceError(
                f"{item.record.reaction_id}: generator returned "
                f"{len(prediction.raw_responses)} responses, expected "
                f"{sample_count_per_reaction}"
            )
        if not item.is_dummy:
            rows.append(
                prediction.to_json_dict(
                    split=split,
                    rank=context.rank,
                    request_sha256=request_sha256,
                    checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
                )
            )
    return rows


def run_thinking_inference(
    config: ExperimentConfig,
    *,
    processed_path: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    split: Split = "val",
    reaction_ids: Sequence[str] = (),
    limit: int | None = None,
    sample_count_per_reaction: int | None = None,
    local_files_only: bool = True,
    environment_installer: Callable[[], Any] = install_frozen_environment,
    context_loader: Callable[[], TorchrunContext] = preflight_torchrun_environment,
    checkpoint_inspector: Callable[..., DurableCheckpoint] = inspect_durable_checkpoint,
    record_loader: Callable[..., tuple[ReactionRecord, ...]] = load_exploration_records,
    arguments_factory: Callable[..., Any] = build_formal_eval_arguments,
    reproducibility_configurer: Callable[..., None] = configure_reproducibility,
    context_validator: Callable[[Any, TorchrunContext, Any], None] = (
        assert_initialized_two_rank_job
    ),
    model_loader: Callable[..., Any] = load_zero3_portable_model,
    tokenizer_loader: Callable[..., Any] = load_pinned_tokenizer,
    trainer_builder: Callable[..., Any] = build_formal_trainer,
    inference_initializer: Callable[[Any], Any] = initialize_inference_engine,
    module_resolver: Callable[[Any], DeepSpeedGenerationProxy] = (generation_module_from_trainer),
    generator: Callable[..., ThinkingPrediction] = generate_thinking_reaction,
    barrier: Callable[[], None] | None = None,
    torch_module: Any | None = None,
) -> ThinkingExplorationReceipt:
    """Run one isolated two-rank exploratory request with injectable heavy steps."""

    if split not in ("val", "test"):
        raise ThinkingInferenceError(f"thinking split must be val or test, got {split!r}")
    validate_thinking_profile(config.thinking_generation)
    environment_installer()
    context = context_loader()
    checkpoint = checkpoint_inspector(checkpoint_dir, processed_path, config=config)
    records = record_loader(config, processed_path, split)
    selected = select_exploration_records(records, reaction_ids=reaction_ids, limit=limit)
    request_payload = _request_payload(
        config,
        checkpoint,
        split=split,
        records=selected,
        sample_count_per_reaction=sample_count_per_reaction,
    )
    run_root, request_sha256 = exploration_run_path(output_dir, request_payload)
    if (run_root / ".complete").is_file():
        return validate_exploration_artifact(run_root, request_sha256=request_sha256)
    run_root.mkdir(parents=True, exist_ok=True)

    # This ordering is critical: TrainingArguments activates ZeRO.Init before
    # the original 27B base and portable adapter are materialized.
    if torch_module is None:
        import torch as torch_module
    reproducibility_configurer(
        torch_module,
        seed=config.seed,
        cuda_device=context.local_rank,
    )
    arguments = arguments_factory(config, output_dir=run_root)
    context_validator(arguments, context, torch_module)
    model = model_loader(
        config,
        checkpoint,
        arguments,
        local_files_only=local_files_only,
    )
    tokenizer = tokenizer_loader(config, local_files_only=local_files_only)
    prepared = PreparedFormalData(
        records=selected,
        eval_dataset=None,
        collator=CausalLMCollator(max_length=config.model.max_sequence_length),
    )
    trainer = trainer_builder(model, arguments, prepared)
    inference_initializer(trainer)
    generation_model = module_resolver(trainer)

    if barrier is None:
        barrier = torch_module.distributed.barrier
    barrier()
    scheduled = schedule_equal_rank_calls(selected)
    rows = _rank_rows(
        generation_model,
        tokenizer,
        config,
        checkpoint,
        context=context,
        split=split,
        scheduled=scheduled,
        request_sha256=request_sha256,
        sample_count_per_reaction=int(request_payload["sample_count_per_reaction"]),
        generator=generator,
    )
    write_exploration_fragment(run_root, rank=context.rank, rows=rows)
    barrier()
    if context.rank == 0:
        finalize_exploration_artifact(
            run_root,
            request_payload,
            request_sha256=request_sha256,
        )
    barrier()
    return validate_exploration_artifact(run_root, request_sha256=request_sha256)


def _seed_local_thinking_batch(torch_module: Any, *, seed: int) -> None:
    """Give each persisted batch a restart-independent sampling stream."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ThinkingInferenceError(f"invalid thinking batch seed {seed!r}")
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)


def run_local_thinking_inference(
    config: ExperimentConfig,
    *,
    processed_path: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    split: Split = "val",
    reaction_ids: Sequence[str] = (),
    limit: int | None = None,
    sample_count_per_reaction: int | None = None,
    local_files_only: bool = True,
    environment_installer: Callable[[], Any] = install_frozen_environment,
    reproducibility_configurer: Callable[..., None] = configure_reproducibility,
    checkpoint_inspector: Callable[..., DurableCheckpoint] = inspect_durable_checkpoint,
    record_loader: Callable[..., tuple[ReactionRecord, ...]] = load_exploration_records,
    model_loader: Callable[..., Any] = load_device_map_portable_model,
    tokenizer_loader: Callable[..., Any] = load_pinned_tokenizer,
    generator: Callable[..., ThinkingPrediction] = generate_thinking_reaction,
    batch_generator: Callable[..., tuple[ThinkingPrediction, ...]] = (generate_thinking_reactions),
    torch_module: Any | None = None,
    progress_interval: int = 1,
) -> ThinkingExplorationReceipt:
    """Run both logical output shards through one model dispatched across two GPUs."""

    if split not in ("val", "test"):
        raise ThinkingInferenceError(f"thinking split must be val or test, got {split!r}")
    validate_thinking_profile(config.thinking_generation)
    environment_installer()
    checkpoint = checkpoint_inspector(checkpoint_dir, processed_path, config=config)
    records = record_loader(config, processed_path, split)
    selected = select_exploration_records(records, reaction_ids=reaction_ids, limit=limit)
    reaction_batch_size = THINKING_TEST_BATCH_SIZE if split == "test" else 1
    request_payload = _request_payload(
        config,
        checkpoint,
        split=split,
        records=selected,
        sample_count_per_reaction=sample_count_per_reaction,
        reaction_batch_size=reaction_batch_size,
    )
    request_payload["runtime_execution"] = "single-process-two-gpu-device-map"
    execution = dict(request_payload["generation_execution"])
    execution["synced_gpus"] = False
    request_payload["generation_execution"] = execution
    request_payload["batch_persistence"] = _batch_persistence_payload()
    run_root, request_sha256 = exploration_run_path(output_dir, request_payload)
    if (run_root / ".complete").is_file():
        return validate_exploration_artifact(run_root, request_sha256=request_sha256)
    run_root.mkdir(parents=True, exist_ok=True)

    scheduled = schedule_equal_rank_calls(selected)
    per_rank = len(scheduled) // WORLD_SIZE
    sample_count = int(request_payload["sample_count_per_reaction"])
    calls_per_rank = (per_rank + reaction_batch_size - 1) // reaction_batch_size
    persisted_batches: dict[tuple[int, int], list[dict[str, Any]]] = {}
    missing_batch_seen = False
    for logical_rank in range(WORLD_SIZE):
        start = logical_rank * per_rank
        rank_items = scheduled[start : start + per_rank]
        for batch_index, batch_start in enumerate(range(0, len(rank_items), reaction_batch_size)):
            items = rank_items[batch_start : batch_start + reaction_batch_size]
            path = exploration_batch_path(
                run_root,
                rank=logical_rank,
                batch_index=batch_index,
            )
            if path.exists() or path.is_symlink():
                if missing_batch_seen:
                    raise ThinkingInferenceError(
                        "persisted thinking batches are not one contiguous prefix"
                    )
                if path.is_symlink() or not path.is_file():
                    raise ThinkingInferenceError(f"invalid persisted thinking batch: {path}")
                rows = _read_jsonl(path)
                _validate_prediction_rows(
                    rows,
                    rank=logical_rank,
                    request_sha256=request_sha256,
                    checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
                    sample_count=sample_count,
                    expected_items=items,
                )
                persisted_batches[(logical_rank, batch_index)] = rows
            else:
                missing_batch_seen = True

    if len(persisted_batches) == calls_per_rank * WORLD_SIZE:
        for logical_rank in range(WORLD_SIZE):
            rows = [
                row
                for batch_index in range(calls_per_rank)
                for row in persisted_batches[(logical_rank, batch_index)]
            ]
            write_exploration_fragment(run_root, rank=logical_rank, rows=rows)
        print(f"[thinking:{split}] {len(scheduled)}/{len(scheduled)} recovered", flush=True)
        return finalize_exploration_artifact(
            run_root,
            request_payload,
            request_sha256=request_sha256,
        )

    if torch_module is None:
        import torch as torch_module
    reproducibility_configurer(torch_module, seed=config.seed, cuda_device=0)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(config.seed)
    model = model_loader(
        config,
        checkpoint,
        local_files_only=local_files_only,
    )
    tokenizer = tokenizer_loader(config, local_files_only=local_files_only)

    completed_records = 0
    for logical_rank in range(WORLD_SIZE):
        rows: list[dict[str, Any]] = []
        start = logical_rank * per_rank
        stop = start + per_rank
        rank_items = scheduled[start:stop]
        for batch_index, batch_start in enumerate(range(0, len(rank_items), reaction_batch_size)):
            items = rank_items[batch_start : batch_start + reaction_batch_size]
            persisted = persisted_batches.get((logical_rank, batch_index))
            if persisted is not None:
                batch_rows = persisted
            else:
                global_batch_index = logical_rank * calls_per_rank + batch_index
                _seed_local_thinking_batch(
                    torch_module,
                    seed=config.seed + global_batch_index,
                )
                if len(items) == 1:
                    predictions = (
                        generator(
                            model,
                            tokenizer,
                            items[0].record,
                            ordinal=items[0].ordinal,
                            profile=config.thinking_generation,
                            sample_count=sample_count,
                            max_input_length=config.model.max_sequence_length,
                            synchronized=False,
                        ),
                    )
                else:
                    predictions = batch_generator(
                        model,
                        tokenizer,
                        tuple(item.record for item in items),
                        ordinals=tuple(item.ordinal for item in items),
                        profile=config.thinking_generation,
                        sample_count=sample_count,
                        max_input_length=config.model.max_sequence_length,
                        synchronized=False,
                    )
                if len(predictions) != len(items):
                    raise ThinkingInferenceError(
                        "thinking batch generator returned the wrong number of predictions"
                    )
                batch_rows = []
                for item, prediction in zip(items, predictions, strict=True):
                    if len(prediction.raw_responses) != sample_count:
                        raise ThinkingInferenceError(
                            f"{item.record.reaction_id}: generator returned "
                            f"{len(prediction.raw_responses)} responses, expected {sample_count}"
                        )
                    if not item.is_dummy:
                        batch_rows.append(
                            prediction.to_json_dict(
                                split=split,
                                rank=logical_rank,
                                request_sha256=request_sha256,
                                checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
                            )
                        )
                write_exploration_batch(
                    run_root,
                    rank=logical_rank,
                    batch_index=batch_index,
                    rows=batch_rows,
                )
                persisted_batches[(logical_rank, batch_index)] = batch_rows
            rows.extend(batch_rows)
            completed_records += len(items)
            if completed_records % progress_interval == 0 or completed_records == len(scheduled):
                print(
                    f"[thinking:{split}] {completed_records}/{len(scheduled)}",
                    flush=True,
                )
        write_exploration_fragment(run_root, rank=logical_rank, rows=rows)
    return finalize_exploration_artifact(
        run_root,
        request_payload,
        request_sha256=request_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--reaction-id",
        action="append",
        nargs="+",
        default=[],
        metavar="ID",
        help="one or more exact reaction IDs; the flag may be repeated",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=f"limit selected IDs; defaults to {DEFAULT_EXPLORATION_LIMIT} without --reaction-id",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    reaction_ids = tuple(reaction_id for group in arguments.reaction_id for reaction_id in group)
    receipt = run_thinking_inference(
        load_config(arguments.config),
        processed_path=arguments.processed_path,
        checkpoint_dir=arguments.checkpoint_dir,
        output_dir=arguments.output_dir,
        split=arguments.split,
        reaction_ids=reaction_ids,
        limit=arguments.limit,
        local_files_only=True,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(receipt.payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_CLASS",
    "DEFAULT_EXPLORATION_LIMIT",
    "THINKING_EXPLORATION_SCHEMA_VERSION",
    "THINKING_PREDICTION_SCHEMA_VERSION",
    "THINKING_TEST_BATCH_SIZE",
    "ThinkingExplorationReceipt",
    "ThinkingInferenceError",
    "ThinkingPrediction",
    "ThinkingPromptEncoding",
    "encode_thinking_prompt",
    "exploration_run_path",
    "finalize_exploration_artifact",
    "generate_thinking_reaction",
    "generate_thinking_reactions",
    "load_exploration_records",
    "render_thinking_prompt",
    "run_local_thinking_inference",
    "run_thinking_inference",
    "schedule_equal_rank_calls",
    "select_exploration_records",
    "thinking_generation_kwargs",
    "validate_exploration_artifact",
    "validate_thinking_profile",
    "write_exploration_fragment",
]
