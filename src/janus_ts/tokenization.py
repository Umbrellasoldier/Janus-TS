"""Qwen chat formatting and loss-mask construction for Janus-TS.

The assistant turn deliberately starts with Qwen's non-thinking prefix.  The
system turn, user graph, assistant role marker, and empty ``<think>`` block are
all context-only.  Supervision starts at the first byte of the canonical
``<TS_EDGES>`` block and includes the terminating ``<|im_end|>`` token.

Nothing in this module truncates.  A sequence that exceeds the frozen context
limit is a data-contract failure, not something that may be repaired by
silently dropping graph edges or target tokens.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .constants import (
    MAX_SEQUENCE_LENGTH,
    QWEN_IM_END_TOKEN_ID,
    QWEN_IM_START_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
    SEED,
    SYSTEM_PROMPT,
)
from .representation import serialize_input, serialize_target
from .schema import Atom, Edge, MolecularState, ReactionRecord

IGNORE_INDEX = -100
PAD_TO_MULTIPLE_OF = 8
EMPTY_THINK_PREFIX = "<think>\n\n</think>\n\n"


class TokenizationContractError(ValueError):
    """Raised when tokenization would violate the frozen training protocol."""


class SequenceTooLongError(TokenizationContractError):
    """Raised instead of truncating an over-length reaction."""


class RecordCollection(Protocol):
    """Minimal interface shared by lists and memory-mapped HF datasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ReactionRecord | Mapping[str, Any]: ...


def _as_int_list(value: Any, *, context: str) -> list[int]:
    """Normalize tokenizer outputs without accepting batched encodings."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TokenizationContractError(f"{context} did not produce a token-ID sequence")
    if value and isinstance(value[0], (list, tuple)):
        raise TokenizationContractError(f"{context} unexpectedly produced a batched sequence")
    try:
        result = [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise TokenizationContractError(f"{context} contains non-integral token IDs") from exc
    if any(token_id < 0 for token_id in result):
        raise TokenizationContractError(f"{context} contains a negative token ID")
    return result


def _encode_without_special_tokens(tokenizer: Any, text: str, *, context: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError) as exc:
        raise TokenizationContractError(f"{context} tokenizer output has no input_ids") from exc
    return _as_int_list(input_ids, context=context)


def render_nonthinking_prompt(
    tokenizer: Any,
    user_content: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Render the two context turns plus Qwen's masked empty-think prefix."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:  # tokenizer/Jinja exceptions vary across versions
        raise TokenizationContractError(
            "Qwen chat template could not render enable_thinking=False"
        ) from exc
    if not isinstance(prompt, str):
        raise TokenizationContractError("Qwen chat template did not return text")
    if not prompt.endswith(EMPTY_THINK_PREFIX):
        raise TokenizationContractError(
            "non-thinking chat prompt must end with the exact empty <think> block"
        )
    return prompt


def validate_qwen_tokenizer(tokenizer: Any) -> None:
    """Hard-check the pinned Qwen tokenizer and non-thinking chat template."""

    failures: list[str] = []
    if getattr(tokenizer, "pad_token_id", None) != QWEN_PAD_TOKEN_ID:
        failures.append(
            f"pad_token_id={getattr(tokenizer, 'pad_token_id', None)!r}, "
            f"expected {QWEN_PAD_TOKEN_ID}"
        )
    for token, expected in (
        ("<|im_start|>", QWEN_IM_START_TOKEN_ID),
        ("<|im_end|>", QWEN_IM_END_TOKEN_ID),
    ):
        try:
            actual = int(tokenizer.convert_tokens_to_ids(token))
        except Exception as exc:  # pragma: no cover - defensive API guard
            raise TokenizationContractError(f"cannot resolve Qwen token {token!r}") from exc
        if actual != expected:
            failures.append(f"{token}={actual}, expected {expected}")
    if failures:
        raise TokenizationContractError("pinned Qwen tokenizer mismatch: " + "; ".join(failures))

    prompt = render_nonthinking_prompt(tokenizer, "MoleCode-TS/v1")
    prompt_ids = _encode_without_special_tokens(
        tokenizer, prompt, context="Qwen chat-template validation"
    )
    if prompt_ids.count(QWEN_IM_START_TOKEN_ID) != 3:
        raise TokenizationContractError(
            "non-thinking prompt must contain system, user, and assistant im_start tokens"
        )
    if prompt_ids.count(QWEN_IM_END_TOKEN_ID) != 2:
        raise TokenizationContractError(
            "non-thinking prompt must close exactly the system and user turns"
        )


def reaction_record_from_mapping(raw: Mapping[str, Any]) -> ReactionRecord:
    """Rehydrate a :class:`ReactionRecord` from a memory-mapped Arrow row."""

    def atom(value: Mapping[str, Any]) -> Atom:
        return Atom(
            atom_id=int(value["atom_id"]),
            atomic_number=int(value["atomic_number"]),
            symbol=str(value["symbol"]),
            formal_charge=int(value.get("formal_charge", 0)),
            radical_electrons=int(value.get("radical_electrons", 0)),
            stereo=value.get("stereo"),
        )

    def edge(value: Mapping[str, Any]) -> Edge:
        return Edge(
            atom_i=int(value["atom_i"]),
            atom_j=int(value["atom_j"]),
            bond_order=float(value["bond_order"]),
            stereo=value.get("stereo"),
        )

    def state(value: Mapping[str, Any]) -> MolecularState:
        return MolecularState(
            atoms=tuple(atom(item) for item in value["atoms"]),
            edges=tuple(edge(item) for item in value["edges"]),
            components=tuple(
                tuple(int(atom_id) for atom_id in component) for component in value["components"]
            ),
        )

    try:
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata is not a mapping")
        return ReactionRecord(
            reaction_id=str(raw["reaction_id"]),
            reactant=state(raw["reactant"]),
            product=state(raw["product"]),
            ts_edges=tuple(edge(item) for item in raw["ts_edges"]),
            split=str(raw["split"]),
            metadata=dict(metadata),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        reaction_id = raw.get("reaction_id", "<unknown>")
        raise TokenizationContractError(
            f"cannot rehydrate processed reaction {reaction_id!r}"
        ) from exc


def coerce_reaction_record(value: ReactionRecord | Mapping[str, Any]) -> ReactionRecord:
    if isinstance(value, ReactionRecord):
        return value
    if isinstance(value, Mapping):
        return reaction_record_from_mapping(value)
    raise TokenizationContractError(f"unsupported processed record type: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EncodedReaction:
    """One unpadded causal-LM example with auditable mask boundaries."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    reaction_id: str

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        if size == 0 or len(self.attention_mask) != size or len(self.labels) != size:
            raise TokenizationContractError(
                "encoded input, mask, and labels must have equal length"
            )
        if not 0 < self.prompt_length < size:
            raise TokenizationContractError("encoded reaction has an invalid supervision boundary")
        if any(value != IGNORE_INDEX for value in self.labels[: self.prompt_length]):
            raise TokenizationContractError("context tokens must all be masked")
        if tuple(self.labels[self.prompt_length :]) != tuple(self.input_ids[self.prompt_length :]):
            raise TokenizationContractError("all target tokens must be supervised")
        if self.labels[-1] != QWEN_IM_END_TOKEN_ID:
            raise TokenizationContractError("the final im_end token must be supervised")

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        return self.sequence_length - self.prompt_length

    def as_feature(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
        }


class QwenTrainingEncoder:
    """Encode graph records using the frozen Qwen non-thinking convention."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = MAX_SEQUENCE_LENGTH,
        seed: int = SEED,
        validate_tokenizer: bool = True,
    ) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise TokenizationContractError("max_length must be a positive integer")
        if seed != SEED:
            raise TokenizationContractError(f"training seed must remain frozen at {SEED}")
        if validate_tokenizer:
            validate_qwen_tokenizer(tokenizer)
        elif getattr(tokenizer, "pad_token_id", QWEN_PAD_TOKEN_ID) != QWEN_PAD_TOKEN_ID:
            raise TokenizationContractError("test tokenizer has the wrong pad token")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.seed = int(seed)

    def encode(
        self,
        record: ReactionRecord | Mapping[str, Any],
        *,
        training: bool,
        epoch: int = 0,
    ) -> EncodedReaction:
        reaction = coerce_reaction_record(record)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TokenizationContractError("epoch must be a non-negative integer")

        user_content = serialize_input(
            reaction,
            training=training,
            epoch=epoch,
            seed=self.seed,
        )
        target = serialize_target(reaction.ts_edges)
        prompt = render_nonthinking_prompt(self.tokenizer, user_content)
        prompt_ids = _encode_without_special_tokens(
            self.tokenizer,
            prompt,
            context=f"{reaction.reaction_id} prompt",
        )
        target_ids = _encode_without_special_tokens(
            self.tokenizer,
            target,
            context=f"{reaction.reaction_id} target",
        )
        if not target_ids:
            raise TokenizationContractError(f"{reaction.reaction_id}: empty target encoding")

        input_ids = tuple([*prompt_ids, *target_ids, QWEN_IM_END_TOKEN_ID])
        if len(input_ids) > self.max_length:
            raise SequenceTooLongError(
                f"{reaction.reaction_id}: tokenized length {len(input_ids)} exceeds "
                f"the hard limit {self.max_length}; truncation is forbidden"
            )
        prompt_length = len(prompt_ids)
        labels = tuple([IGNORE_INDEX] * prompt_length + target_ids + [QWEN_IM_END_TOKEN_ID])
        return EncodedReaction(
            input_ids=input_ids,
            attention_mask=(1,) * len(input_ids),
            labels=labels,
            prompt_length=prompt_length,
            reaction_id=reaction.reaction_id,
        )


class EpochAwareTokenizedDataset(torch.utils.data.Dataset):
    """On-the-fly epoch-aware encoding over an immutable processed dataset.

    ``set_epoch`` must be called before the epoch iterator is constructed.  As
    serialization is a pure function of ``(seed, epoch, reaction_id, section)``,
    resuming and skipping already-consumed batches reproduces their original
    text exactly and does not depend on worker or rank RNG state.
    """

    def __init__(
        self,
        records: RecordCollection,
        encoder: QwenTrainingEncoder,
        *,
        training: bool,
    ) -> None:
        self.records = records
        self.encoder = encoder
        self.training = bool(training)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TokenizationContractError("logical dataset epoch must be a non-negative int")
        self._epoch = epoch

    def encoded(self, index: int) -> EncodedReaction:
        return self.encoder.encode(
            self.records[index],
            training=self.training,
            epoch=self._epoch if self.training else 0,
        )

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.encoded(index).as_feature()


@dataclass(frozen=True, slots=True)
class LengthAudit:
    examples: int
    logical_epochs: int
    maximum_length: int
    maximum_reaction_id: str
    supervised_tokens: int


def audit_dataset_lengths(
    dataset: EpochAwareTokenizedDataset,
    *,
    logical_epochs: int = 1,
) -> LengthAudit:
    """Eagerly prove the no-truncation gate for every requested epoch."""

    if (
        isinstance(logical_epochs, bool)
        or not isinstance(logical_epochs, int)
        or logical_epochs <= 0
    ):
        raise TokenizationContractError("logical_epochs must be a positive integer")
    epochs = range(logical_epochs) if dataset.training else range(1)
    old_epoch = dataset.epoch
    maximum_length = -1
    maximum_reaction_id = ""
    supervised_tokens = 0
    try:
        for epoch in epochs:
            dataset.set_epoch(epoch)
            for index in range(len(dataset)):
                encoded = dataset.encoded(index)
                supervised_tokens += encoded.supervised_tokens
                if encoded.sequence_length > maximum_length:
                    maximum_length = encoded.sequence_length
                    maximum_reaction_id = encoded.reaction_id
    finally:
        dataset.set_epoch(old_epoch)
    return LengthAudit(
        examples=len(dataset),
        logical_epochs=len(tuple(epochs)),
        maximum_length=max(maximum_length, 0),
        maximum_reaction_id=maximum_reaction_id,
        supervised_tokens=supervised_tokens,
    )


@dataclass(slots=True)
class CausalLMCollator:
    """Right-pad causal examples dynamically to a multiple of eight."""

    pad_token_id: int = QWEN_PAD_TOKEN_ID
    label_pad_token_id: int = IGNORE_INDEX
    pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF
    max_length: int = MAX_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.pad_token_id != QWEN_PAD_TOKEN_ID:
            raise TokenizationContractError(
                f"pad_token_id must remain frozen at {QWEN_PAD_TOKEN_ID}"
            )
        if self.label_pad_token_id != IGNORE_INDEX:
            raise TokenizationContractError(f"labels must pad with {IGNORE_INDEX}")
        if self.pad_to_multiple_of != PAD_TO_MULTIPLE_OF:
            raise TokenizationContractError(
                f"padding multiple must remain frozen at {PAD_TO_MULTIPLE_OF}"
            )
        if self.max_length <= 0 or self.max_length % self.pad_to_multiple_of:
            raise TokenizationContractError(
                "max_length must be positive and divisible by the padding multiple"
            )

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise TokenizationContractError("cannot collate an empty batch")
        lengths: list[int] = []
        for feature in features:
            try:
                lengths_here = tuple(
                    len(feature[name]) for name in ("input_ids", "attention_mask", "labels")
                )
            except KeyError as exc:
                raise TokenizationContractError(f"batch feature is missing {exc.args[0]}") from exc
            if len(set(lengths_here)) != 1 or lengths_here[0] == 0:
                raise TokenizationContractError(
                    "each feature must contain equally sized, non-empty input_ids/mask/labels"
                )
            if any(int(value) != 1 for value in feature["attention_mask"]):
                raise TokenizationContractError(
                    "unpadded dataset features must have an all-one attention mask"
                )
            if int(feature["labels"][-1]) != QWEN_IM_END_TOKEN_ID:
                raise TokenizationContractError(
                    "each unpadded feature must supervise a final im_end token"
                )
            if any(
                int(label) != IGNORE_INDEX and int(label) != int(token_id)
                for token_id, label in zip(feature["input_ids"], feature["labels"], strict=True)
            ):
                raise TokenizationContractError(
                    "every non-masked causal label must equal its input token"
                )
            if lengths_here[0] > self.max_length:
                raise SequenceTooLongError(
                    f"collator received length {lengths_here[0]} above {self.max_length}"
                )
            lengths.append(lengths_here[0])

        longest = max(lengths)
        padded_length = math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of
        if padded_length > self.max_length:
            raise SequenceTooLongError(f"dynamic padding would exceed hard limit {self.max_length}")

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for feature, length in zip(features, lengths, strict=True):
            padding = padded_length - length
            batch_input_ids.append(
                [*map(int, feature["input_ids"]), *([self.pad_token_id] * padding)]
            )
            batch_attention_mask.append([*map(int, feature["attention_mask"]), *([0] * padding)])
            batch_labels.append(
                [*map(int, feature["labels"]), *([self.label_pad_token_id] * padding)]
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
