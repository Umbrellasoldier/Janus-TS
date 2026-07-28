from __future__ import annotations

from typing import Any

import pytest
import torch

from janus_ts.constants import (
    QWEN_IM_END_TOKEN_ID,
    QWEN_IM_START_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)
from janus_ts.representation import serialize_target
from janus_ts.schema import Atom, Edge, MolecularState, ReactionRecord
from janus_ts.tokenization import (
    EMPTY_THINK_PREFIX,
    IGNORE_INDEX,
    CausalLMCollator,
    EpochAwareTokenizedDataset,
    QwenTrainingEncoder,
    SequenceTooLongError,
    TokenizationContractError,
    audit_dataset_lengths,
    reaction_record_from_mapping,
    validate_qwen_tokenizer,
)


class FakeTokenizer:
    pad_token_id = QWEN_PAD_TOKEN_ID

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None

    @staticmethod
    def _encode(text: str) -> list[int]:
        replacements = {
            "<|im_start|>": chr(0xE000),
            "<|im_end|>": chr(0xE001),
        }
        for marker, replacement in replacements.items():
            text = text.replace(marker, replacement)
        result = []
        for char in text:
            if char == chr(0xE000):
                result.append(QWEN_IM_START_TOKEN_ID)
            elif char == chr(0xE001):
                result.append(QWEN_IM_END_TOKEN_ID)
            else:
                result.append(ord(char) + 1)
        return result

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|im_start|>": QWEN_IM_START_TOKEN_ID,
            "<|im_end|>": QWEN_IM_END_TOKEN_ID,
        }[token]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.last_messages = messages
        rendered = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        rendered += "<|im_start|>assistant\n" + EMPTY_THINK_PREFIX
        return self._encode(rendered) if tokenize else rendered

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        return {"input_ids": self._encode(text)}


def reaction_record() -> ReactionRecord:
    atoms = (
        Atom(0, 6, "C"),
        Atom(1, 8, "O"),
        Atom(2, 1, "H"),
        Atom(3, 1, "H"),
    )
    reactant = MolecularState(
        atoms=atoms,
        edges=(Edge(0, 1, 1.0), Edge(0, 2, 1.0), Edge(1, 3, 1.0)),
        components=((0, 1, 2, 3),),
    )
    product = MolecularState(
        atoms=atoms,
        edges=(Edge(0, 1, 1.0), Edge(0, 3, 1.0), Edge(1, 2, 1.0)),
        components=((0, 1, 2, 3),),
    )
    return ReactionRecord(
        reaction_id="rxn_test",
        reactant=reactant,
        product=product,
        ts_edges=(Edge(0, 1, 1.5), Edge(0, 2, 0.5), Edge(1, 3, 0.5)),
        split="train",
    )


def test_qwen_nonthinking_mask_supervises_only_target_through_im_end():
    tokenizer = FakeTokenizer()
    validate_qwen_tokenizer(tokenizer)
    encoder = QwenTrainingEncoder(tokenizer)
    encoded = encoder.encode(reaction_record(), training=True, epoch=0)

    assert tokenizer.last_messages is not None
    assert [item["role"] for item in tokenizer.last_messages] == ["system", "user"]
    assert all(label == IGNORE_INDEX for label in encoded.labels[: encoded.prompt_length])
    expected_target = tokenizer._encode(serialize_target(reaction_record().ts_edges))
    assert list(encoded.labels[encoded.prompt_length :]) == [
        *expected_target,
        QWEN_IM_END_TOKEN_ID,
    ]
    assert encoded.labels[-1] == QWEN_IM_END_TOKEN_ID
    assert encoded.supervised_tokens == len(expected_target) + 1


def test_over_length_is_a_hard_error_not_truncation():
    encoder = QwenTrainingEncoder(FakeTokenizer(), max_length=32)
    with pytest.raises(SequenceTooLongError, match="truncation is forbidden"):
        encoder.encode(reaction_record(), training=True, epoch=0)


def test_dynamic_collator_right_pads_to_multiple_of_eight():
    collator = CausalLMCollator(max_length=32)
    batch = collator(
        [
            {
                "input_ids": [1, 2, QWEN_IM_END_TOKEN_ID],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, QWEN_IM_END_TOKEN_ID],
            },
            {
                "input_ids": [4, 5, 6, 7, QWEN_IM_END_TOKEN_ID],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [-100, -100, 6, 7, QWEN_IM_END_TOKEN_ID],
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"][0, 3:].tolist() == [QWEN_PAD_TOKEN_ID] * 5
    assert batch["attention_mask"][0, 3:].tolist() == [0] * 5
    assert batch["labels"][0, 3:].tolist() == [IGNORE_INDEX] * 5


def test_epoch_aware_dataset_is_stateless_and_auditable(monkeypatch):
    calls: list[tuple[bool, int, int]] = []

    def fake_serialize_input(record, *, training, epoch, seed):
        calls.append((training, epoch, seed))
        return f"MoleCode-TS/v1 epoch={epoch}"

    monkeypatch.setattr("janus_ts.tokenization.serialize_input", fake_serialize_input)
    encoder = QwenTrainingEncoder(FakeTokenizer())
    dataset = EpochAwareTokenizedDataset([reaction_record()], encoder, training=True)
    dataset.set_epoch(3)

    first = dataset[0]
    second = dataset[0]
    assert first == second
    assert calls[-2:] == [(True, 3, 42), (True, 3, 42)]

    audit = audit_dataset_lengths(dataset, logical_epochs=2)
    assert audit.examples == 1
    assert audit.logical_epochs == 2
    assert audit.maximum_reaction_id == "rxn_test"
    assert dataset.epoch == 3


def test_arrow_mapping_round_trip():
    original = reaction_record()
    recovered = reaction_record_from_mapping(original.to_dict())
    assert recovered == original


def test_collator_rejects_bad_shapes_and_oversize():
    collator = CausalLMCollator(max_length=8)
    with pytest.raises(TokenizationContractError, match="equally sized"):
        collator([{"input_ids": [1], "attention_mask": [1, 1], "labels": [1]}])
    with pytest.raises(SequenceTooLongError):
        collator(
            [
                {
                    "input_ids": [*range(8), QWEN_IM_END_TOKEN_ID],
                    "attention_mask": [1] * 9,
                    "labels": [*range(8), QWEN_IM_END_TOKEN_ID],
                }
            ]
        )
