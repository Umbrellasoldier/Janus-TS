from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from janus_ts.constants import (
    QWEN_IM_END_TOKEN_ID,
    QWEN_IM_START_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)
from janus_ts.generation import (
    FORMAL_NUM_BEAMS,
    GenerationContractError,
    GenerationIdentity,
    GenerationRow,
    SelectionProof,
    TestAlreadyEvaluatedError,
    TestEvaluationLease,
    encode_formal_prompt,
    evaluate_generation_rows,
    formal_generation_kwargs,
    generate_reaction,
    load_merged_predictions,
    load_selection_proof,
    merge_generation_fragments,
    rank_shard_bounds,
    resolve_eval_loss,
    trainer_eval_loss_hook,
    write_generation_fragment,
    write_selection_proof,
)
from janus_ts.representation import serialize_input, serialize_target
from janus_ts.schema import Atom, Edge, MolecularState, ReactionRecord
from janus_ts.tokenization import EMPTY_THINK_PREFIX


class FakeTokenizer:
    pad_token_id = QWEN_PAD_TOKEN_ID

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None

    @staticmethod
    def encode(text: str) -> list[int]:
        text = text.replace("<|im_start|>", chr(0xE000))
        text = text.replace("<|im_end|>", chr(0xE001))
        result: list[int] = []
        for char in text:
            if char == chr(0xE000):
                result.append(QWEN_IM_START_TOKEN_ID)
            elif char == chr(0xE001):
                result.append(QWEN_IM_END_TOKEN_ID)
            else:
                result.append(ord(char) + 1)
        return result

    @staticmethod
    def decode(ids: list[int]) -> str:
        result: list[str] = []
        for token_id in ids:
            if token_id == QWEN_IM_START_TOKEN_ID:
                result.append("<|im_start|>")
            elif token_id == QWEN_IM_END_TOKEN_ID:
                result.append("<|im_end|>")
            elif token_id == QWEN_PAD_TOKEN_ID:
                result.append("<|endoftext|>")
            else:
                result.append(chr(token_id - 1))
        return "".join(result)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.last_messages = messages
        rendered = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        return rendered + "<|im_start|>assistant\n" + EMPTY_THINK_PREFIX

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["padding"] is False
        assert kwargs["truncation"] is False
        return {"input_ids": self.encode(text)}

    def batch_decode(self, rows: list[list[int]], **kwargs: Any) -> list[str]:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return [self.decode(row) for row in rows]


class FakeModel:
    def __init__(self, tokenizer: FakeTokenizer, beams: list[str]) -> None:
        self.tokenizer = tokenizer
        self.beams = beams
        self.device = torch.device("cpu")
        self.training = True
        self.last_kwargs: dict[str, Any] | None = None

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def generate(self, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.last_kwargs = kwargs
        continuations = [
            [*self.tokenizer.encode(beam), QWEN_IM_END_TOKEN_ID] for beam in self.beams
        ]
        longest = max(map(len, continuations))
        sequences = []
        for continuation in continuations:
            sequences.append(
                [
                    *input_ids[0].tolist(),
                    *continuation,
                    *([QWEN_PAD_TOKEN_ID] * (longest - len(continuation))),
                ]
            )
        return torch.tensor(sequences, dtype=torch.long)


def reaction_record(reaction_id: str = "rxn0001", *, split: str = "val") -> ReactionRecord:
    atoms = (Atom(0, 6, "C"), Atom(1, 8, "O"), Atom(2, 1, "H"))
    reactant = MolecularState(
        atoms=atoms,
        edges=(Edge(0, 1, 1.0), Edge(0, 2, 1.0)),
        components=((0, 1, 2),),
    )
    product = MolecularState(
        atoms=atoms,
        edges=(Edge(0, 1, 2.0), Edge(0, 2, 1.0)),
        components=((0, 1, 2),),
    )
    return ReactionRecord(
        reaction_id=reaction_id,
        reactant=reactant,
        product=product,
        ts_edges=(Edge(0, 1, 1.5), Edge(0, 2, 1.0)),
        split=split,
    )


def identity(split: str = "val") -> GenerationIdentity:
    return GenerationIdentity(
        split=split,
        data_fingerprint="a" * 64,
        run_fingerprint="b" * 64,
        checkpoint_fingerprint="c" * 64,
    )


def prediction_row(reaction: ReactionRecord, ordinal: int) -> GenerationRow:
    target = serialize_target(reaction.ts_edges) + "<|im_end|>"
    return GenerationRow(
        reaction_id=reaction.reaction_id,
        ordinal=ordinal,
        atom_count=reaction.atom_count,
        prompt_sha256="d" * 64,
        raw_beams=(target,) * FORMAL_NUM_BEAMS,
    )


def test_formal_prompt_and_generate_contract_retain_raw_duplicate_beams() -> None:
    tokenizer = FakeTokenizer()
    reaction = reaction_record()
    target = serialize_target(reaction.ts_edges)
    beams = [target, target, *(["<TS_EDGES>\n</TS_EDGES>"] * 8)]
    model = FakeModel(tokenizer, beams)

    prompt = encode_formal_prompt(tokenizer, reaction)
    row = generate_reaction(model, tokenizer, reaction, ordinal=7, synchronized=True)

    assert tokenizer.last_messages is not None
    assert tokenizer.last_messages[1]["content"] == serialize_input(
        reaction, training=False, epoch=999
    )
    assert prompt.prompt_text.endswith(EMPTY_THINK_PREFIX)
    assert row.ordinal == 7
    assert row.raw_beams[0] == row.raw_beams[1]
    assert row.raw_beams[0].endswith("<|im_end|>")
    assert "<|endoftext|>" not in row.raw_beams[0]
    assert model.last_kwargs is not None
    attention_mask = model.last_kwargs.pop("attention_mask")
    assert torch.all(attention_mask == 1)
    assert model.last_kwargs == formal_generation_kwargs(synchronized=True)


def test_generation_kwargs_are_the_frozen_beam_profile() -> None:
    assert formal_generation_kwargs() == {
        "num_beams": 10,
        "num_return_sequences": 10,
        "do_sample": False,
        "max_new_tokens": 512,
        "length_penalty": 1.0,
        "early_stopping": False,
        "eos_token_id": QWEN_IM_END_TOKEN_ID,
        "pad_token_id": QWEN_PAD_TOKEN_ID,
        "use_cache": True,
        "synced_gpus": True,
    }


def test_equal_contiguous_rank_halves_and_odd_count_rejection() -> None:
    assert rank_shard_bounds(994, rank=0) == (0, 497)
    assert rank_shard_bounds(994, rank=1) == (497, 994)
    with pytest.raises(GenerationContractError, match="equal halves"):
        rank_shard_bounds(995, rank=0)
    with pytest.raises(GenerationContractError, match="exactly 2 ranks"):
        rank_shard_bounds(994, rank=0, world_size=1)


def test_atomic_fragments_merge_in_expected_id_order_and_strict_reload(
    tmp_path: Path,
) -> None:
    reactions = tuple(reaction_record(f"rxn{index:04d}") for index in range(4))
    rows = tuple(prediction_row(reaction, index) for index, reaction in enumerate(reactions))
    formal_identity = identity()

    # Deliberately write rank one first; merged order must still follow the dataset.
    write_generation_fragment(tmp_path, formal_identity, rank=1, rows=rows[2:])
    write_generation_fragment(tmp_path, formal_identity, rank=0, rows=rows[:2])
    merged_path, merged = merge_generation_fragments(
        tmp_path,
        formal_identity,
        expected_reaction_ids=[reaction.reaction_id for reaction in reactions],
    )

    assert merged == rows
    assert merged_path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert (
        load_merged_predictions(
            merged_path,
            formal_identity,
            expected_reaction_ids=[reaction.reaction_id for reaction in reactions],
        )
        == rows
    )
    with pytest.raises(GenerationContractError, match="expected dataset order"):
        merge_generation_fragments(
            tmp_path,
            formal_identity,
            expected_reaction_ids=[
                reactions[1].reaction_id,
                reactions[0].reaction_id,
                *[reaction.reaction_id for reaction in reactions[2:]],
            ],
        )


def test_generation_rows_use_shared_strict_evaluator_and_aggregate() -> None:
    reactions = (reaction_record("rxn0001"), reaction_record("rxn0002"))
    rows = tuple(prediction_row(reaction, index) for index, reaction in enumerate(reactions))
    evaluations, report = evaluate_generation_rows(reactions, rows)
    assert len(evaluations) == 2
    assert report.reaction_count == 2
    assert report.at_k(1).metrics.connectivity_successes == 2
    assert report.at_k(10).metrics.exact_successes == 2


def test_selection_proof_and_test_lease_allow_only_one_completed_run(
    tmp_path: Path,
) -> None:
    test_identity = identity("test")
    proof = SelectionProof(
        data_fingerprint=test_identity.data_fingerprint,
        run_fingerprint=test_identity.run_fingerprint,
        selected_checkpoint_fingerprint=test_identity.checkpoint_fingerprint,
        selected_epoch=3,
    )
    proof_path = write_selection_proof(tmp_path / "selection.json", proof)
    assert load_selection_proof(proof_path, test_identity) == proof
    with pytest.raises(FileExistsError):
        write_selection_proof(proof_path, proof)

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    lease = TestEvaluationLease(tmp_path, test_identity)
    lease.acquire()
    lease.complete(predictions_path=predictions, metrics_path_value=metrics)
    assert lease.completion_path.is_file()
    assert not lease.claim_path.exists()
    with pytest.raises(TestAlreadyEvaluatedError):
        TestEvaluationLease(tmp_path, test_identity).acquire()


def test_eval_loss_hook_accepts_trainer_metric_and_rejects_nonfinite() -> None:
    class Trainer:
        def evaluate(self, *, eval_dataset: Any) -> dict[str, float]:
            assert eval_dataset == "validation"
            return {"eval_loss": 1.25}

    hook = trainer_eval_loss_hook(Trainer(), eval_dataset="validation")
    assert hook() == 1.25
    assert resolve_eval_loss(lambda: 0.5) == 0.5
    with pytest.raises(GenerationContractError, match="finite"):
        resolve_eval_loss(float("nan"))


def test_strict_fingerprints_and_model_output_shape_are_hard_errors() -> None:
    with pytest.raises(GenerationContractError, match="SHA256"):
        GenerationIdentity("val", "not-a-hash", "b" * 64, "c" * 64)

    tokenizer = FakeTokenizer()

    class BadModel(FakeModel):
        def generate(self, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return input_ids.repeat(9, 1)

    with pytest.raises(GenerationContractError, match="must return shape"):
        generate_reaction(BadModel(tokenizer, []), tokenizer, reaction_record(), ordinal=0)
