from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import janus_ts.thinking_inference as thinking
from janus_ts.config import load_config
from janus_ts.constants import QWEN_IM_END_TOKEN_ID, QWEN_PAD_TOKEN_ID
from janus_ts.distributed import TorchrunContext
from janus_ts.representation import serialize_input
from janus_ts.schema import Atom, Edge, MolecularState, ReactionRecord
from janus_ts.thinking_inference import (
    ARTIFACT_CLASS,
    THINKING_EXPLORATION_SCHEMA_VERSION,
    ThinkingInferenceError,
    ThinkingPrediction,
    encode_thinking_prompt,
    finalize_exploration_artifact,
    generate_thinking_reaction,
    run_thinking_inference,
    schedule_equal_rank_calls,
    select_exploration_records,
    thinking_generation_kwargs,
    validate_exploration_artifact,
    validate_thinking_profile,
    write_exploration_fragment,
)
from janus_ts.tokenization import EMPTY_THINK_PREFIX


def _record(reaction_id: str, *, split: str = "val") -> ReactionRecord:
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


class _ThinkingTokenizer:
    def __init__(self, *, empty_prefix: bool = False) -> None:
        self.empty_prefix = empty_prefix
        self.last_messages: list[dict[str, str]] | None = None
        self.enable_thinking: bool | None = None

    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(character) + 1 for character in text]

    @staticmethod
    def decode(values: list[int]) -> str:
        result = []
        for value in values:
            if value == QWEN_IM_END_TOKEN_ID:
                result.append("<|im_end|>")
            elif value == QWEN_PAD_TOKEN_ID:
                result.append("<|endoftext|>")
            else:
                result.append(chr(value - 1))
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
        self.last_messages = messages
        self.enable_thinking = enable_thinking
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        suffix = EMPTY_THINK_PREFIX if self.empty_prefix else "<think>\n"
        return rendered + "<|im_start|>assistant\n" + suffix

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_attention_mask": False,
        }
        return {"input_ids": self.encode(text)}

    def batch_decode(self, rows: list[list[int]], **kwargs: Any) -> list[str]:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return [self.decode(row) for row in rows]


class _ThinkingModel:
    device = torch.device("cpu")

    def __init__(self, tokenizer: _ThinkingTokenizer, response: str) -> None:
        self.tokenizer = tokenizer
        self.response = response
        self.kwargs: dict[str, Any] | None = None

    def generate(self, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.kwargs = kwargs
        continuation = [*self.tokenizer.encode(self.response), QWEN_IM_END_TOKEN_ID]
        sequence = [*input_ids[0].tolist(), *continuation]
        return torch.tensor(
            [sequence] * int(kwargs["num_return_sequences"]),
            dtype=torch.long,
        )


def _checkpoint(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        path=tmp_path / "epoch-00003",
        checkpoint_fingerprint="c" * 64,
        data_fingerprint="d" * 64,
        run_fingerprint="a" * 64,
        model_fingerprint="b" * 64,
        epoch=3,
        global_step=150,
    )


def test_thinking_prompt_uses_same_molecode_and_never_empty_think_prefix() -> None:
    tokenizer = _ThinkingTokenizer()
    record = _record("rxn0001")

    encoded = encode_thinking_prompt(tokenizer, record, max_length=100_000)

    assert tokenizer.enable_thinking is True
    assert tokenizer.last_messages is not None
    assert tokenizer.last_messages[1]["content"] == serialize_input(
        record, training=False, epoch=999
    )
    assert encoded.prompt_text.endswith("<think>\n")
    assert EMPTY_THINK_PREFIX not in encoded.prompt_text

    with pytest.raises(ThinkingInferenceError, match="empty <think>"):
        encode_thinking_prompt(_ThinkingTokenizer(empty_prefix=True), record, max_length=100_000)


def test_thinking_profile_is_exact_and_presence_penalty_is_record_only() -> None:
    profile = load_config("configs/transition1x.yaml").thinking_generation

    assert thinking_generation_kwargs(profile) == {
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8192,
        "eos_token_id": QWEN_IM_END_TOKEN_ID,
        "pad_token_id": QWEN_PAD_TOKEN_ID,
        "use_cache": True,
        "synced_gpus": True,
    }
    assert "presence_penalty" not in thinking_generation_kwargs(profile)

    changed = profile.model_copy(update={"temperature": 0.7})
    with pytest.raises(ThinkingInferenceError, match="temperature"):
        validate_thinking_profile(changed)


def test_sampled_generation_keeps_raw_thinking_response_and_exact_kwargs() -> None:
    config = load_config("configs/transition1x.yaml")
    tokenizer = _ThinkingTokenizer()
    model = _ThinkingModel(tokenizer, "reasoning\n</think>\n<TS_EDGES></TS_EDGES>")

    prediction = generate_thinking_reaction(
        model,
        tokenizer,
        _record("rxn0007"),
        ordinal=7,
        profile=config.thinking_generation,
        sample_count=10,
        max_input_length=100_000,
    )

    assert prediction.ordinal == 7
    assert (
        prediction.raw_responses == ("reasoning\n</think>\n<TS_EDGES></TS_EDGES><|im_end|>",) * 10
    )
    assert model.kwargs is not None
    attention_mask = model.kwargs.pop("attention_mask")
    assert torch.all(attention_mask == 1)
    assert model.kwargs == thinking_generation_kwargs(
        config.thinking_generation,
        sample_count=10,
    )


def test_selection_is_finite_by_default_and_odd_counts_get_one_dummy() -> None:
    records = tuple(_record(f"rxn{index:04d}") for index in range(1, 6))

    selected = select_exploration_records(records)
    assert [record.reaction_id for record in selected] == ["rxn0001", "rxn0002"]

    explicit = select_exploration_records(records, reaction_ids=("rxn0004", "rxn0002", "rxn0001"))
    scheduled = schedule_equal_rank_calls(explicit)
    assert [item.record.reaction_id for item in scheduled] == [
        "rxn0004",
        "rxn0002",
        "rxn0001",
        "rxn0001",
    ]
    assert [item.is_dummy for item in scheduled] == [False, False, False, True]


def test_exploration_manifest_is_content_verified_and_formally_ineligible(
    tmp_path: Path,
) -> None:
    config = load_config("configs/transition1x.yaml")
    checkpoint = _checkpoint(tmp_path)
    records = tuple(_record(f"rxn{index:04d}") for index in range(1, 4))
    request = thinking._request_payload(config, checkpoint, split="val", records=records)
    root, request_sha256 = thinking.exploration_run_path(tmp_path, request)
    root.mkdir(parents=True)

    rows = []
    for ordinal, record in enumerate(records):
        rank = 0 if ordinal < 2 else 1
        row = ThinkingPrediction(
            reaction_id=record.reaction_id,
            ordinal=ordinal,
            atom_count=record.atom_count,
            prompt_sha256=str(ordinal) * 64,
            prompt_tokens=100 + ordinal,
            raw_responses=(f"response-{ordinal}",),
        ).to_json_dict(
            split="val",
            rank=rank,
            request_sha256=request_sha256,
            checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
        )
        rows.append(row)
    write_exploration_fragment(root, rank=0, rows=rows[:2])
    write_exploration_fragment(root, rank=1, rows=rows[2:])

    receipt = finalize_exploration_artifact(root, request, request_sha256=request_sha256)

    assert receipt.payload["schema_version"] == THINKING_EXPLORATION_SCHEMA_VERSION
    assert receipt.payload["artifact_class"] == ARTIFACT_CLASS
    assert receipt.payload["formal_eligible"] is False
    assert receipt.payload["affects_checkpoint_selection"] is False
    assert receipt.payload["metrics_computed"] is False
    assert receipt.payload["writes_formal_test_lease"] is False
    assert receipt.payload["writes_run_state"] is False
    assert receipt.payload["dummy_generate_calls"] == 1
    assert (
        receipt.payload["generation_profile"]["presence_penalty_forwarded_to_transformers"] is False
    )
    assert set(receipt.payload["payload_inventory"]) == {
        "rank-00000-of-00002.jsonl",
        "rank-00001-of-00002.jsonl",
        "predictions.jsonl",
    }
    predictions = [
        json.loads(line)
        for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["reaction_id"] for row in predictions] == [
        "rxn0001",
        "rxn0002",
        "rxn0003",
    ]

    (root / "predictions.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ThinkingInferenceError, match="failed verification"):
        validate_exploration_artifact(root, request_sha256=request_sha256)


def test_full_runtime_is_cpu_injectable_and_reuses_formal_restore_chain(
    tmp_path: Path,
) -> None:
    config = load_config("configs/transition1x.yaml")
    checkpoint = _checkpoint(tmp_path)
    records = (_record("rxn0001"), _record("rxn0002"))
    events: list[str] = []
    barrier_calls = 0

    def arguments_factory(_config: Any, *, output_dir: Path) -> Any:
        events.append("arguments")
        assert "thinking-exploratory" in output_dir.parts
        return SimpleNamespace(name="arguments")

    def model_loader(
        _config: Any,
        actual_checkpoint: Any,
        arguments: Any,
        *,
        local_files_only: bool,
    ) -> str:
        events.append("original-base-plus-portable")
        assert actual_checkpoint is checkpoint
        assert arguments.name == "arguments"
        assert local_files_only is True
        return "model"

    def generator(
        _model: Any,
        _tokenizer: Any,
        record: ReactionRecord,
        *,
        ordinal: int,
        **kwargs: Any,
    ) -> ThinkingPrediction:
        events.append(f"generate:{record.reaction_id}")
        assert kwargs["profile"] is config.thinking_generation
        assert kwargs["sample_count"] == 1
        return ThinkingPrediction(
            reaction_id=record.reaction_id,
            ordinal=ordinal,
            atom_count=record.atom_count,
            prompt_sha256="e" * 64,
            prompt_tokens=111,
            raw_responses=(f"thinking-{record.reaction_id}",),
        )

    def fake_barrier() -> None:
        nonlocal barrier_calls
        barrier_calls += 1
        if barrier_calls != 2:
            return
        rank_zero_fragments = list(
            (tmp_path / "outputs").glob("thinking-exploratory/*/val/*/rank-00000-of-00002.jsonl")
        )
        assert len(rank_zero_fragments) == 1
        root = rank_zero_fragments[0].parent
        rank_zero_row = json.loads(rank_zero_fragments[0].read_text(encoding="utf-8").strip())
        rank_one = dict(rank_zero_row)
        rank_one.update(
            {
                "rank": 1,
                "ordinal": 1,
                "reaction_id": "rxn0002",
                "sample_count": 1,
                "raw_responses": ["thinking-rxn0002"],
            }
        )
        write_exploration_fragment(root, rank=1, rows=[rank_one])

    receipt = run_thinking_inference(
        config,
        processed_path=tmp_path / "processed",
        checkpoint_dir=checkpoint.path,
        output_dir=tmp_path / "outputs",
        environment_installer=lambda: events.append("environment"),
        context_loader=lambda: TorchrunContext(0, 0, 2, 0),
        checkpoint_inspector=lambda *args, **kwargs: checkpoint,
        record_loader=lambda *args, **kwargs: records,
        arguments_factory=arguments_factory,
        reproducibility_configurer=lambda *args, **kwargs: events.append("seed"),
        context_validator=lambda *args, **kwargs: events.append("two-rank-zero3"),
        model_loader=model_loader,
        tokenizer_loader=lambda *args, **kwargs: "tokenizer",
        trainer_builder=lambda model, arguments, prepared: SimpleNamespace(
            model=model, arguments=arguments, prepared=prepared
        ),
        inference_initializer=lambda trainer: events.append("deepspeed-engine"),
        module_resolver=lambda trainer: "generation-proxy",
        generator=generator,
        barrier=fake_barrier,
        torch_module=SimpleNamespace(),
    )

    assert events.index("arguments") < events.index("original-base-plus-portable")
    assert events == [
        "environment",
        "arguments",
        "seed",
        "two-rank-zero3",
        "original-base-plus-portable",
        "deepspeed-engine",
        "generate:rxn0001",
    ]
    assert barrier_calls == 3
    assert receipt.payload["selection_count"] == 2
    assert receipt.payload["checkpoint_fingerprint"] == checkpoint.checkpoint_fingerprint

    recovered = run_thinking_inference(
        config,
        processed_path=tmp_path / "processed",
        checkpoint_dir=checkpoint.path,
        output_dir=tmp_path / "outputs",
        environment_installer=lambda: None,
        context_loader=lambda: TorchrunContext(0, 0, 2, 0),
        checkpoint_inspector=lambda *args, **kwargs: checkpoint,
        record_loader=lambda *args, **kwargs: records,
        arguments_factory=lambda *args, **kwargs: pytest.fail(
            "completed exploration must be recovered before model initialization"
        ),
    )
    assert recovered.path == receipt.path
