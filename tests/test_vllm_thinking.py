from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from janus_ts.config import load_config
from janus_ts.constants import QWEN_IM_END_TOKEN_ID
from janus_ts.schema import Atom, Edge, MolecularState, ReactionRecord
from janus_ts.vllm_thinking import (
    VLLM_ENVIRONMENT,
    VLLM_REQUEST_CONCURRENCY,
    VllmCompletion,
    VllmEndpoint,
    rewrite_safetensors_lora_prefix,
    run_vllm_thinking_inference,
)


def _record(reaction_id: str) -> ReactionRecord:
    atoms = (Atom(0, 6, "C"), Atom(1, 8, "O"))
    state = MolecularState(
        atoms=atoms,
        edges=(Edge(0, 1, 1.0),),
        components=((0, 1),),
    )
    return ReactionRecord(
        reaction_id=reaction_id,
        reactant=state,
        product=state,
        ts_edges=(Edge(0, 1, 1.5),),
        split="test",
    )


class _Tokenizer:
    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(character) + 1 for character in text]

    @staticmethod
    def decode(values: list[int], **kwargs: Any) -> str:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(
            "<|im_end|>" if value == QWEN_IM_END_TOKEN_ID else chr(value - 1) for value in values
        )

    @staticmethod
    def apply_chat_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is True
        rendered = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        return rendered + "<|im_start|>assistant\n<think>\n"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": self.encode(text)}


def _checkpoint(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        path=tmp_path / "checkpoint",
        checkpoint_fingerprint="c" * 64,
        data_fingerprint="d" * 64,
        run_fingerprint="r" * 64,
        model_fingerprint="m" * 64,
        epoch=5,
        global_step=2490,
    )


def test_safetensors_prefix_rewrite_preserves_tensor_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "destination.safetensors"
    tensors = {
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.arange(
            6, dtype=torch.bfloat16
        ).reshape(2, 3),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.arange(
            8, dtype=torch.bfloat16
        ).reshape(4, 2),
    }
    save_file(tensors, source, metadata={"format": "pt"})

    assert rewrite_safetensors_lora_prefix(source, destination) == 2

    converted = load_file(destination)
    expected = {
        name.replace(
            "base_model.model.model.layers.",
            "base_model.model.model.language_model.layers.",
            1,
        ): value
        for name, value in tensors.items()
    }
    assert set(converted) == set(expected)
    assert all(torch.equal(converted[name], value) for name, value in expected.items())
    with safe_open(destination, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {"format": "pt"}


def test_vllm_runtime_uses_concurrency_and_persists_each_request(tmp_path: Path) -> None:
    config = load_config("configs/transition1x.yaml")
    runtime = config.runtime.model_copy(update={"local_cache_root": tmp_path / "cache"})
    counts = {**config.data.expected_retained_counts, "test": 4}
    data = config.data.model_copy(update={"expected_retained_counts": counts})
    config = config.model_copy(update={"runtime": runtime, "data": data})
    metadata = (
        runtime.local_cache_root
        / "venvs"
        / VLLM_ENVIRONMENT
        / "lib/python3.12/site-packages/vllm-0.25.1.dist-info/METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: vllm\nVersion: 0.25.1\n", encoding="utf-8")
    records = tuple(_record(f"rxn{index:04d}") for index in range(4))
    tokenizer = _Tokenizer()
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    @contextmanager
    def server_factory(*_args: Any, **_kwargs: Any):
        log = tmp_path / "server" / "server.log"
        log.parent.mkdir()
        log.write_text("ready\n", encoding="utf-8")
        yield VllmEndpoint("http://unused", "base", "base", log)

    def requester(
        _endpoint: VllmEndpoint,
        input_ids: tuple[int, ...],
        _profile: Any,
        **_kwargs: Any,
    ) -> VllmCompletion:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        response = tokenizer.encode(
            "reasoning</think>\n<TS_EDGES>\na0 --[bo=1.5]-- a1\n</TS_EDGES>"
        )
        return VllmCompletion(tuple(input_ids), tuple(response), "stop")

    receipt = run_vllm_thinking_inference(
        config,
        processed_path=tmp_path / "processed",
        checkpoint=_checkpoint(tmp_path),
        output_dir=tmp_path / "output",
        split="test",
        limit=4,
        portable_adapter=None,
        environment_installer=lambda: None,
        record_loader=lambda *_args, **_kwargs: records,
        tokenizer_loader=lambda *_args, **_kwargs: tokenizer,
        server_factory=server_factory,
        completion_requester=requester,
        server_validator=lambda *_args, **_kwargs: None,
    )

    assert maximum_active == 2
    assert receipt.payload["runtime_execution"] == "vllm-openai-compatible-two-gpu"
    assert receipt.payload["inference_engine"]["request_concurrency"] == (VLLM_REQUEST_CONCURRENCY)
    assert len(list(receipt.path.glob("batches/rank-*/*.jsonl"))) == 4
    rows = [
        json.loads(line) for line in (receipt.path / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["reaction_id"] for row in rows] == [
        "rxn0000",
        "rxn0001",
        "rxn0002",
        "rxn0003",
    ]
