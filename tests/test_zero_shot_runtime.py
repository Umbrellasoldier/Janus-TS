from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from janus_ts.artifacts import mark_complete, sha256_file, write_json
from janus_ts.config import load_config
from janus_ts.formal_eval_runtime import PreparedFormalData
from janus_ts.generation import (
    EVALUATION_SCHEMA_VERSION,
    TestEvaluationLease,
    merged_predictions_path,
    metrics_path,
)
from janus_ts.zero_shot_runtime import (
    assert_frozen_bf16_qwen,
    assert_zero_shot_engine_precision,
    build_zero_shot_baseline,
    load_zero3_zero_shot_model,
    run_zero_shot_evaluation,
)


def _processed(tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    mark_complete(root, {"fingerprint": "d" * 64})
    return root


def test_zero_shot_identity_freezes_raw_model_prompt_and_generation(tmp_path: Path) -> None:
    config = load_config("configs/transition1x.yaml")
    baseline = build_zero_shot_baseline(
        config,
        _processed(tmp_path),
        run_fingerprint="f" * 64,
    )

    assert baseline.epoch == baseline.global_step == 0
    assert baseline.protocol["adapter"] is None
    assert baseline.protocol["training_updates"] == 0
    assert baseline.protocol["few_shot_examples"] == 0
    assert baseline.protocol["enable_thinking"] is False
    assert baseline.protocol["generation"]["num_beams"] == 10
    assert baseline.generation_identity().checkpoint_fingerprint == baseline.model_fingerprint


class _TinyRawQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.config = SimpleNamespace(use_cache=True, pad_token_id=None, eos_token_id=None)
        self.generation_config = SimpleNamespace(
            use_cache=True,
            pad_token_id=None,
            eos_token_id=None,
        )


def test_zero_shot_loader_never_adds_an_adapter_or_trainable_parameter(tmp_path: Path) -> None:
    config = load_config("configs/transition1x.yaml")
    baseline = build_zero_shot_baseline(
        config,
        _processed(tmp_path),
        run_fingerprint="f" * 64,
    )
    model = _TinyRawQwen()
    events: list[str] = []

    loaded = load_zero3_zero_shot_model(
        config,
        baseline,
        object(),
        base_loader=lambda **kwargs: events.append("base") or model,
        zero3_validator=lambda _arguments: events.append("zero3"),
        precision_validator=lambda _arguments, candidate: {
            "parameters": assert_frozen_bf16_qwen(candidate)
        },
    )

    assert loaded is model
    assert events == ["zero3", "base"]
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert loaded.training is False
    assert loaded.config.use_cache is False


def test_zero_shot_engine_requires_frozen_bf16_zero3_shards() -> None:
    model = _TinyRawQwen()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.ds_id = id(parameter)
        parameter.ds_numel = parameter.numel()
        parameter.ds_tensor = parameter.detach().clone()
    engine = SimpleNamespace(
        module=model,
        zero_optimization_stage=lambda: 3,
        bfloat16_enabled=lambda: True,
        fp16_enabled=lambda: False,
        torch_autocast_enabled=lambda: False,
    )
    report = assert_zero_shot_engine_precision(engine)
    assert report["trainable_parameters"] == 0
    assert report["zero_stage"] == 3


def test_zero_shot_runtime_uses_test_role_without_selection_proof(tmp_path: Path) -> None:
    config = load_config("configs/transition1x.yaml")
    processed = _processed(tmp_path)
    output = tmp_path / "evaluation"
    events: list[str] = []
    trainer = SimpleNamespace()

    def generation_runner(
        _model: Any,
        _tokenizer: Any,
        _records: Any,
        identity: Any,
        **kwargs: Any,
    ) -> Any:
        events.append("generate")
        assert kwargs["selection_proof_path"] is None
        assert kwargs["test_role"] == "frozen_zero_shot"
        predictions = merged_predictions_path(output, identity)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_text("{}\n", encoding="utf-8")
        metrics = metrics_path(output, identity)
        write_json(
            metrics,
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                **identity.to_json_dict(),
                "predictions_sha256": sha256_file(predictions),
                "eval_loss": None,
                "evaluation": {},
            },
        )
        lease = TestEvaluationLease(output, identity)
        lease.acquire()
        lease.complete(predictions_path=predictions, metrics_path_value=metrics)
        return SimpleNamespace(predictions_path=predictions, metrics_path=metrics)

    receipt = run_zero_shot_evaluation(
        config,
        processed_path=processed,
        output_dir=output,
        run_fingerprint="f" * 64,
        environment_installer=lambda: events.append("environment"),
        context_loader=lambda: events.append("context") or SimpleNamespace(local_rank=0),
        arguments_factory=lambda *_args, **_kwargs: events.append("arguments") or object(),
        context_validator=lambda *_args: events.append("context_valid"),
        reproducibility_configurer=lambda *_args, **_kwargs: events.append("seed"),
        model_loader=lambda *_args, **_kwargs: events.append("model") or object(),
        tokenizer_loader=lambda *_args, **_kwargs: events.append("tokenizer") or object(),
        data_preparer=lambda *_args, **_kwargs: PreparedFormalData(
            records=(), eval_dataset=None, collator=object()
        ),
        trainer_builder=lambda *_args, **_kwargs: events.append("trainer") or trainer,
        inference_initializer=lambda value: events.append("inference") or value,
        module_resolver=lambda _value: events.append("resolve") or object(),
        generation_runner=generation_runner,
        torch_module=object(),
    )

    assert receipt is not None
    assert receipt.path.is_file()
    assert events.index("arguments") < events.index("model") < events.index("generate")
