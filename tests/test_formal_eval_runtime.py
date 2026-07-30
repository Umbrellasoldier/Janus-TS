from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from janus_ts.artifacts import mark_complete, sha256_file, write_json
from janus_ts.checkpointing import CHECKPOINT_SCHEMA_VERSION
from janus_ts.config import ExperimentConfig, load_config
from janus_ts.formal_eval_runtime import (
    DeepSpeedGenerationProxy,
    DurableCheckpoint,
    FormalEvalRuntimeError,
    PreparedFormalData,
    _formal_inference_deepspeed_config,
    assert_formal_zero3_precision,
    inspect_durable_checkpoint,
    load_checkpoint_score,
    load_zero3_portable_model,
    run_formal_checkpoint_evaluation,
    runtime_receipt_path,
    validate_runtime_receipt,
)
from janus_ts.generation import (
    EVALUATION_SCHEMA_VERSION,
    TEST_COMPLETION_SCHEMA_VERSION,
    SelectionProof,
    TestEvaluationLease,
    merged_predictions_path,
    metrics_path,
    write_selection_proof,
)
from janus_ts.modeling import EXPECTED_PORTABLE_PARAMETERS


def _complete_processed(root: Path, data_fingerprint: str) -> Path:
    root.mkdir()
    mark_complete(root, {"fingerprint": data_fingerprint})
    return root


def _complete_checkpoint(
    root: Path,
    *,
    data_fingerprint: str,
    config_fingerprint: str,
    epoch: float = 3,
    global_step: int = 150,
    kind: str = "epoch",
) -> Path:
    adapter = root / "portable_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":64}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"portable-weights")
    inventory = {}
    for path in sorted(adapter.iterdir()):
        relative = path.relative_to(root).as_posix()
        inventory[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    mark_complete(
        root,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            "run_fingerprint": "a" * 64,
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
            "model_fingerprint": "b" * 64,
            "global_step": global_step,
            "epoch": float(epoch),
            "world_size": 2,
            "exclude_frozen_parameters": True,
            "pissa_initial_adapter_fingerprint": "e" * 64,
            "payload_inventory": inventory,
            **({"train_loss": 0.125} if kind == "train-loss" else {}),
        },
    )
    return root


def _formal_assets(
    tmp_path: Path,
    config: ExperimentConfig,
    *,
    epoch: int = 3,
) -> tuple[Path, Path, DurableCheckpoint]:
    processed = _complete_processed(tmp_path / "processed", "d" * 64)
    checkpoint_dir = _complete_checkpoint(
        tmp_path / "checkpoint",
        data_fingerprint="d" * 64,
        config_fingerprint=config.sha256,
        epoch=epoch,
    )
    checkpoint = inspect_durable_checkpoint(checkpoint_dir, processed, config=config)
    return processed, checkpoint_dir, checkpoint


def _fraction(value: int, denominator: int) -> dict[str, int | float]:
    from fractions import Fraction

    exact = Fraction(value, denominator)
    return {
        "numerator": exact.numerator,
        "denominator": exact.denominator,
        "value": float(exact),
    }


def test_formal_inference_removes_training_only_zero3_buffers() -> None:
    config = load_config("configs/transition1x.yaml")

    payload = _formal_inference_deepspeed_config(config)
    zero = payload["zero_optimization"]

    assert payload["bf16"]["enabled"] is True
    assert zero["stage"] == 3
    assert zero["offload_param"] == {"device": "none"}
    assert zero["overlap_comm"] is False
    assert zero["contiguous_gradients"] is False
    assert zero["allgather_bucket_size"] == 4_000_000
    assert zero["reduce_bucket_size"] == 4_000_000
    assert zero["stage3_prefetch_bucket_size"] == 4_000_000
    assert zero["stage3_param_persistence_threshold"] == 0


def _metrics_payload(
    checkpoint: DurableCheckpoint,
    split: str,
    predictions_sha256: str,
    *,
    eval_loss: float | None,
    with_score: bool = False,
) -> dict[str, Any]:
    evaluation: dict[str, Any] = {}
    if with_score:
        count = 994
        iou_sum = {"numerator": 1491, "denominator": 2, "value": 745.5}
        f1_sum = {"numerator": 3976, "denominator": 5, "value": 795.2}
        evaluation = {
            "metrics": {
                "@10": {
                    "count": count,
                    "connectivity": {**_fraction(800, count), "successes": 800},
                    "exact": {**_fraction(700, count), "successes": 700},
                    "edit_bond": {
                        **_fraction(1200, count),
                        "total": 1200,
                    },
                    "edge_iou": {
                        **_fraction(1491, 2 * count),
                        "sum": iou_sum,
                    },
                    "edge_f1": {
                        **_fraction(3976, 5 * count),
                        "sum": f1_sum,
                    },
                }
            }
        }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        **checkpoint.generation_identity(split).to_json_dict(),
        "predictions_sha256": predictions_sha256,
        "eval_loss": eval_loss,
        "evaluation": evaluation,
    }


def test_durable_checkpoint_identity_and_portable_payload_are_content_verified(
    tmp_path: Path,
) -> None:
    config = load_config("configs/transition1x.yaml")
    processed, checkpoint_dir, checkpoint = _formal_assets(tmp_path, config)

    assert checkpoint.epoch == 3
    assert checkpoint.global_step == 150
    assert checkpoint.checkpoint_fingerprint == sha256_file(checkpoint_dir / "manifest.json")
    assert checkpoint.portable_adapter_path == checkpoint_dir / "portable_adapter"

    weights = checkpoint.portable_adapter_path / "adapter_model.safetensors"
    weights.write_bytes(b"tampered")
    with pytest.raises(FormalEvalRuntimeError, match="size mismatch|digest mismatch"):
        inspect_durable_checkpoint(checkpoint_dir, processed, config=config)


def test_fractional_train_loss_checkpoint_is_valid_for_formal_evaluation(tmp_path):
    config = load_config("configs/transition1x.yaml")
    processed = _complete_processed(tmp_path / "processed", "d" * 64)
    checkpoint_dir = _complete_checkpoint(
        tmp_path / "checkpoint",
        data_fingerprint="d" * 64,
        config_fingerprint=config.sha256,
        epoch=2.75,
        global_step=1370,
        kind="train-loss",
    )

    checkpoint = inspect_durable_checkpoint(checkpoint_dir, processed, config=config)
    assert checkpoint.epoch == 2.75
    assert checkpoint.global_step == 1370


class _TinyPortableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(1, 2, bias=False, dtype=torch.float32)}
        )
        self.config = SimpleNamespace(use_cache=True, pad_token_id=None, eos_token_id=None)
        self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None)


def test_portable_loader_uses_original_base_and_validates_bf16_after_adapter(
    tmp_path: Path,
) -> None:
    config = load_config("configs/transition1x.yaml")
    _, _, checkpoint = _formal_assets(tmp_path, config)
    events: list[str] = []
    base = _TinyPortableModel()

    def base_loader(**kwargs: Any) -> Any:
        events.append("base")
        assert kwargs["cache_dir"] == config.model.cache_dir
        assert kwargs["local_files_only"] is True
        assert kwargs["low_cpu_mem_usage"] is False
        return base

    def adapter_loader(base_model: Any, path: Path, **kwargs: Any) -> Any:
        events.append("adapter")
        assert base_model is base
        assert path == checkpoint.portable_adapter_path
        assert kwargs == {
            "is_trainable": True,
            "autocast_adapter_dtype": False,
            "low_cpu_mem_usage": False,
        }
        return base_model

    def precision(_arguments: Any, model: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("precision")
        assert kwargs["expected_trainable_parameters"] == EXPECTED_PORTABLE_PARAMETERS
        named = dict(model.named_parameters())
        assert named["base.weight"].requires_grad is False
        assert named["base.weight"].dtype == torch.bfloat16
        assert named["lora_A.default.weight"].requires_grad is True
        assert named["lora_A.default.weight"].dtype == torch.bfloat16
        assert named["lora_B.default.weight"].requires_grad is True
        return {"phase": "formal"}

    model = load_zero3_portable_model(
        config,
        checkpoint,
        SimpleNamespace(),
        base_loader=base_loader,
        adapter_loader=adapter_loader,
        zero3_validator=lambda _arguments: events.append("zero3"),
        portable_config_validator=lambda _path, _config: events.append("validate"),
        precision_validator=precision,
    )

    assert model is base
    assert events == ["zero3", "base", "validate", "adapter", "precision"]
    assert model.training is False
    assert model.config.use_cache is False


class _AutocastProbeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})

    def generate(self, value: torch.Tensor) -> torch.Tensor:
        return self.lora_B["default"](self.lora_A["default"](value))


def test_generation_proxy_enters_autocast_and_hard_checks_lora_bf16_outputs() -> None:
    module = _AutocastProbeModel()
    engine = SimpleNamespace(module=module)

    @contextmanager
    def cpu_bf16(_engine: Any):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            yield

    proxy = DeepSpeedGenerationProxy(engine, autocast_context=cpu_bf16)
    result = proxy.generate(torch.ones(1, 2))
    assert result.dtype == torch.bfloat16
    assert proxy._bf16_compute_verified is True

    unsafe = DeepSpeedGenerationProxy(engine, autocast_context=lambda _engine: nullcontext())
    with pytest.raises(FormalEvalRuntimeError, match="did not compute in BF16"):
        unsafe.generate(torch.ones(1, 2))


def test_formal_engine_precision_audits_shards_without_optimizer() -> None:
    class Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
            self.lora_A = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))}
            )
            self.lora_B = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))}
            )
            for parameter in self.parameters():
                parameter.ds_id = id(parameter)
                parameter.ds_tensor = parameter.detach().clone()

    engine = SimpleNamespace(
        module=Module(),
        zero_optimization_stage=lambda: 3,
        bfloat16_enabled=lambda: True,
        fp16_enabled=lambda: False,
        torch_autocast_enabled=lambda: False,
        torch_autocast_dtype=lambda: torch.bfloat16,
    )
    report = assert_formal_zero3_precision(
        engine,
        expected_trainable_parameters=2,
    )
    assert report["zero_stage"] == 3
    assert report["adapter_shard_dtype"] == "bfloat16"

    engine.module.lora_A["default"].ds_tensor = torch.ones(1, dtype=torch.float32)
    with pytest.raises(FormalEvalRuntimeError, match="wrong shard dtype"):
        assert_formal_zero3_precision(engine, expected_trainable_parameters=2)


def test_validation_runtime_orders_zero_init_before_model_and_writes_receipt(
    tmp_path: Path,
) -> None:
    config = load_config("configs/transition1x.yaml")
    processed, checkpoint_dir, checkpoint = _formal_assets(tmp_path, config)
    output = tmp_path / "evaluation"
    events: list[str] = []
    generated_module = object()

    class Trainer:
        model_wrapped: Any = None

        def evaluate(self, *, eval_dataset: Any) -> dict[str, float]:
            events.append("evaluate")
            assert eval_dataset == "loss-data"
            self.model_wrapped = SimpleNamespace(module=generated_module)
            return {"eval_loss": 1.25}

    trainer = Trainer()

    def generation_runner(
        model: Any,
        _tokenizer: Any,
        _records: Any,
        identity: Any,
        **kwargs: Any,
    ) -> Any:
        events.append("generate")
        assert model is generated_module
        assert kwargs["eval_loss"] == 1.25
        predictions = merged_predictions_path(output, identity)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_text("{}\n", encoding="utf-8")
        metrics = metrics_path(output, identity)
        write_json(
            metrics,
            _metrics_payload(
                checkpoint,
                "val",
                sha256_file(predictions),
                eval_loss=1.25,
            ),
        )
        return SimpleNamespace(
            predictions_path=predictions,
            metrics_path=metrics,
            eval_loss=1.25,
        )

    receipt = run_formal_checkpoint_evaluation(
        config,
        processed_path=processed,
        checkpoint_dir=checkpoint_dir,
        output_dir=output,
        split="val",
        environment_installer=lambda: events.append("environment"),
        context_loader=lambda: events.append("context") or SimpleNamespace(local_rank=0),
        arguments_factory=lambda _config, **_kwargs: events.append("arguments") or object(),
        context_validator=lambda *_args: events.append("context_valid"),
        reproducibility_configurer=lambda *_args, **_kwargs: events.append("seed"),
        model_loader=lambda *_args, **_kwargs: events.append("model") or object(),
        tokenizer_loader=lambda *_args, **_kwargs: events.append("tokenizer") or object(),
        data_preparer=lambda *_args, **_kwargs: PreparedFormalData(
            records=(), eval_dataset="loss-data", collator=object()
        ),
        trainer_builder=lambda *_args, **_kwargs: events.append("trainer") or trainer,
        module_resolver=lambda value: events.append("resolve") or value.model_wrapped.module,
        generation_runner=generation_runner,
        torch_module=object(),
    )

    assert receipt is not None
    assert receipt.payload["eval_loss"] == 1.25
    assert receipt.path == runtime_receipt_path(output, checkpoint.generation_identity("val"))
    assert events.index("arguments") < events.index("model")
    assert events.index("evaluate") < events.index("resolve") < events.index("generate")


def test_completed_test_rebuilds_missing_receipt_and_never_loads_model(
    tmp_path: Path,
) -> None:
    config = load_config("configs/transition1x.yaml")
    processed, checkpoint_dir, checkpoint = _formal_assets(tmp_path, config)
    output = tmp_path / "evaluation"
    identity = checkpoint.generation_identity("test")
    predictions = merged_predictions_path(output, identity)
    predictions.parent.mkdir(parents=True)
    predictions.write_text("{}\n", encoding="utf-8")
    metrics = metrics_path(output, identity)
    write_json(
        metrics,
        _metrics_payload(
            checkpoint,
            "test",
            sha256_file(predictions),
            eval_loss=None,
        ),
    )
    completion = TestEvaluationLease(output, identity).completion_path
    write_json(
        completion,
        {
            "schema_version": TEST_COMPLETION_SCHEMA_VERSION,
            **identity.to_json_dict(),
            "predictions_sha256": sha256_file(predictions),
            "metrics_sha256": sha256_file(metrics),
        },
    )
    selection_path = write_selection_proof(
        tmp_path / "selection.json",
        SelectionProof(
            data_fingerprint=checkpoint.data_fingerprint,
            run_fingerprint=checkpoint.run_fingerprint,
            selected_checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
            selected_epoch=checkpoint.epoch,
            selected_global_step=checkpoint.global_step,
        ),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("completed test must not initialize TrainingArguments or a model")

    first = run_formal_checkpoint_evaluation(
        config,
        processed_path=processed,
        checkpoint_dir=checkpoint_dir,
        output_dir=output,
        split="test",
        selection_proof_path=selection_path,
        environment_installer=lambda: None,
        arguments_factory=forbidden,
    )
    second = run_formal_checkpoint_evaluation(
        config,
        processed_path=processed,
        checkpoint_dir=checkpoint_dir,
        output_dir=output,
        split="test",
        selection_proof_path=selection_path,
        environment_installer=lambda: None,
        arguments_factory=forbidden,
    )

    assert first is not None and second is not None
    assert first.payload == second.payload
    assert first.path.is_file()
    assert (
        validate_runtime_receipt(
            first.path,
            checkpoint_dir,
            processed,
            config=config,
        ).payload
        == first.payload
    )


def test_load_checkpoint_score_reconstructs_exact_at10_fractions(tmp_path: Path) -> None:
    config = load_config("configs/transition1x.yaml")
    processed, checkpoint_dir, checkpoint = _formal_assets(tmp_path, config, epoch=4)
    identity = checkpoint.generation_identity("val")
    predictions = merged_predictions_path(tmp_path, identity)
    predictions.write_text("{}\n", encoding="utf-8")
    metrics = metrics_path(tmp_path, identity)
    write_json(
        metrics,
        _metrics_payload(
            checkpoint,
            "val",
            sha256_file(predictions),
            eval_loss=0.75,
            with_score=True,
        ),
    )

    score = load_checkpoint_score(
        checkpoint_dir,
        metrics,
        processed_path=processed,
        config=config,
    )
    assert score.epoch == 4
    assert score.checkpoint_id == checkpoint.checkpoint_fingerprint
    assert score.metrics_at_10.connectivity_successes == 800
    assert score.metrics_at_10.edge_iou_sum.numerator == 1491
    assert score.eval_loss == 0.75


def test_runtime_rejects_test_without_selection_proof(tmp_path: Path) -> None:
    config = load_config("configs/transition1x.yaml")
    with pytest.raises(FormalEvalRuntimeError, match="selection proof"):
        run_formal_checkpoint_evaluation(
            config,
            processed_path=tmp_path / "missing",
            checkpoint_dir=tmp_path / "missing-checkpoint",
            output_dir=tmp_path,
            split="test",
            selection_proof_path=None,
            environment_installer=lambda: None,
        )
