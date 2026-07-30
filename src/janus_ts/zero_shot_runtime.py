"""Formal test evaluation for the frozen, unmodified Qwen base model."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    read_complete_manifest,
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_json,
)
from .config import ExperimentConfig, load_config
from .constants import (
    MODEL_ID,
    MODEL_REVISION,
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
    REPRESENTATION_VERSION,
    SYSTEM_PROMPT,
)
from .distributed import (
    TorchrunContext,
    assert_initialized_two_rank_job,
    assert_zero3_no_offload,
    configure_reproducibility,
    preflight_torchrun_environment,
)
from .formal_eval_runtime import (
    DeepSpeedGenerationProxy,
    PreparedFormalData,
    _validate_completion,
    build_formal_eval_arguments,
    build_formal_trainer,
    initialize_inference_engine,
    prepare_formal_data,
)
from .generation import (
    FormalEvaluationResult,
    GenerationIdentity,
    formal_generation_kwargs,
    run_formal_generation,
)
from .modeling import load_qwen_text_base
from .preprocessing import load_pinned_tokenizer
from .runtime import install_frozen_environment
from .training import assert_zero3_bf16_runtime

ZERO_SHOT_BASELINE_SCHEMA_VERSION = "janus-ts-qwen-zero-shot-baseline-v1"
ZERO_SHOT_RUNTIME_RECEIPT_SCHEMA_VERSION = "janus-ts-zero-shot-runtime-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ZeroShotRuntimeError(RuntimeError):
    """The frozen Qwen zero-shot evaluation violated its protocol."""


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ZeroShotRuntimeError(f"{name} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ZeroShotBaseline:
    """Content identity for the raw Qwen model and its zero-shot protocol."""

    data_fingerprint: str
    run_fingerprint: str
    model_fingerprint: str
    protocol: Mapping[str, Any]
    epoch: int = 0
    global_step: int = 0

    @property
    def checkpoint_fingerprint(self) -> str:
        # GenerationIdentity predates the baseline and calls this field a
        # checkpoint fingerprint.  For zero-shot it is the frozen raw-model
        # protocol fingerprint; no checkpoint or adapter is loaded.
        return self.model_fingerprint

    def generation_identity(self, split: str = "test") -> GenerationIdentity:
        if split != "test":
            raise ZeroShotRuntimeError("the formal zero-shot baseline is test-only")
        return GenerationIdentity(
            split="test",
            data_fingerprint=self.data_fingerprint,
            run_fingerprint=self.run_fingerprint,
            checkpoint_fingerprint=self.model_fingerprint,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ZERO_SHOT_BASELINE_SCHEMA_VERSION,
            "data_fingerprint": self.data_fingerprint,
            "run_fingerprint": self.run_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "protocol": dict(self.protocol),
        }


@dataclass(frozen=True, slots=True)
class ZeroShotRuntimeReceipt:
    path: Path
    payload: Mapping[str, Any]


def build_zero_shot_baseline(
    config: ExperimentConfig,
    processed_path: str | Path,
    *,
    run_fingerprint: str,
) -> ZeroShotBaseline:
    """Bind the raw pinned Qwen model to the exact formal zero-shot prompt."""

    _require_sha256(run_fingerprint, name="run_fingerprint")
    try:
        manifest = read_complete_manifest(processed_path)
    except ArtifactError as exc:
        raise ZeroShotRuntimeError(str(exc)) from exc
    data_fingerprint = _require_sha256(
        manifest.get("fingerprint"), name="processed data fingerprint"
    )
    if config.model.model_id != MODEL_ID or config.model.revision != MODEL_REVISION:
        raise ZeroShotRuntimeError("configuration does not identify the frozen Qwen model")
    protocol = {
        "schema_version": ZERO_SHOT_BASELINE_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": "bfloat16",
        "adapter": None,
        "training_updates": 0,
        "few_shot_examples": 0,
        "representation": REPRESENTATION_VERSION,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "enable_thinking": False,
        "generation": formal_generation_kwargs(synchronized=True),
    }
    return ZeroShotBaseline(
        data_fingerprint=data_fingerprint,
        run_fingerprint=run_fingerprint,
        model_fingerprint=sha256_json(protocol),
        protocol=protocol,
    )


def _is_lora_parameter(name: str) -> bool:
    return (
        name.startswith("lora_A.")
        or name.startswith("lora_B.")
        or ".lora_A." in name
        or ".lora_B." in name
    )


def assert_frozen_bf16_qwen(model: Any) -> dict[str, Any]:
    """Require an adapter-free, entirely frozen BF16 text model."""

    import torch

    tensors = 0
    parameters = 0
    problems: list[str] = []
    for name, parameter in model.named_parameters():
        tensors += 1
        parameters += int(getattr(parameter, "ds_numel", parameter.numel()))
        if _is_lora_parameter(name):
            problems.append(f"zero-shot model contains adapter parameter {name}")
        if parameter.requires_grad:
            problems.append(f"zero-shot parameter is trainable: {name}")
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            problems.append(f"non-BF16 zero-shot parameter {name}={parameter.dtype}")
    if tensors == 0:
        problems.append("zero-shot model has no parameters")
    if problems:
        raise ZeroShotRuntimeError("zero-shot model contract failed: " + "; ".join(problems[:12]))
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "parameter_tensors": tensors,
        "parameters": parameters,
        "trainable_parameters": 0,
        "adapter_parameters": 0,
    }


def assert_zero_shot_pre_engine_precision(arguments: Any, model: Any) -> dict[str, Any]:
    return {
        "runtime": assert_zero3_bf16_runtime(arguments),
        "parameters": assert_frozen_bf16_qwen(model),
    }


def load_zero3_zero_shot_model(
    config: ExperimentConfig,
    _baseline: ZeroShotBaseline,
    arguments: Any,
    *,
    local_files_only: bool = True,
    base_loader: Callable[..., Any] = load_qwen_text_base,
    zero3_validator: Callable[[Any], None] = assert_zero3_no_offload,
    precision_validator: Callable[[Any, Any], Mapping[str, Any]] = (
        assert_zero_shot_pre_engine_precision
    ),
) -> Any:
    """Load the original Qwen text checkpoint without PEFT under ZeRO-Init."""

    zero3_validator(arguments)
    model = base_loader(
        cache_dir=config.model.cache_dir,
        local_files_only=local_files_only,
        low_cpu_mem_usage=False,
    )
    if getattr(model, "hf_device_map", None):
        raise ZeroShotRuntimeError("zero-shot ZeRO-3 load unexpectedly created a device map")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.config.pad_token_id = QWEN_PAD_TOKEN_ID
    model.config.eos_token_id = QWEN_IM_END_TOKEN_ID
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False
        generation_config.pad_token_id = QWEN_PAD_TOKEN_ID
        generation_config.eos_token_id = QWEN_IM_END_TOKEN_ID
    model._janus_ts_zero_shot_precision_report = dict(precision_validator(arguments, model))
    model.eval()
    return model


def assert_zero_shot_engine_precision(engine: Any) -> dict[str, Any]:
    """Verify that the raw frozen model remains adapter-free BF16 under ZeRO-3."""

    import torch

    required = (
        "bfloat16_enabled",
        "fp16_enabled",
        "torch_autocast_enabled",
        "zero_optimization_stage",
    )
    missing = [name for name in required if not callable(getattr(engine, name, None))]
    if missing:
        raise ZeroShotRuntimeError(f"zero-shot engine lacks DeepSpeed methods: {missing!r}")
    if (
        engine.zero_optimization_stage() != 3
        or not engine.bfloat16_enabled()
        or engine.fp16_enabled()
        or engine.torch_autocast_enabled()
    ):
        raise ZeroShotRuntimeError("zero-shot engine has the wrong native-BF16 mode")
    module = getattr(engine, "module", None)
    if module is None:
        raise ZeroShotRuntimeError("zero-shot DeepSpeed engine has no module")
    parameter_report = assert_frozen_bf16_qwen(module)
    problems: list[str] = []
    for name, parameter in module.named_parameters():
        shard = getattr(parameter, "ds_tensor", None)
        if not hasattr(parameter, "ds_id") or shard is None:
            problems.append(f"unpartitioned parameter {name}")
        elif parameter.is_floating_point() and shard.dtype != torch.bfloat16:
            problems.append(f"wrong shard dtype {name}={shard.dtype}")
    if problems:
        raise ZeroShotRuntimeError(
            "zero-shot ZeRO-3 storage contract failed: " + "; ".join(problems[:12])
        )
    return {**parameter_report, "zero_stage": 3, "native_bf16": True}


def zero_shot_generation_module_from_trainer(trainer: Any) -> DeepSpeedGenerationProxy:
    engine = getattr(trainer, "model_wrapped", None)
    module = getattr(engine, "module", None)
    if module is None or not callable(getattr(module, "generate", None)):
        raise ZeroShotRuntimeError("zero-shot generation requires engine.module.generate")
    parameters = tuple(module.parameters())
    if not parameters or not any(hasattr(parameter, "ds_id") for parameter in parameters):
        raise ZeroShotRuntimeError("zero-shot model lacks ZeRO-3 parameter hooks")
    module._janus_ts_zero_shot_engine_precision_report = assert_zero_shot_engine_precision(engine)
    module.config.use_cache = True
    generation_config = getattr(module, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = True
        generation_config.pad_token_id = QWEN_PAD_TOKEN_ID
        generation_config.eos_token_id = QWEN_IM_END_TOKEN_ID
    module.eval()
    return DeepSpeedGenerationProxy(engine, probe_factory=None)


def zero_shot_receipt_path(output_dir: str | Path, baseline: ZeroShotBaseline) -> Path:
    return Path(output_dir) / f"zero-shot.{baseline.model_fingerprint}.runtime.json"


def _receipt_payload(
    baseline: ZeroShotBaseline,
    *,
    predictions: Path,
    metrics: Path,
) -> dict[str, Any]:
    return {
        "schema_version": ZERO_SHOT_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "baseline": baseline.to_json_dict(),
        "predictions": {"filename": predictions.name, "sha256": sha256_file(predictions)},
        "metrics": {"filename": metrics.name, "sha256": sha256_file(metrics)},
    }


def validate_zero_shot_completion(
    output_dir: str | Path,
    baseline: ZeroShotBaseline,
) -> ZeroShotRuntimeReceipt | None:
    """Validate or install the small receipt for a completed baseline test."""

    completed = _validate_completion(Path(output_dir), baseline)  # type: ignore[arg-type]
    if completed is None:
        return None
    predictions, metrics, _ = completed
    expected = _receipt_payload(baseline, predictions=predictions, metrics=metrics)
    path = zero_shot_receipt_path(output_dir, baseline)
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ZeroShotRuntimeError(f"cannot read zero-shot receipt: {exc}") from exc
        if actual != expected:
            raise ZeroShotRuntimeError("existing zero-shot receipt differs from completed output")
    else:
        write_json(path, expected)
    return ZeroShotRuntimeReceipt(path=path, payload=expected)


def run_zero_shot_evaluation(
    config: ExperimentConfig,
    *,
    processed_path: str | Path,
    output_dir: str | Path,
    run_fingerprint: str,
    local_files_only: bool = True,
    environment_installer: Callable[[], Any] = install_frozen_environment,
    context_loader: Callable[[], TorchrunContext] = preflight_torchrun_environment,
    arguments_factory: Callable[..., Any] = build_formal_eval_arguments,
    context_validator: Callable[[Any, TorchrunContext, Any], None] = (
        assert_initialized_two_rank_job
    ),
    reproducibility_configurer: Callable[..., None] = configure_reproducibility,
    model_loader: Callable[..., Any] = load_zero3_zero_shot_model,
    tokenizer_loader: Callable[..., Any] = load_pinned_tokenizer,
    data_preparer: Callable[..., PreparedFormalData] = prepare_formal_data,
    trainer_builder: Callable[..., Any] = build_formal_trainer,
    inference_initializer: Callable[[Any], Any] = initialize_inference_engine,
    module_resolver: Callable[[Any], Any] = zero_shot_generation_module_from_trainer,
    generation_runner: Callable[..., FormalEvaluationResult | None] = run_formal_generation,
    torch_module: Any | None = None,
) -> ZeroShotRuntimeReceipt | None:
    """Run the fixed raw-Qwen baseline once on the complete test split."""

    environment_installer()
    baseline = build_zero_shot_baseline(
        config,
        processed_path,
        run_fingerprint=run_fingerprint,
    )
    output_root = Path(output_dir)
    recovered = validate_zero_shot_completion(output_root, baseline)
    if recovered is not None:
        return recovered

    context = context_loader()
    if torch_module is None:
        import torch as torch_module
    reproducibility_configurer(
        torch_module,
        seed=config.seed,
        cuda_device=context.local_rank,
    )
    arguments = arguments_factory(config, output_dir=output_root / "zero-shot")
    context_validator(arguments, context, torch_module)
    model = model_loader(
        config,
        baseline,
        arguments,
        local_files_only=local_files_only,
    )
    tokenizer = tokenizer_loader(config, local_files_only=local_files_only)
    prepared = data_preparer(config, processed_path, tokenizer, "test")
    trainer = trainer_builder(model, arguments, prepared)
    inference_initializer(trainer)
    generation_model = module_resolver(trainer)
    result = generation_runner(
        generation_model,
        tokenizer,
        prepared.records,
        baseline.generation_identity("test"),
        output_dir=output_root,
        eval_loss=None,
        selection_proof_path=None,
        test_role="frozen_zero_shot",
    )
    if result is None:
        return None
    receipt = validate_zero_shot_completion(output_root, baseline)
    if receipt is None:
        raise ZeroShotRuntimeError("zero-shot generation returned without durable completion")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-fingerprint", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = run_zero_shot_evaluation(
        load_config(arguments.config),
        processed_path=arguments.processed_path,
        output_dir=arguments.output_dir,
        run_fingerprint=arguments.run_fingerprint,
        local_files_only=True,
    )
    if receipt is not None and int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(receipt.payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ZERO_SHOT_BASELINE_SCHEMA_VERSION",
    "ZERO_SHOT_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "ZeroShotBaseline",
    "ZeroShotRuntimeError",
    "ZeroShotRuntimeReceipt",
    "assert_frozen_bf16_qwen",
    "assert_zero_shot_pre_engine_precision",
    "assert_zero_shot_engine_precision",
    "build_zero_shot_baseline",
    "load_zero3_zero_shot_model",
    "run_zero_shot_evaluation",
    "validate_zero_shot_completion",
    "zero_shot_generation_module_from_trainer",
    "zero_shot_receipt_path",
]
