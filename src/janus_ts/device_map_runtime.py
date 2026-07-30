"""Single-process, two-GPU BF16 inference for the 27B Qwen text model."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .constants import QWEN_IM_END_TOKEN_ID, QWEN_PAD_TOKEN_ID
from .modeling import (
    DEFAULT_PISSA_PREPARATION_SPEC,
    EXPECTED_PORTABLE_PARAMETERS,
    PissaPreparationSpec,
    assert_gpu_only_dispatch,
    load_qwen_text_base_for_pissa_preparation,
    validate_portable_adapter_config,
)


class DeviceMapRuntimeError(RuntimeError):
    """The local two-GPU inference contract was violated."""


def _is_lora_parameter(name: str) -> bool:
    return (
        name.startswith("lora_A.")
        or name.startswith("lora_B.")
        or ".lora_A." in name
        or ".lora_B." in name
    )


def _freeze_and_audit_bf16(
    model: Any,
    *,
    expected_adapter_parameters: int,
) -> dict[str, Any]:
    import torch

    adapter_parameters = 0
    adapter_devices: set[str] = set()
    base_parameters = 0
    problems: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        count = int(parameter.numel())
        if _is_lora_parameter(name):
            adapter_parameters += count
            adapter_devices.add(str(parameter.device))
        else:
            base_parameters += count
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            problems.append(f"{name}={parameter.dtype}")
    if adapter_parameters != expected_adapter_parameters:
        problems.append(
            f"adapter_parameters={adapter_parameters:,}, expected={expected_adapter_parameters:,}"
        )
    if expected_adapter_parameters and adapter_devices != {"cuda:0", "cuda:1"}:
        problems.append(f"adapter_devices={sorted(adapter_devices)!r}")
    if problems:
        raise DeviceMapRuntimeError(
            "device-map BF16 parameter contract failed: " + "; ".join(problems[:12])
        )
    return {
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "adapter_devices": sorted(adapter_devices),
        "dtype": "bfloat16",
        "trainable_parameters": 0,
    }


def _configure_generation(model: Any) -> None:
    model.config.use_cache = True
    model.config.pad_token_id = QWEN_PAD_TOKEN_ID
    model.config.eos_token_id = QWEN_IM_END_TOKEN_ID
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = True
        generation_config.pad_token_id = QWEN_PAD_TOKEN_ID
        generation_config.eos_token_id = QWEN_IM_END_TOKEN_ID
    model.eval()


def load_device_map_portable_model(
    config: ExperimentConfig,
    checkpoint: Any,
    *,
    local_files_only: bool = True,
    spec: PissaPreparationSpec = DEFAULT_PISSA_PREPARATION_SPEC,
    base_loader: Callable[..., Any] = load_qwen_text_base_for_pissa_preparation,
    adapter_loader: Callable[..., Any] | None = None,
    portable_config_validator: Callable[[str | Path, Any], None] = (
        validate_portable_adapter_config
    ),
) -> Any:
    """Load original Qwen plus the checkpoint's rank-64 adapter over both GPUs."""

    import torch

    base = base_loader(
        cache_dir=config.model.cache_dir,
        spec=spec,
        local_files_only=local_files_only,
    )
    assert_gpu_only_dispatch(base, spec)
    adapter_path = Path(checkpoint.portable_adapter_path)
    portable_config_validator(adapter_path / "adapter_config.json", base.config)
    if adapter_loader is None:
        from peft import PeftModel

        adapter_loader = PeftModel.from_pretrained
    model = adapter_loader(
        base,
        adapter_path,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    for name, parameter in model.named_parameters():
        if _is_lora_parameter(name) and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    assert_gpu_only_dispatch(model, spec)
    report = _freeze_and_audit_bf16(
        model,
        expected_adapter_parameters=EXPECTED_PORTABLE_PARAMETERS,
    )
    model._janus_ts_device_map_precision_report = report
    _configure_generation(model)
    return model


def load_device_map_zero_shot_model(
    config: ExperimentConfig,
    _baseline: Any,
    *,
    local_files_only: bool = True,
    spec: PissaPreparationSpec = DEFAULT_PISSA_PREPARATION_SPEC,
    base_loader: Callable[..., Any] = load_qwen_text_base_for_pissa_preparation,
) -> Any:
    """Load the frozen adapter-free Qwen text model over both GPUs."""

    model = base_loader(
        cache_dir=config.model.cache_dir,
        spec=spec,
        local_files_only=local_files_only,
    )
    assert_gpu_only_dispatch(model, spec)
    report = _freeze_and_audit_bf16(model, expected_adapter_parameters=0)
    model._janus_ts_device_map_precision_report = report
    _configure_generation(model)
    return model


def _model_input_device(model: Any) -> Any:
    import torch

    value = getattr(model, "device", None)
    if value is not None:
        return torch.device(value)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError) as exc:
        raise DeviceMapRuntimeError("cannot resolve the model input device") from exc


def compute_global_target_token_eval_loss(
    model: Any,
    eval_dataset: Any,
    collator: Callable[[list[Mapping[str, Any]]], Mapping[str, Any]],
    *,
    progress_interval: int = 25,
) -> float:
    """Return target-token-normalized causal NLL over the complete split."""

    import torch

    count = len(eval_dataset)
    if count <= 0:
        raise DeviceMapRuntimeError("eval dataset is empty")
    input_device = _model_input_device(model)
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for index in range(count):
            batch = dict(collator([eval_dataset[index]]))
            labels = batch.get("labels")
            if not isinstance(labels, torch.Tensor):
                raise DeviceMapRuntimeError("collator did not return tensor labels")
            supervised_tokens = int(labels.ne(-100).sum().item())
            if supervised_tokens <= 0:
                raise DeviceMapRuntimeError(f"eval row {index} has no supervised tokens")
            inputs = {
                name: value.to(input_device) if isinstance(value, torch.Tensor) else value
                for name, value in batch.items()
            }
            denominator = torch.tensor(supervised_tokens, device=input_device)
            outputs = model(
                **inputs,
                use_cache=False,
                num_items_in_batch=denominator,
            )
            loss = float(outputs.loss.detach().float().cpu())
            if not math.isfinite(loss) or loss < 0.0:
                raise DeviceMapRuntimeError(f"eval row {index} returned invalid loss {loss!r}")
            total_nll += loss * supervised_tokens
            total_tokens += supervised_tokens
            completed = index + 1
            if completed % progress_interval == 0 or completed == count:
                print(
                    f"[eval-loss] {completed}/{count} loss={total_nll / total_tokens:.10f}",
                    flush=True,
                )
    if total_tokens <= 0:
        raise DeviceMapRuntimeError("eval split has no supervised tokens")
    torch.cuda.empty_cache()
    return total_nll / total_tokens


__all__ = [
    "DeviceMapRuntimeError",
    "compute_global_target_token_eval_loss",
    "load_device_map_portable_model",
    "load_device_map_zero_shot_model",
]
