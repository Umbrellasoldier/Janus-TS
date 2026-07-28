"""Two-rank CUDA gate for native-BF16 model storage and compute.

Run this only after the host-resource gate has established exclusive access::

    torchrun --standalone --nproc-per-node=2 --module janus_ts.dtype_gate \
      --output artifacts/gates/zero3-dtype.json

The toy graph exercises the exact DeepSpeed 0.19.2 mechanism used by the
27B run without loading Qwen weights.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .runtime import install_frozen_environment
from .training import (
    ZERO3_AUTOCAST_SAFE_MODULES,
    assert_zero3_engine_precision_contract,
)

TINY_LORA_PARAMETERS = 256


def snapshot_trainable_zero3_shards(model: Any) -> dict[str, Any]:
    """Clone every trainable ZeRO shard without relying on PEFT name shapes."""

    snapshots: dict[str, Any] = {}
    missing_shards: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        shard = getattr(parameter, "ds_tensor", None)
        if shard is None:
            missing_shards.append(name)
            continue
        snapshots[name] = shard.detach().clone()
    if missing_shards:
        raise RuntimeError(f"trainable parameters have no ZeRO shard: {missing_shards!r}")
    if not snapshots:
        raise RuntimeError("the dtype gate found no trainable ZeRO shards to snapshot")
    return snapshots


def measure_trainable_zero3_shard_updates(
    model: Any,
    before_shards: dict[str, Any],
) -> tuple[list[str], float]:
    """Return changed trainable shard names and their largest absolute delta."""

    after_shards = {
        name: parameter.ds_tensor
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and getattr(parameter, "ds_tensor", None) is not None
    }
    before_names = set(before_shards)
    after_names = set(after_shards)
    if before_names != after_names:
        raise RuntimeError(
            "trainable ZeRO shard names changed across the optimizer step: "
            f"before={sorted(before_names)!r}, after={sorted(after_names)!r}"
        )

    changed_names: list[str] = []
    max_abs_delta = 0.0
    for name in sorted(before_names):
        before = before_shards[name]
        after = after_shards[name]
        if before.shape != after.shape:
            raise RuntimeError(
                f"trainable ZeRO shard shape changed for {name}: "
                f"before={tuple(before.shape)!r}, after={tuple(after.shape)!r}"
            )
        if not after.equal(before):
            changed_names.append(name)
        if after.numel():
            delta = float((after.detach().float() - before.float()).abs().max())
            max_abs_delta = max(max_abs_delta, delta)
    return changed_names, max_abs_delta


def build_tiny_zero3_dtype_gate_config() -> dict[str, Any]:
    """Return the minimal runtime config matching the full training phase."""

    return {
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "torch_autocast": {
            "enabled": False,
            "dtype": "bfloat16",
            "lower_precision_safe_modules": list(ZERO3_AUTOCAST_SAFE_MODULES),
        },
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 4,
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 4096,
            "stage3_prefetch_bucket_size": 4096,
            "stage3_param_persistence_threshold": 0,
            "stage3_gather_16bit_weights_on_model_save": False,
        },
        "steps_per_print": 2**31,
        "wall_clock_breakdown": False,
    }


def _run_gate(output: Path) -> None:
    import deepspeed
    import torch
    import torch.distributed as dist
    from torch import nn

    class PeftLikeLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(32, 4, bias=False, dtype=torch.bfloat16)}
            )
            self.lora_B = nn.ModuleDict(
                {"default": nn.Linear(4, 32, bias=False, dtype=torch.bfloat16)}
            )
            self.base_layer.requires_grad_(False)

        def forward(self, inputs: Any) -> Any:
            residual = self.base_layer(inputs)
            adapter = self.lora_B["default"](self.lora_A["default"](inputs))
            return residual + adapter

    install_frozen_environment()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("dtype gate requires exactly two visible CUDA devices")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("launch dtype gate with torchrun --nproc-per-node=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = PeftLikeLinear()
    observed_output_dtypes: list[str] = []
    model.lora_A["default"].register_forward_hook(
        lambda _module, _inputs, result: observed_output_dtypes.append(str(result.dtype))
    )
    model.lora_B["default"].register_forward_hook(
        lambda _module, _inputs, result: observed_output_dtypes.append(str(result.dtype))
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=build_tiny_zero3_dtype_gate_config(),
    )
    before_report = assert_zero3_engine_precision_contract(
        engine,
        expected_trainable_parameters=TINY_LORA_PARAMETERS,
        require_optimizer_states=False,
    )
    before_shards = snapshot_trainable_zero3_shards(engine.module)

    inputs = torch.randn(2, 32, device=engine.device, dtype=torch.bfloat16)
    engine.train()
    outputs = engine(inputs)
    loss = outputs.float().square().mean()
    engine.backward(loss)
    engine.step()

    after_report = assert_zero3_engine_precision_contract(
        engine,
        expected_trainable_parameters=TINY_LORA_PARAMETERS,
        require_optimizer_states=True,
    )
    changed_names, max_abs_delta = measure_trainable_zero3_shard_updates(
        engine.module, before_shards
    )
    if not changed_names:
        raise RuntimeError("the optimizer update did not change any BF16 LoRA shard")
    if observed_output_dtypes != [str(torch.bfloat16), str(torch.bfloat16)]:
        raise RuntimeError(f"LoRA Linear compute did not use BF16: {observed_output_dtypes!r}")

    local_report = {
        "rank": rank,
        "local_rank": local_rank,
        "loss": float(loss.detach()),
        "lora_linear_output_dtypes": observed_output_dtypes,
        "adapter_updated": True,
        "changed_names": changed_names,
        "max_abs_delta": max_abs_delta,
        "before": before_report,
        "after": after_report,
    }
    reports: list[dict[str, Any] | None] = [None, None]
    dist.all_gather_object(reports, local_report)
    if rank == 0:
        write_json(
            output,
            {
                "status": "pass",
                "world_size": 2,
                "seed": 42,
                "reports": reports,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    _run_gate(arguments.output)


if __name__ == "__main__":
    main()
