"""Two-rank CUDA gate for BF16 compute with true FP32 LoRA storage.

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


def build_tiny_zero3_dtype_gate_config() -> dict[str, Any]:
    """Return the minimal runtime config matching the full training phase."""

    return {
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
        "torch_autocast": {
            "enabled": True,
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
                {"default": nn.Linear(32, 4, bias=False, dtype=torch.float32)}
            )
            self.lora_B = nn.ModuleDict(
                {"default": nn.Linear(4, 32, bias=False, dtype=torch.float32)}
            )
            self.base_layer.requires_grad_(False)

        def forward(self, inputs: Any) -> Any:
            residual = self.base_layer(inputs)
            adapter_inputs = inputs.to(self.lora_A["default"].weight.dtype)
            adapter = self.lora_B["default"](self.lora_A["default"](adapter_inputs))
            return residual + adapter.to(residual.dtype)

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
    before_shards = {
        name: parameter.ds_tensor.detach().clone()
        for name, parameter in engine.module.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }

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
    changed = any(
        not torch.equal(before_shards[name], parameter.ds_tensor)
        for name, parameter in engine.module.named_parameters()
        if name in before_shards
    )
    if not changed:
        raise RuntimeError("the optimizer update did not change any FP32 LoRA shard")
    if observed_output_dtypes != [str(torch.bfloat16), str(torch.bfloat16)]:
        raise RuntimeError(f"LoRA Linear compute did not use BF16: {observed_output_dtypes!r}")

    local_report = {
        "rank": rank,
        "local_rank": local_rank,
        "loss": float(loss.detach()),
        "lora_linear_output_dtypes": observed_output_dtypes,
        "adapter_updated": changed,
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
