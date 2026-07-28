"""Distributed CUDA gates executed before any 27B allocation.

Run with exactly two ranks, for example::

    torchrun --standalone --nproc-per-node=2 -m janus_ts.gpu_gates
"""

from __future__ import annotations

import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .constants import MODEL_REVISION, SEED
from .runtime import install_frozen_environment


class GpuGateError(RuntimeError):
    """A CUDA, kernel, or collective hard gate failed."""


def _swap_used_kib() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values["SwapTotal"] - values["SwapFree"]


def _kernel_gate(torch: Any, device: Any) -> list[dict[str, object]]:
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module

    if not qwen_module.is_fast_path_available:
        raise GpuGateError("Qwen3.5 linear-attention fast path is unavailable")
    expected_modules = {
        "causal_conv1d_fn": "causal_conv1d",
        "causal_conv1d_update": "causal_conv1d",
        "chunk_gated_delta_rule": "fla",
        "fused_recurrent_gated_delta_rule": "fla",
    }
    functions: dict[str, str] = {}
    for name, prefix in expected_modules.items():
        function = getattr(qwen_module, name)
        module_name = getattr(function, "__module__", "")
        if not module_name.startswith(prefix):
            raise GpuGateError(f"{name} resolved to fallback/unexpected module {module_name!r}")
        functions[name] = module_name

    # This is the actual Qwen token mixer with the production projection
    # dimensions.  Testing just this layer exercises causal-conv1d, FLA's
    # chunked delta rule, and fused gated RMS normalization without allocating
    # an unnecessary 17408-wide MLP for the gate.
    from transformers import Qwen3_5TextConfig

    from .modeling import load_qwen_text_config

    config = load_qwen_text_config(
        cache_dir="/home/caoxiangyu/.cache/huggingface", local_files_only=True
    )
    if not isinstance(config, Qwen3_5TextConfig):
        raise GpuGateError(f"unexpected text config class: {type(config).__name__}")
    config.dtype = torch.bfloat16
    mixer = qwen_module.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(
        device=device, dtype=torch.bfloat16
    )
    mixer.train()
    results: list[dict[str, object]] = []
    for length in (63, 64, 65):
        mixer.zero_grad(set_to_none=True)
        hidden = torch.randn(
            2,
            length,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        mask = torch.ones((2, length), device=device, dtype=torch.bool)
        mask[1, -1] = False
        output = mixer(hidden_states=hidden, attention_mask=mask)
        if output.shape != hidden.shape or not torch.isfinite(output).all():
            raise GpuGateError(f"invalid Qwen fast-path output at length {length}")
        output.float().square().mean().backward()
        if hidden.grad is None or not torch.isfinite(hidden.grad).all():
            raise GpuGateError(f"invalid Qwen fast-path gradient at length {length}")
        results.append(
            {
                "length": length,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
            }
        )
        del hidden, mask, output
    del mixer
    torch.cuda.empty_cache()
    return [{"resolved_fast_path_modules": functions}, *results]


def run_distributed_gates(output_path: str | Path | None = None) -> dict[str, object]:
    install_frozen_environment()
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise GpuGateError("CUDA is unavailable")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise GpuGateError("GPU gate requires WORLD_SIZE=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 2:
        raise GpuGateError(f"expected two NCCL ranks, got {dist.get_world_size()}")

    torch.manual_seed(SEED + local_rank)
    torch.cuda.manual_seed_all(SEED + local_rank)
    swap_before = _swap_used_kib()
    collective = torch.tensor([float(local_rank + 1)], device=device)
    dist.all_reduce(collective)
    if collective.item() != 3.0:
        raise GpuGateError(f"NCCL all-reduce returned {collective.item()}, expected 3")
    kernel_results = _kernel_gate(torch, device)
    torch.cuda.synchronize(device)
    local = {
        "rank": dist.get_rank(),
        "local_rank": local_rank,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) // 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) // 1024**2,
        "swap_growth_kib": _swap_used_kib() - swap_before,
        "kernel_results": kernel_results,
    }
    gathered: list[dict[str, object] | None] = [None, None]
    dist.all_gather_object(gathered, local)
    dist.barrier()
    excessive_swap = [
        item for item in gathered if item is not None and int(item["swap_growth_kib"]) > 256 * 1024
    ]
    if excessive_swap:
        raise GpuGateError(f"GPU gate grew swap by more than 256 MiB: {excessive_swap!r}")
    report: dict[str, object] = {
        "gate": "two-rank-nccl-qwen-linear-fast-path",
        "status": "pass",
        "host": socket.gethostname(),
        "model_revision": MODEL_REVISION,
        "ranks": gathered,
    }
    if dist.get_rank() == 0 and output_path is not None:
        write_json(output_path, report)
    dist.destroy_process_group()
    return report


def main() -> None:
    output = os.environ.get("JANUS_TS_GATE_OUTPUT")
    report = run_distributed_gates(output)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
