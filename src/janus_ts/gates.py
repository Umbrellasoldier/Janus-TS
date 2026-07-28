"""Hard-gate orchestration and immutable evidence reports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifacts import mark_complete, read_complete_manifest, sha256_file, write_json
from .constants import MODEL_REVISION, QWEN_IM_END_TOKEN_ID, SEED
from .modeling import (
    DEFAULT_PISSA_PREPARATION_SPEC,
    prepare_pissa_residual_bundle,
)
from .native_stack import audit_native_stack
from .runtime import (
    approved_external_gpu_pids,
    assert_host_ready,
    descendant_pids,
    exclusive_lock,
    host_status,
    install_frozen_environment,
)
from .snapshot import audit_snapshot

PISSA_PARITY_ATOL = 0.125
PISSA_PARITY_RTOL = 0.02


class GateError(RuntimeError):
    """A frozen launch gate did not pass."""


def _swap_used_kib() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            values[name] = int(value.strip().split()[0])
    return values["SwapTotal"] - values["SwapFree"]


class _PissaLoraOutputProbe:
    """Prove that PiSSA A/B linears follow the training BF16 compute path."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._handles: list[Any] = []
        self._observed: dict[str, tuple[str, Any]] = {}

    @staticmethod
    def _label(name: str) -> str | None:
        if ".lora_A." in name or name.startswith("lora_A."):
            return "lora_A"
        if ".lora_B." in name or name.startswith("lora_B."):
            return "lora_B"
        return None

    def start(self) -> None:
        selected: dict[str, tuple[str, Any]] = {}
        for name, child in self._model.named_modules():
            label = self._label(name)
            if label is not None and label not in selected:
                selected[label] = (name, child)
            if set(selected) == {"lora_A", "lora_B"}:
                break
        if set(selected) != {"lora_A", "lora_B"}:
            raise GateError("cannot locate PiSSA LoRA A/B modules for BF16 compute probe")

        def hook(label: str, name: str) -> Callable[..., None]:
            def record(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, (tuple, list)) and output else output
                self._observed[label] = (name, getattr(value, "dtype", None))

            return record

        for label, (name, child) in selected.items():
            self._handles.append(child.register_forward_hook(hook(label, name)))

    def abort(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finish(self) -> dict[str, str]:
        import torch

        self.abort()
        evidence: dict[str, str] = {}
        problems: list[str] = []
        for label in ("lora_A", "lora_B"):
            observed = self._observed.get(label)
            if observed is None:
                problems.append(f"{label} forward hook did not fire")
                continue
            name, dtype = observed
            evidence[label] = str(dtype)
            if dtype != torch.bfloat16:
                problems.append(f"{name} output dtype={dtype}")
        if problems:
            raise GateError("PiSSA adapter did not compute in BF16: " + "; ".join(problems))
        return evidence


class PissaParityProbe:
    """Three-stage untouched/PiSSA/reload logit comparison.

    Only a small CPU tensor survives between stages; no model or CUDA tensor
    is retained by the hook.
    """

    def __init__(self, tokenizer: Any) -> None:
        text = (
            "<|im_start|>system\nMoleCode-TS/v1<|im_end|>\n"
            "<|im_start|>user\nMoleCode-TS/v1\n<ATOMS>\n"
            "a0 [element=H,Z=1]\n</ATOMS><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        self._input_ids = encoded["input_ids"].cpu()
        self._attention_mask = encoded["attention_mask"].cpu()
        self._reference: Any | None = None

    def __call__(self, stage: str, model: Any) -> dict[str, Any]:
        import torch

        device = model.get_input_embeddings().weight.device
        inputs = {
            "input_ids": self._input_ids.to(device),
            "attention_mask": self._attention_mask.to(device),
        }
        model.eval()
        adapter_probe = None if stage == "original_base" else _PissaLoraOutputProbe(model)
        if adapter_probe is not None:
            adapter_probe.start()
        try:
            # DeepSpeed owns an equivalent explicit BF16 autocast context in
            # training and formal evaluation.  PiSSA A/B parameters remain
            # FP32 for optimizer state, but their matrix multiplies must be
            # evaluated in BF16 when comparing the reconstructed model with
            # the untouched BF16 checkpoint.
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
            ):
                output = model(**inputs, use_cache=False)
        except BaseException:
            if adapter_probe is not None:
                adapter_probe.abort()
            raise
        lora_output_dtypes = None if adapter_probe is None else adapter_probe.finish()
        output_dtype = str(output.logits.dtype)
        logits = output.logits[:, -1, :].float().cpu()
        result: dict[str, Any] = {
            "stage": stage,
            "tokens": int(self._input_ids.shape[1]),
            "argmax": int(logits.argmax()),
            "finite": bool(torch.isfinite(logits).all()),
            "autocast_device_type": device.type,
            "autocast_dtype": str(torch.bfloat16),
            "logits_dtype_before_cpu_cast": output_dtype,
        }
        if lora_output_dtypes is not None:
            result["lora_output_dtypes"] = lora_output_dtypes
        if not result["finite"]:
            raise GateError(f"non-finite PiSSA probe logits at {stage}")
        if stage == "original_base":
            self._reference = logits
            result.update({"reference": True, "atol": PISSA_PARITY_ATOL, "rtol": PISSA_PARITY_RTOL})
            return result
        if self._reference is None:
            raise GateError(f"PiSSA parity stage {stage!r} ran before original_base")
        difference = (logits - self._reference).abs()
        result.update(
            {
                "reference": False,
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "argmax_equal": bool(logits.argmax() == self._reference.argmax()),
                "atol": PISSA_PARITY_ATOL,
                "rtol": PISSA_PARITY_RTOL,
            }
        )
        try:
            torch.testing.assert_close(
                logits,
                self._reference,
                atol=PISSA_PARITY_ATOL,
                rtol=PISSA_PARITY_RTOL,
            )
        except AssertionError as exc:
            raise GateError(f"PiSSA probe parity failed at {stage}: {result!r}") from exc
        if not result["argmax_equal"]:
            raise GateError(f"PiSSA probe argmax changed at {stage}: {result!r}")
        return result


def pissa_bundle_path(local_cache_root: str | Path) -> Path:
    return (
        Path(local_cache_root)
        / "models"
        / f"qwen3.6-27b-{MODEL_REVISION}"
        / "pissa-residual-v1"
    )


def verify_bundle_payload(bundle_dir: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_dir)
    manifest = read_complete_manifest(bundle)
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or not hashes:
        raise GateError("PiSSA manifest has no payload hashes")
    for relative, expected_hash in hashes.items():
        path = bundle / relative
        if not path.is_file():
            raise GateError(f"PiSSA payload is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise GateError(f"PiSSA payload hash mismatch for {relative}: {actual}")
    return manifest


def prepare_pissa_gate(
    *,
    hub_cache_dir: str | Path,
    local_cache_root: str | Path,
    gpu_lock_path: str | Path,
) -> dict[str, Any]:
    """Run or verify the one-time GPU-only PiSSA residual preparation."""

    install_frozen_environment()
    destination = pissa_bundle_path(local_cache_root)
    if destination.exists():
        return verify_bundle_payload(destination)
    with exclusive_lock(gpu_lock_path):
        allowed = tuple(
            sorted(set(descendant_pids()) | set(approved_external_gpu_pids()))
        )
        before = assert_host_ready(allowed_pids=allowed)
        native = audit_native_stack()
        snapshot = audit_snapshot(cache_dir=hub_cache_dir)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot["snapshot"], local_files_only=True, trust_remote_code=False
        )
        probe = PissaParityProbe(tokenizer)
        swap_before = _swap_used_kib()
        manifest = prepare_pissa_residual_bundle(
            destination,
            cache_dir=hub_cache_dir,
            parity_hook=probe,
            manifest_metadata={
                "snapshot_critical_sha256": snapshot["critical_sha256"],
                "native_stack": native,
                "host_before": before.to_dict(),
                "seed": SEED,
                "probe_terminal_token": QWEN_IM_END_TOKEN_ID,
            },
            spec=DEFAULT_PISSA_PREPARATION_SPEC,
            local_files_only=True,
        )
        swap_growth = _swap_used_kib() - swap_before
        if swap_growth > 256 * 1024:
            raise GateError(
                f"PiSSA preparation grew swap by {swap_growth / 1024:.1f} MiB (>256 MiB)"
            )
        # The immutable bundle has already recorded its stage parity. This
        # outer value is returned for the launch-gate report.
        return {**manifest, "observed_swap_growth_kib": swap_growth}


def write_environment_gate(output_dir: str | Path) -> dict[str, Any]:
    """Record a CPU/read-only host report; GPU idleness is rechecked at launch."""

    install_frozen_environment()
    status = host_status()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "observed",
        "host": status.to_dict(),
        "native_stack": audit_native_stack(),
        "note": "foreign compute processes are transient and are rechecked before every GPU phase",
    }
    write_json(output / "environment.json", report)
    return report


def complete_gate_directory(output_dir: str | Path, reports: dict[str, Any]) -> None:
    """Seal a gate directory only after its caller has run every required gate."""

    mark_complete(output_dir, {"status": "pass", "reports": reports})


__all__ = [
    "GateError",
    "PISSA_PARITY_ATOL",
    "PISSA_PARITY_RTOL",
    "PissaParityProbe",
    "complete_gate_directory",
    "pissa_bundle_path",
    "prepare_pissa_gate",
    "verify_bundle_payload",
    "write_environment_gate",
]
