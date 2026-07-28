"""Hard-gate orchestration and immutable evidence reports."""

from __future__ import annotations

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
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        result: dict[str, Any] = {
            "stage": stage,
            "tokens": int(self._input_ids.shape[1]),
            "argmax": int(logits.argmax()),
            "finite": bool(torch.isfinite(logits).all()),
        }
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
