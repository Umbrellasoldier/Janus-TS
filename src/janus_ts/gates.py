"""Hard-gate orchestration and immutable evidence reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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

PISSA_PARITY_TOP_K = 32
PISSA_PARITY_MIN_TOP_K_OVERLAP = 0.90
PISSA_PARITY_MAX_TOTAL_VARIATION = 0.05
PISSA_PARITY_MAX_JENSEN_SHANNON = 0.002
PISSA_PARITY_MAX_CENTERED_NRMSE = 0.05
PISSA_MIN_MEM_AVAILABLE_KIB = 16 * 1024 * 1024
PISSA_RESOURCE_POLICY = "pissa-one-time-serialization-memavailable-v1"


class GateError(RuntimeError):
    """A frozen launch gate did not pass."""


def _pissa_resource_evidence(
    before: Any,
    after: Any,
    *,
    scope: str,
    creation_host_before: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the confirmed one-time serialization host-memory policy."""

    if scope not in {"full_serialization", "cached_payload_verification"}:
        raise GateError(f"invalid PiSSA resource observation scope: {scope!r}")
    for label, status in (("before", before), ("after", after)):
        available = getattr(status, "mem_available_kib", None)
        if type(available) is not int or available < PISSA_MIN_MEM_AVAILABLE_KIB:
            raise GateError(
                f"PiSSA {scope} MemAvailable {label}={available!r} KiB, "
                f"requires >= {PISSA_MIN_MEM_AVAILABLE_KIB} KiB"
            )
        swap_free = getattr(status, "swap_free_kib", None)
        if type(swap_free) is not int or swap_free < 0:
            raise GateError(f"PiSSA {scope} SwapFree {label} is invalid: {swap_free!r}")

    evidence = {
        "policy": PISSA_RESOURCE_POLICY,
        "scope": scope,
        "minimum_mem_available_kib": PISSA_MIN_MEM_AVAILABLE_KIB,
        "swap_growth_is_diagnostic": True,
        "other_gpu_phases_max_swap_growth_kib": 256 * 1024,
        "observed_swap_growth_kib": before.swap_free_kib - after.swap_free_kib,
        "host_before": before.to_dict(),
        "host_after": after.to_dict(),
    }
    if creation_host_before is not None:
        evidence["creation_host_before"] = dict(creation_host_before)
    return evidence


def _creation_host_before(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise GateError("PiSSA bundle manifest lacks metadata")
    host = metadata.get("host_before")
    if not isinstance(host, Mapping):
        raise GateError("PiSSA bundle manifest lacks creation host evidence")
    return host


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


def _pissa_parity_policy() -> dict[str, Any]:
    return {
        "name": "bf16-behavioral-distribution-v1",
        "top_k": PISSA_PARITY_TOP_K,
        "min_top_k_overlap": PISSA_PARITY_MIN_TOP_K_OVERLAP,
        "max_total_variation": PISSA_PARITY_MAX_TOTAL_VARIATION,
        "max_jensen_shannon": PISSA_PARITY_MAX_JENSEN_SHANNON,
        "max_centered_nrmse": PISSA_PARITY_MAX_CENTERED_NRMSE,
        "requires_exact_argmax": True,
    }


def _pissa_behavioral_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    """Measure BF16 reconstruction error in generation-relevant coordinates."""

    import math

    import torch

    left = torch.as_tensor(reference).detach().cpu().float().flatten().contiguous()
    right = torch.as_tensor(candidate).detach().cpu().float().flatten().contiguous()
    if left.numel() == 0 or right.shape != left.shape:
        raise GateError(
            f"PiSSA parity logits have invalid shapes: {tuple(left.shape)} / {tuple(right.shape)}"
        )
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise GateError("PiSSA behavioral parity logits contain NaN or infinity")

    difference = right - left
    absolute = difference.abs()
    centered_left = left - left.mean()
    centered_difference = difference - difference.mean()
    reference_rms = torch.sqrt(torch.mean(centered_left.square()))
    centered_rmse = torch.sqrt(torch.mean(centered_difference.square()))
    centered_nrmse = centered_rmse / reference_rms.clamp_min(torch.finfo(torch.float32).eps)

    log_left = torch.log_softmax(left, dim=-1)
    log_right = torch.log_softmax(right, dim=-1)
    left_probability = log_left.exp()
    right_probability = log_right.exp()
    log_midpoint = torch.logaddexp(log_left, log_right) - math.log(2.0)
    jensen_shannon = (
        0.5
        * (
            torch.sum(left_probability * (log_left - log_midpoint))
            + torch.sum(right_probability * (log_right - log_midpoint))
        )
    ).clamp_min(0.0)
    probability_difference = (right_probability - left_probability).abs()

    top_k = min(PISSA_PARITY_TOP_K, int(left.numel()))
    left_top = torch.topk(left, top_k, sorted=True).indices.tolist()
    right_top = torch.topk(right, top_k, sorted=True).indices.tolist()
    overlap_count = len(set(left_top) & set(right_top))
    left_argmax = int(left.argmax().item())
    right_argmax = int(right.argmax().item())
    return {
        "vocabulary_size": int(left.numel()),
        "reference_argmax": left_argmax,
        "candidate_argmax": right_argmax,
        "argmax_equal": left_argmax == right_argmax,
        "top_k": top_k,
        "top_k_overlap_count": overlap_count,
        "top_k_overlap": overlap_count / top_k,
        "reference_top_k_token_ids": left_top,
        "candidate_top_k_token_ids": right_top,
        "total_variation": float(0.5 * probability_difference.sum()),
        "jensen_shannon": float(jensen_shannon),
        "max_probability_delta": float(probability_difference.max()),
        "centered_nrmse": float(centered_nrmse),
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "centered_rmse": float(centered_rmse),
        "reference_centered_rms": float(reference_rms),
    }


def _assert_pissa_behavioral_parity(metrics: dict[str, Any], *, stage: str) -> None:
    failures: list[str] = []
    if not metrics["argmax_equal"]:
        failures.append("argmax changed")
    if metrics["top_k_overlap"] < PISSA_PARITY_MIN_TOP_K_OVERLAP:
        failures.append("top-k overlap below minimum")
    if metrics["total_variation"] > PISSA_PARITY_MAX_TOTAL_VARIATION:
        failures.append("total variation above maximum")
    if metrics["jensen_shannon"] > PISSA_PARITY_MAX_JENSEN_SHANNON:
        failures.append("Jensen-Shannon divergence above maximum")
    if metrics["centered_nrmse"] > PISSA_PARITY_MAX_CENTERED_NRMSE:
        failures.append("centered logit NRMSE above maximum")
    if failures:
        raise GateError(
            f"PiSSA behavioral parity failed at {stage}: {', '.join(failures)}; "
            f"metrics={metrics!r}; policy={_pissa_parity_policy()!r}"
        )


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
            result.update({"reference": True, "parity_policy": _pissa_parity_policy()})
            return result
        if self._reference is None:
            raise GateError(f"PiSSA parity stage {stage!r} ran before original_base")
        metrics = _pissa_behavioral_metrics(self._reference, logits)
        result.update(
            {
                "reference": False,
                **metrics,
                "parity_policy": _pissa_parity_policy(),
            }
        )
        _assert_pissa_behavioral_parity(metrics, stage=stage)
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
    with exclusive_lock(gpu_lock_path):
        allowed = tuple(
            sorted(set(descendant_pids()) | set(approved_external_gpu_pids()))
        )
        before = assert_host_ready(allowed_pids=allowed)
        if destination.exists():
            manifest = verify_bundle_payload(destination)
            after = assert_host_ready(allowed_pids=allowed)
            resources = _pissa_resource_evidence(
                before,
                after,
                scope="cached_payload_verification",
                creation_host_before=_creation_host_before(manifest),
            )
            return {**manifest, "preparation_resources": resources}
        native = audit_native_stack()
        snapshot = audit_snapshot(cache_dir=hub_cache_dir)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot["snapshot"], local_files_only=True, trust_remote_code=False
        )
        probe = PissaParityProbe(tokenizer)
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
        after = assert_host_ready(allowed_pids=allowed)
        resources = _pissa_resource_evidence(
            before,
            after,
            scope="full_serialization",
            creation_host_before=_creation_host_before(manifest),
        )
        return {**manifest, "preparation_resources": resources}


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
    "PISSA_PARITY_MAX_CENTERED_NRMSE",
    "PISSA_PARITY_MAX_JENSEN_SHANNON",
    "PISSA_PARITY_MAX_TOTAL_VARIATION",
    "PISSA_PARITY_MIN_TOP_K_OVERLAP",
    "PISSA_PARITY_TOP_K",
    "PISSA_MIN_MEM_AVAILABLE_KIB",
    "PISSA_RESOURCE_POLICY",
    "PissaParityProbe",
    "complete_gate_directory",
    "pissa_bundle_path",
    "prepare_pissa_gate",
    "verify_bundle_payload",
    "write_environment_gate",
]
