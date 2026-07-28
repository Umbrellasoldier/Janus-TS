"""Identity and ABI audit for Qwen's native linear-attention stack."""

from __future__ import annotations

import importlib
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .runtime import install_frozen_environment

CAUSAL_CONV1D_VERSION = "1.6.2.post1+cu128torch2.9cxx11abitrueglibc228"
CAUSAL_CONV1D_WHEEL = Path(
    "/home/caoxiangyu/.cache/janus-ts/wheels/"
    "causal_conv1d-1.6.2.post1+cu128torch2.9cxx11abitrueglibc228-"
    "cp311-cp311-linux_x86_64.whl"
)
CAUSAL_CONV1D_WHEEL_SHA256 = (
    "33a15b2e298de90aa095680bbaa1025dc43a1a27b07488c5e060c2cea28b4ed0"
)
CAUSAL_CONV1D_EXTENSION_SHA256 = (
    "f8f840d36f5b822cf8a36e69653a68f4b53b387a8dcd61294e31577a5cfa0c76"
)
CAUSAL_CONV1D_SOURCE_COMMIT = "4f6ae4e26ae5fe8af9372f8d312ab25cc4595223"
CAUSAL_CONV1D_PATCH_SHA256 = (
    "ef259fe6a265096e02a20da5659037e9f5f4918c4eb74fb562360eac037c01f5"
)
EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.9.1",
    "transformers": "5.9.0",
    "causal-conv1d": CAUSAL_CONV1D_VERSION,
    "fla-core": "0.5.0",
    "flash-linear-attention": "0.5.0",
}
_MAX_GLIBC = (2, 28)
_VERSION_PATTERN = re.compile(r"(?<![A-Z0-9_])GLIBC_(\d+(?:\.\d+)+)")


class NativeStackError(RuntimeError):
    """The installed CUDA-extension stack differs from the frozen build."""


def _numeric_version(value: str) -> tuple[int, ...]:
    return tuple(int(component) for component in value.split("."))


def _glibc_versions(symbol_table: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(_VERSION_PATTERN.findall(symbol_table)),
            key=_numeric_version,
        )
    )


def _objdump_symbols(path: Path) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/objdump", "-T", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeStackError(f"cannot inspect native extension {path}: {exc}") from exc
    return result.stdout


def audit_native_stack() -> dict[str, Any]:
    """Import and hash the exact SM89 wheel without executing a CUDA kernel."""

    install_frozen_environment()
    installed: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not-installed"
        installed[package] = actual
        if actual != expected:
            mismatches.append(f"{package}={actual!r}, expected {expected!r}")

    if not CAUSAL_CONV1D_WHEEL.is_file():
        mismatches.append(f"missing frozen wheel {CAUSAL_CONV1D_WHEEL}")
        wheel_hash = None
    else:
        wheel_hash = sha256_file(CAUSAL_CONV1D_WHEEL)
        if wheel_hash != CAUSAL_CONV1D_WHEEL_SHA256:
            mismatches.append(f"causal-conv1d wheel sha256={wheel_hash}")

    project_root = Path(__file__).resolve().parents[2]
    patch_path = project_root / "patches" / "causal-conv1d-v1.6.2.post1-sm89.patch"
    patch_hash = sha256_file(patch_path) if patch_path.is_file() else None
    if patch_hash != CAUSAL_CONV1D_PATCH_SHA256:
        mismatches.append(f"causal-conv1d patch sha256={patch_hash}")

    # Import torch first so its private shared libraries are already visible to
    # the extension loader. No tensor is created and no CUDA kernel is run.
    torch = importlib.import_module("torch")
    causal_cuda = importlib.import_module("causal_conv1d_cuda")
    importlib.import_module("fla")
    qwen = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    extension_path = Path(causal_cuda.__file__).resolve()
    extension_hash = sha256_file(extension_path)
    if extension_hash != CAUSAL_CONV1D_EXTENSION_SHA256:
        mismatches.append(f"causal-conv1d extension sha256={extension_hash}")
    if torch.version.cuda != "12.8":
        mismatches.append(f"torch CUDA={torch.version.cuda!r}, expected '12.8'")

    functions = {
        "causal_conv1d_fn": "causal_conv1d",
        "causal_conv1d_update": "causal_conv1d",
        "chunk_gated_delta_rule": "fla",
        "fused_recurrent_gated_delta_rule": "fla",
    }
    resolved: dict[str, str] = {}
    if qwen.is_fast_path_available is not True:
        mismatches.append("Qwen linear-attention fast path is unavailable")
    for name, expected_prefix in functions.items():
        module_name = getattr(getattr(qwen, name, None), "__module__", "")
        resolved[name] = module_name
        if not module_name.startswith(expected_prefix):
            mismatches.append(f"{name} resolved to {module_name!r}")

    glibc_versions = _glibc_versions(_objdump_symbols(extension_path))
    if not glibc_versions:
        mismatches.append("native extension exposes no GLIBC symbol requirements")
        max_glibc = None
    else:
        max_glibc = glibc_versions[-1]
        if _numeric_version(max_glibc) > _MAX_GLIBC:
            mismatches.append(f"native extension requires GLIBC_{max_glibc} (>2.28)")
    if mismatches:
        raise NativeStackError("native stack audit failed: " + "; ".join(mismatches))
    return {
        "status": "pass",
        "packages": installed,
        "torch_cuda": torch.version.cuda,
        "source_commit": CAUSAL_CONV1D_SOURCE_COMMIT,
        "wheel": str(CAUSAL_CONV1D_WHEEL),
        "wheel_sha256": wheel_hash,
        "patch_sha256": patch_hash,
        "extension": str(extension_path),
        "extension_sha256": extension_hash,
        "glibc_versions": list(glibc_versions),
        "max_glibc": max_glibc,
        "qwen_fast_path": True,
        "resolved_functions": resolved,
    }


__all__ = [
    "CAUSAL_CONV1D_EXTENSION_SHA256",
    "CAUSAL_CONV1D_PATCH_SHA256",
    "CAUSAL_CONV1D_SOURCE_COMMIT",
    "CAUSAL_CONV1D_VERSION",
    "CAUSAL_CONV1D_WHEEL",
    "CAUSAL_CONV1D_WHEEL_SHA256",
    "EXPECTED_PACKAGE_VERSIONS",
    "NativeStackError",
    "audit_native_stack",
]
