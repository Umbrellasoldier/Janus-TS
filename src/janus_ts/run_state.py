"""Stable run identity and monotonic workflow state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import read_complete_manifest, sha256_file, sha256_json, write_json
from .config import ExperimentConfig
from .constants import MODEL_ID, MODEL_REVISION


class RunStateError(RuntimeError):
    """A run path contains incompatible or regressing state."""


@dataclass(frozen=True, slots=True)
class RunIdentity:
    fingerprint: str
    experiment: str
    config_sha256: str
    data_fingerprint: str
    pissa_manifest_sha256: str
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunPaths:
    project: Path
    local: Path

    @property
    def local_checkpoints(self) -> Path:
        return self.local / "checkpoints"

    @property
    def durable_checkpoints(self) -> Path:
        return self.project / "checkpoints"

    @property
    def portable_adapters(self) -> Path:
        return self.project / "portable_adapters"

    @property
    def evaluations(self) -> Path:
        return self.project / "evaluations"


_STAGE_ORDER = {
    "initialized": 0,
    "training": 1,
    "validation": 2,
    "selected": 3,
    "testing": 4,
    "complete": 5,
    "failed": 99,
}


def build_run_identity(
    config: ExperimentConfig,
    *,
    processed_data_dir: str | Path,
    pissa_bundle_dir: str | Path,
    micro_batch_size_per_gpu: int,
    gradient_accumulation_steps: int,
) -> RunIdentity:
    allowed_geometry = {
        (
            config.train.micro_batch_size_per_gpu,
            config.train.gradient_accumulation_steps,
        ),
        (
            config.train.candidate_micro_batch_size_per_gpu,
            config.train.candidate_gradient_accumulation_steps,
        ),
    }
    geometry = (micro_batch_size_per_gpu, gradient_accumulation_steps)
    if geometry not in allowed_geometry:
        raise RunStateError(
            f"run identity has unconfirmed batch geometry {geometry!r}; "
            f"allowed {sorted(allowed_geometry)!r}"
        )
    if (
        micro_batch_size_per_gpu
        * config.runtime.required_gpu_count
        * gradient_accumulation_steps
        != config.train.global_batch_size
    ):
        raise RunStateError("run identity batch geometry changes the global batch size")
    data_manifest = read_complete_manifest(processed_data_dir)
    data_fingerprint = data_manifest.get("fingerprint")
    if not isinstance(data_fingerprint, str) or not data_fingerprint:
        raise RunStateError("processed-data manifest has no fingerprint")
    pissa_manifest_path = Path(pissa_bundle_dir) / "manifest.json"
    read_complete_manifest(pissa_bundle_dir)
    pissa_manifest_sha256 = sha256_file(pissa_manifest_path)
    payload = {
        "experiment": config.experiment,
        "config_sha256": config.sha256,
        "data_fingerprint": data_fingerprint,
        "pissa_manifest_sha256": pissa_manifest_sha256,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    return RunIdentity(fingerprint=sha256_json(payload), **payload)


def resolve_run_paths(config: ExperimentConfig, identity: RunIdentity) -> RunPaths:
    name = f"{config.data.name}-{identity.fingerprint[:16]}"
    return RunPaths(
        project=config.runtime.artifacts_root / "runs" / name,
        local=config.runtime.local_cache_root / "runs" / identity.fingerprint,
    )


def initialize_run(paths: RunPaths, identity: RunIdentity) -> dict[str, Any]:
    paths.project.mkdir(parents=True, exist_ok=True)
    paths.local.mkdir(parents=True, exist_ok=True)
    identity_path = paths.project / "identity.json"
    if identity_path.exists():
        import json

        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity.to_dict():
            raise RunStateError(f"run identity drift at {paths.project}")
    else:
        write_json(identity_path, identity.to_dict())
    state_path = paths.project / "state.json"
    if state_path.exists():
        import json

        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_fingerprint") != identity.fingerprint:
            raise RunStateError("run state fingerprint differs from identity")
        return state
    state = {
        "run_fingerprint": identity.fingerprint,
        "stage": "initialized",
        "global_step": 0,
        "selected_epoch": None,
        "test_evaluated": False,
    }
    write_json(state_path, state)
    return state


def update_run_state(
    paths: RunPaths,
    identity: RunIdentity,
    *,
    stage: str,
    global_step: int,
    **updates: Any,
) -> dict[str, Any]:
    import json

    if stage not in _STAGE_ORDER:
        raise RunStateError(f"unknown run stage {stage!r}")
    state_path = paths.project / "state.json"
    try:
        current = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunStateError(f"cannot read run state: {exc}") from exc
    if current.get("run_fingerprint") != identity.fingerprint:
        raise RunStateError("cannot update a foreign run state")
    old_stage = current.get("stage")
    if old_stage not in _STAGE_ORDER:
        raise RunStateError(f"stored run stage is invalid: {old_stage!r}")
    if old_stage == "failed" and stage != "failed":
        raise RunStateError("failed run state cannot be silently resumed")
    if stage != "failed" and _STAGE_ORDER[stage] < _STAGE_ORDER[old_stage]:
        raise RunStateError(f"run stage would regress from {old_stage!r} to {stage!r}")
    old_step = current.get("global_step")
    if not isinstance(global_step, int) or global_step < 0:
        raise RunStateError("global_step must be a non-negative integer")
    if not isinstance(old_step, int) or global_step < old_step:
        raise RunStateError(f"global step would regress from {old_step!r} to {global_step}")
    updated = {**current, **updates, "stage": stage, "global_step": global_step}
    write_json(state_path, updated)
    return updated


__all__ = [
    "RunIdentity",
    "RunPaths",
    "RunStateError",
    "build_run_identity",
    "initialize_run",
    "resolve_run_paths",
    "update_run_state",
]
