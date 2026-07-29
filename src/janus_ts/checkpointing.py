"""Crash-safe ZeRO-3/PEFT resume checkpoints and portable adapters.

DeepSpeed 0.19.2 can omit frozen parameters from a ZeRO-3 checkpoint.  That is
safe for Janus-TS because the frozen PiSSA residual base is an immutable,
content-addressed input to every run.  Resume therefore follows one exact
sequence:

1. construct a *fresh* engine from the identical residual base and rank-32
   PiSSA initialization adapter;
2. load the ZeRO-3 checkpoint with ``load_module_strict=False`` (the only
   missing module keys are frozen base parameters);
3. restore optimizer, scheduler, Trainer state and per-rank RNG state; and
4. gather only LoRA tensors and compare them bit-for-bit with the separately
   saved adapter.

Falling back to a full frozen-base checkpoint is intentionally unsupported.
It would gather/write the 27B base and conceal a broken immutable-base
contract.  All ranks must call the save/load methods, just as required by
DeepSpeed itself.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import math
import os
import random
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from transformers import TrainerCallback

from .artifacts import (
    COMPLETE_MARKER,
    ArtifactError,
    canonical_json_bytes,
    mark_complete,
    read_complete_manifest,
    sha256_file,
    sha256_json,
    write_json,
)
from .modeling import EXPECTED_TRAINABLE_PARAMETERS, validate_portable_adapter_config

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_DIR_PREFIX = "resume-step-"
DEEPSPEED_SUBDIR = "deepspeed"
RESUME_ADAPTER_SUBDIR = "resume_adapter"
PORTABLE_ADAPTER_SUBDIR = "portable_adapter"
RNG_SUBDIR = "rng"
TRAINER_STATE_NAME = "trainer_state.json"
SCHEDULER_STATE_NAME = "scheduler.pt"

CheckpointKind = Literal["checkpoint0", "rolling", "epoch"]
_VALID_KINDS: frozenset[str] = frozenset({"checkpoint0", "rolling", "epoch"})
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_KEY_RE = re.compile(r"(\.lora_[AB])\.([^.]+)(\.)")
_PACKAGES = ("torch", "transformers", "peft", "deepspeed", "accelerate")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is unsafe, incomplete, or incompatible."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Content identities that must agree before any state is loaded."""

    run_fingerprint: str
    config_fingerprint: str
    data_fingerprint: str
    model_fingerprint: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None:
                raise CheckpointError(f"{field.name} must be a lowercase SHA-256 digest")

    @classmethod
    def derive(
        cls, *, config_fingerprint: str, data_fingerprint: str, model_fingerprint: str
    ) -> CheckpointIdentity:
        """Derive the run identity without relying on paths or timestamps."""

        run = sha256_json(
            {
                "config_fingerprint": config_fingerprint,
                "data_fingerprint": data_fingerprint,
                "model_fingerprint": model_fingerprint,
            }
        )
        return cls(run, config_fingerprint, data_fingerprint, model_fingerprint)

    def as_dict(self) -> dict[str, str]:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    """A validated complete checkpoint selected for resume."""

    path: Path
    manifest: Mapping[str, Any]

    @property
    def global_step(self) -> int:
        return int(self.manifest["global_step"])


@dataclass(frozen=True, slots=True)
class RestoredTrainingState:
    """State returned after DeepSpeed and RNG restoration."""

    checkpoint: ResumeCheckpoint
    trainer_state: Mapping[str, Any]
    client_state: Mapping[str, Any]


class DistributedCoordinator(Protocol):
    """Minimal process-group surface used by checkpoint code and unit mocks."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def barrier(self) -> None: ...

    def broadcast(self, value: Any, *, source: int = 0) -> Any: ...


class TorchDistributedCoordinator:
    """Coordinator backed by the initialized torch process group."""

    def __init__(self) -> None:
        import torch.distributed as dist

        self._dist = dist
        self._distributed = dist.is_available() and dist.is_initialized()

    @property
    def rank(self) -> int:
        return self._dist.get_rank() if self._distributed else 0

    @property
    def world_size(self) -> int:
        return self._dist.get_world_size() if self._distributed else 1

    def barrier(self) -> None:
        if self._distributed:
            self._dist.barrier()

    def broadcast(self, value: Any, *, source: int = 0) -> Any:
        if not self._distributed:
            return value
        values = [value if self.rank == source else None]
        self._dist.broadcast_object_list(values, src=source)
        return values[0]


def _installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _engine_module(engine: Any) -> Any:
    module = getattr(engine, "module", None)
    if module is None:
        raise CheckpointError("DeepSpeed engine has no underlying module")
    return module


def _logical_numel(parameter: Any) -> int:
    value = getattr(parameter, "ds_numel", None)
    return int(value) if value is not None else int(parameter.numel())


def _is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def _gather_context(parameter: Any) -> Any:
    if not hasattr(parameter, "ds_id"):
        return nullcontext()
    import deepspeed

    # No tensor is modified, so modifier_rank must remain None.  Every rank
    # enters the context, while only rank zero retains a CPU clone.
    return deepspeed.zero.GatheredParameters([parameter], modifier_rank=None)


def gather_lora_state_dict(
    engine: Any,
    coordinator: DistributedCoordinator,
    *,
    expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> dict[str, Any] | None:
    """Gather only trainable LoRA tensors, one parameter at a time.

    The function never calls ``model.state_dict()`` and therefore never
    materializes the frozen 27B base.  All distributed ranks must enter it.
    """

    import torch

    module = _engine_module(engine)
    trainable = sorted(
        (
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    non_lora = [name for name, _ in trainable if not _is_lora_parameter(name)]
    unsupported_dtype = [
        name
        for name, parameter in trainable
        if parameter.dtype not in {torch.bfloat16, torch.float32}
    ]
    trainable_count = sum(_logical_numel(parameter) for _, parameter in trainable)
    if (
        not trainable
        or non_lora
        or unsupported_dtype
        or trainable_count != expected_trainable_parameters
    ):
        raise CheckpointError(
            "trainable adapter contract failed: "
            f"parameters={trainable_count:,} (expected {expected_trainable_parameters:,}), "
            f"non_lora={non_lora[:8]!r}, unsupported_dtype={unsupported_dtype[:8]!r}"
        )

    gathered: dict[str, Any] | None = {} if coordinator.rank == 0 else None
    for name, parameter in trainable:
        with _gather_context(parameter):
            if gathered is not None:
                gathered[name] = (
                    parameter.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
                )
    return gathered


def normalize_lora_state_dict(
    state_dict: Mapping[str, Any], *, adapter_name: str = "default"
) -> dict[str, Any]:
    """Remove the PEFT adapter-name segment from raw module keys."""

    result: dict[str, Any] = {}
    for raw_name, tensor in state_dict.items():
        match = _ADAPTER_KEY_RE.search(raw_name)
        if match is None or match.group(2) != adapter_name:
            raise CheckpointError(f"unexpected LoRA parameter key: {raw_name}")
        name = raw_name[: match.start()] + match.group(1) + match.group(3) + raw_name[match.end() :]
        if name in result:
            raise CheckpointError(f"adapter key normalization collision: {name}")
        result[name] = tensor
    return result


def convert_pissa_to_portable_state(
    trained_state: Mapping[str, Any], initial_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Perform PEFT's exact PiSSA-to-rank-2r LoRA factor conversion.

    ``A' = [A; A0]`` and ``B' = [B, -B0]`` represent
    ``B A - B0 A0`` relative to the untouched base.
    """

    import torch

    if set(trained_state) != set(initial_state):
        missing = sorted(set(initial_state) - set(trained_state))
        extra = sorted(set(trained_state) - set(initial_state))
        raise CheckpointError(
            f"trained/initial adapter keys differ: missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    portable: dict[str, Any] = {}
    for name in sorted(trained_state):
        trained = trained_state[name]
        initial = initial_state[name]
        if trained.shape != initial.shape or trained.dtype != initial.dtype:
            raise CheckpointError(
                f"trained/initial tensor mismatch for {name}: "
                f"{tuple(trained.shape)}/{trained.dtype} != "
                f"{tuple(initial.shape)}/{initial.dtype}"
            )
        if ".lora_A.weight" in name:
            value = torch.cat((trained, initial), dim=0)
        elif ".lora_B.weight" in name:
            value = torch.cat((trained, -initial), dim=1)
        else:
            raise CheckpointError(f"portable conversion found a non-A/B tensor: {name}")
        portable[name] = value.contiguous()
    return portable


def _adapter_config(module: Any, adapter_name: str) -> Any:
    configs = getattr(module, "peft_config", None)
    if not isinstance(configs, Mapping) or adapter_name not in configs:
        raise CheckpointError(f"module has no PEFT adapter named {adapter_name!r}")
    config = copy.deepcopy(configs[adapter_name])
    if getattr(config, "init_lora_weights", None) is not True:
        raise CheckpointError(
            "training adapter must be loaded from the prepared residual base with "
            "init_lora_weights=True"
        )
    if getattr(config, "r", None) != 32 or not math.isclose(
        float(getattr(config, "lora_alpha", math.nan)), 16.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise CheckpointError("training adapter is not the frozen rank-32/alpha-16 adapter")
    if getattr(config, "use_rslora", None) is not True:
        raise CheckpointError("training adapter must use rsLoRA")
    return config


def _write_adapter(
    directory: Path,
    module: Any,
    normalized_state: Mapping[str, Any],
    *,
    adapter_name: str,
) -> None:
    from safetensors.torch import save_file

    config = _adapter_config(module, adapter_name)
    config.inference_mode = True
    directory.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(directory)
    save_file(
        dict(normalized_state),
        directory / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )


def _hash_tree(root: Path) -> str:
    if not root.is_dir():
        raise CheckpointError(f"artifact directory does not exist: {root}")
    inventory = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }
    if not inventory:
        raise CheckpointError(f"artifact directory is empty: {root}")
    return sha256_json(inventory)


def _write_portable_adapter(
    directory: Path,
    module: Any,
    normalized_trained_state: Mapping[str, Any],
    *,
    initial_adapter_dir: Path,
    adapter_name: str,
) -> str:
    from safetensors.torch import load_file, save_file

    initial_weights = initial_adapter_dir / "adapter_model.safetensors"
    initial_config = initial_adapter_dir / "adapter_config.json"
    if not initial_weights.is_file() or not initial_config.is_file():
        raise CheckpointError(f"incomplete immutable PiSSA initialization: {initial_adapter_dir}")

    before = _hash_tree(initial_adapter_dir)
    initial_state = load_file(initial_weights, device="cpu")
    portable_state = convert_pissa_to_portable_state(normalized_trained_state, initial_state)

    config = _adapter_config(module, adapter_name)
    config.inference_mode = True
    config.init_lora_weights = True
    config.r *= 2
    config.lora_alpha *= math.sqrt(2.0)
    if getattr(config, "rank_pattern", None) or getattr(config, "alpha_pattern", None):
        raise CheckpointError("portable rsLoRA conversion forbids rank/alpha patterns")

    directory.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(directory)
    save_file(
        portable_state,
        directory / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    validate_portable_adapter_config(directory / "adapter_config.json", module.config)

    after = _hash_tree(initial_adapter_dir)
    if after != before:
        raise CheckpointError("immutable PiSSA initialization changed during portable export")
    return before


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"{description} must be a JSON object: {path}")
    return payload


def _link_or_copy_file(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != getattr(os, "EXDEV", 18):
            raise
        shutil.copy2(source, destination)


def _validate_materialized_train_loss_checkpoint(
    root: Path,
    *,
    identity: CheckpointIdentity,
    source_checkpoint_fingerprint: str,
    train_loss: float,
) -> Mapping[str, Any]:
    try:
        manifest = read_complete_manifest(root)
    except ArtifactError as exc:
        raise CheckpointError(str(exc)) from exc
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported materialized checkpoint schema")
    if manifest.get("kind") != "train-loss":
        raise CheckpointError("materialized checkpoint kind is not train-loss")
    if _manifest_identity(manifest) != identity:
        raise CheckpointError("materialized checkpoint belongs to another run")
    if manifest.get("source_checkpoint_fingerprint") != source_checkpoint_fingerprint:
        raise CheckpointError("materialized checkpoint source differs")
    if manifest.get("train_loss") != train_loss:
        raise CheckpointError("materialized checkpoint train loss differs")
    step, epoch = _validate_step_epoch(
        manifest.get("global_step"),
        manifest.get("epoch"),
        kind="rolling",
    )
    if step <= 0 or epoch <= 0:
        raise CheckpointError("materialized checkpoint has an invalid training position")
    _verify_payload_inventory(root, manifest.get("payload_inventory"))
    expected = {
        f"{RESUME_ADAPTER_SUBDIR}/adapter_config.json",
        f"{RESUME_ADAPTER_SUBDIR}/adapter_model.safetensors",
        f"{PORTABLE_ADAPTER_SUBDIR}/adapter_config.json",
        f"{PORTABLE_ADAPTER_SUBDIR}/adapter_model.safetensors",
    }
    inventory = manifest["payload_inventory"]
    if set(inventory) != expected:
        raise CheckpointError("materialized checkpoint has unexpected payload files")
    return manifest


def materialize_train_loss_checkpoint(
    source_checkpoint: str | Path,
    destination: str | Path,
    *,
    identity: CheckpointIdentity,
    initial_adapter_dir: str | Path,
    portable_config_template: str | Path,
    expected_initial_adapter_fingerprint: str,
    train_loss: float,
) -> Path:
    """Convert one retained rolling checkpoint into a compact eval checkpoint.

    The rolling tree is validated as a complete resume checkpoint.  Its
    rank-32 adapter is then combined with the immutable PiSSA initialization
    to produce the exact rank-64 portable adapter used by formal evaluation.
    Optimizer, scheduler, RNG, and DeepSpeed shards are deliberately omitted.
    """

    from safetensors.torch import load_file, save_file

    if isinstance(train_loss, bool) or not isinstance(train_loss, (int, float)):
        raise CheckpointError(f"invalid train loss: {train_loss!r}")
    loss = float(train_loss)
    if not math.isfinite(loss) or loss < 0.0:
        raise CheckpointError(f"invalid train loss: {train_loss!r}")
    if (
        not isinstance(expected_initial_adapter_fingerprint, str)
        or _FINGERPRINT_RE.fullmatch(expected_initial_adapter_fingerprint) is None
    ):
        raise CheckpointError("expected PiSSA initialization fingerprint is invalid")

    source = Path(source_checkpoint).resolve(strict=True)
    try:
        source_manifest = read_complete_manifest(source)
    except ArtifactError as exc:
        raise CheckpointError(str(exc)) from exc
    step, epoch, kind = _validate_checkpoint_payload(
        source,
        source_manifest,
        identity,
        expected_world_size=2,
    )
    if kind != "rolling":
        raise CheckpointError(f"train-loss materialization requires rolling, got {kind!r}")
    source_fingerprint = sha256_file(source / "manifest.json")

    target = Path(destination)
    if target.exists():
        _validate_materialized_train_loss_checkpoint(
            target,
            identity=identity,
            source_checkpoint_fingerprint=source_fingerprint,
            train_loss=loss,
        )
        return target

    initial_root = Path(initial_adapter_dir).resolve(strict=True)
    initial_fingerprint = _hash_tree(initial_root)
    if initial_fingerprint != expected_initial_adapter_fingerprint:
        raise CheckpointError("PiSSA initialization fingerprint differs from final checkpoint")

    resume_root = source / RESUME_ADAPTER_SUBDIR
    resume_files = {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    actual_resume_files = {
        path.relative_to(resume_root).as_posix()
        for path in resume_root.rglob("*")
        if path.is_file()
    }
    if actual_resume_files != resume_files:
        raise CheckpointError(
            f"rolling resume adapter files differ: {sorted(actual_resume_files)!r}"
        )

    trained_config = _read_json_object(
        resume_root / "adapter_config.json",
        description="rolling adapter config",
    )
    portable_config = copy.deepcopy(trained_config)
    if (
        portable_config.get("r") != 32
        or not math.isclose(
            float(portable_config.get("lora_alpha", math.nan)),
            16.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or portable_config.get("use_rslora") is not True
        or portable_config.get("init_lora_weights") is not True
        or portable_config.get("rank_pattern") not in ({}, None)
        or portable_config.get("alpha_pattern") not in ({}, None)
    ):
        raise CheckpointError("rolling adapter config violates the rank-32 PiSSA contract")
    portable_config["r"] = 64
    portable_config["lora_alpha"] = 16.0 * math.sqrt(2.0)
    portable_config["inference_mode"] = True

    template_path = Path(portable_config_template).resolve(strict=True)
    template = _read_json_object(template_path, description="portable config template")
    if portable_config != template:
        raise CheckpointError("derived portable config differs from final checkpoint template")

    initial_weights = initial_root / "adapter_model.safetensors"
    trained_state = load_file(resume_root / "adapter_model.safetensors", device="cpu")
    initial_state = load_file(initial_weights, device="cpu")
    portable_state = convert_pissa_to_portable_state(trained_state, initial_state)

    from .artifacts import atomic_directory

    with atomic_directory(target) as staging:
        staged_resume = staging / RESUME_ADAPTER_SUBDIR
        staged_resume.mkdir()
        for name in sorted(resume_files):
            _link_or_copy_file(resume_root / name, staged_resume / name)

        staged_portable = staging / PORTABLE_ADAPTER_SUBDIR
        staged_portable.mkdir()
        write_json(staged_portable / "adapter_config.json", template)
        save_file(
            portable_state,
            staged_portable / "adapter_model.safetensors",
            metadata={"format": "pt"},
        )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "train-loss",
            **identity.as_dict(),
            "global_step": step,
            "epoch": epoch,
            "world_size": source_manifest["world_size"],
            "exclude_frozen_parameters": True,
            "pissa_initial_adapter_fingerprint": initial_fingerprint,
            "train_loss": loss,
            "source_checkpoint_kind": kind,
            "source_checkpoint_fingerprint": source_fingerprint,
            "portable_config_template_sha256": sha256_file(template_path),
            "payload_inventory": _payload_inventory(staging),
        }
        mark_complete(staging, manifest)

    _validate_materialized_train_loss_checkpoint(
        target,
        identity=identity,
        source_checkpoint_fingerprint=source_fingerprint,
        train_loss=loss,
    )
    return target


def _capture_rng_state() -> dict[str, Any]:
    import torch

    cuda_initialized = torch.cuda.is_initialized()
    cuda_device = torch.cuda.current_device() if cuda_initialized else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        # A torchrun rank sees both cards. Saving every visible generator would
        # create a second CUDA context on the peer card, so each rank owns only
        # its already-initialized current device.
        "cuda_device": cuda_device,
        "torch_cuda": (
            torch.cuda.get_rng_state(cuda_device) if cuda_initialized else None
        ),
    }


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _restore_rng_state(path: Path) -> None:
    import torch

    # This is a trusted local artifact whose digest was checked before load.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"python", "numpy", "torch_cpu", "cuda_device", "torch_cuda"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise CheckpointError(f"invalid RNG state payload: {path}")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if payload["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        current_device = torch.cuda.current_device()
        if payload["cuda_device"] != current_device:
            raise CheckpointError(
                "rank-local CUDA RNG device differs: "
                f"saved={payload['cuda_device']!r}, current={current_device}"
            )
        torch.cuda.set_rng_state(payload["torch_cuda"], device=current_device)
    elif payload["cuda_device"] is not None:
        raise CheckpointError("CUDA RNG payload has a device but no generator state")


def _trainer_state_dict(state: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        payload = dataclasses.asdict(state)
    elif isinstance(state, Mapping):
        payload = dict(state)
    else:
        raise CheckpointError("Trainer state must be a dataclass or mapping")
    # Enforce JSON safety before the distributed save begins.
    return json.loads(canonical_json_bytes(payload))


def _scheduler_state_dict(scheduler: Any, *, global_step: int) -> dict[str, Any]:
    if scheduler is None or not callable(getattr(scheduler, "state_dict", None)):
        raise CheckpointError("Trainer learning-rate scheduler is unavailable")
    state = scheduler.state_dict()
    if not isinstance(state, Mapping):
        raise CheckpointError("learning-rate scheduler state must be a mapping")
    payload = copy.deepcopy(dict(state))
    last_epoch = payload.get("last_epoch")
    step_count = payload.get("_step_count")
    if last_epoch != global_step or step_count != global_step + 1:
        raise CheckpointError(
            "scheduler/Trainer step mismatch: "
            f"last_epoch={last_epoch!r}, _step_count={step_count!r}, "
            f"global_step={global_step}"
        )
    return payload


def _scheduler_states_equal(expected: Any, actual: Any) -> bool:
    import torch

    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        return (
            isinstance(expected, torch.Tensor)
            and isinstance(actual, torch.Tensor)
            and torch.equal(expected, actual)
        )
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        return (
            isinstance(expected, Mapping)
            and isinstance(actual, Mapping)
            and set(expected) == set(actual)
            and all(_scheduler_states_equal(expected[key], actual[key]) for key in expected)
        )
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        return (
            isinstance(expected, (list, tuple))
            and isinstance(actual, (list, tuple))
            and len(expected) == len(actual)
            and all(
                _scheduler_states_equal(left, right)
                for left, right in zip(expected, actual, strict=True)
            )
        )
    return bool(expected == actual)


def _validate_step_epoch(global_step: Any, epoch: Any, *, kind: str) -> tuple[int, float]:
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise CheckpointError(f"invalid global step: {global_step!r}")
    if kind == "checkpoint0" and global_step != 0:
        raise CheckpointError("checkpoint0 must be saved at global_step=0")
    if kind == "rolling" and global_step == 0:
        raise CheckpointError("rolling checkpoints require at least one optimizer update")
    try:
        epoch_value = float(epoch)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid epoch: {epoch!r}") from exc
    if not math.isfinite(epoch_value) or epoch_value < 0:
        raise CheckpointError(f"invalid epoch: {epoch!r}")
    if kind == "epoch" and (
        epoch_value < 1 or not math.isclose(epoch_value, round(epoch_value), abs_tol=1e-8)
    ):
        raise CheckpointError(
            f"durable epoch checkpoint requires an integral completed epoch: {epoch}"
        )
    return global_step, epoch_value


def _payload_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {COMPLETE_MARKER, "manifest.json"}:
            continue
        entry: dict[str, Any] = {"size_bytes": path.stat().st_size}
        if not relative.startswith(f"{DEEPSPEED_SUBDIR}/"):
            entry["sha256"] = sha256_file(path)
        inventory[relative] = entry
    return inventory


def _verify_payload_inventory(root: Path, inventory: Any) -> None:
    if not isinstance(inventory, Mapping) or not inventory:
        raise CheckpointError(f"checkpoint has no payload inventory: {root}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {COMPLETE_MARKER, "manifest.json"}
    }
    expected_paths = set(inventory)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise CheckpointError(
            f"checkpoint payload inventory differs: missing={missing[:8]!r}, extra={extra[:8]!r}"
        )
    for relative, raw_entry in inventory.items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise CheckpointError(f"unsafe payload path in manifest: {relative!r}")
        if not isinstance(raw_entry, Mapping):
            raise CheckpointError(f"invalid payload inventory entry: {relative}")
        path = root / relative
        expected_size = raw_entry.get("size_bytes")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise CheckpointError(f"checkpoint payload missing or resized: {path}")
        expected_hash = raw_entry.get("sha256")
        if expected_hash is not None and sha256_file(path) != expected_hash:
            raise CheckpointError(f"checkpoint payload digest mismatch: {path}")


def _validate_deepspeed_payload(root: Path, *, tag: str, world_size: int) -> None:
    tag_root = root / DEEPSPEED_SUBDIR / tag
    if not tag_root.is_dir():
        raise CheckpointError(f"DeepSpeed tag directory is missing: {tag_root}")
    model_states = tuple(tag_root.rglob("*model_states.pt"))
    optimizer_states = tuple(tag_root.rglob("*optim_states.pt"))
    if not model_states or len(optimizer_states) < world_size:
        raise CheckpointError(
            "incomplete ZeRO checkpoint: "
            f"model_state_files={len(model_states)}, "
            f"optimizer_state_files={len(optimizer_states)}, "
            f"world_size={world_size}"
        )
    empty = [path for path in (*model_states, *optimizer_states) if path.stat().st_size == 0]
    if empty:
        raise CheckpointError(f"empty DeepSpeed state files: {empty[:4]!r}")


def _validate_checkpoint_payload(
    root: Path,
    manifest: Mapping[str, Any],
    identity: CheckpointIdentity,
    *,
    expected_world_size: int | None = None,
) -> tuple[int, float, CheckpointKind]:
    """Apply the complete resume-selector validation contract to one tree."""

    step, epoch, kind = _validate_manifest(manifest, identity)
    _verify_payload_inventory(root, manifest.get("payload_inventory"))
    world_size = manifest.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise CheckpointError(f"invalid checkpoint world size: {world_size!r}")
    if expected_world_size is not None and world_size != expected_world_size:
        raise CheckpointError(
            f"checkpoint world_size={world_size}, current={expected_world_size}"
        )
    tag = manifest.get("deepspeed_tag")
    expected_tag = f"global_step{step:09d}"
    if not isinstance(tag, str) or tag != expected_tag:
        raise CheckpointError(
            f"checkpoint DeepSpeed tag={tag!r}, expected {expected_tag!r}"
        )
    _validate_deepspeed_payload(root, tag=tag, world_size=world_size)
    return step, epoch, kind


def _manifest_identity(manifest: Mapping[str, Any]) -> CheckpointIdentity:
    try:
        return CheckpointIdentity(
            run_fingerprint=manifest["run_fingerprint"],
            config_fingerprint=manifest["config_fingerprint"],
            data_fingerprint=manifest["data_fingerprint"],
            model_fingerprint=manifest["model_fingerprint"],
        )
    except KeyError as exc:
        raise CheckpointError(f"checkpoint manifest lacks {exc.args[0]!r}") from exc


def _validate_manifest(
    manifest: Mapping[str, Any], identity: CheckpointIdentity
) -> tuple[int, float, CheckpointKind]:
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema: {manifest.get('schema_version')!r}")
    kind = manifest.get("kind")
    if kind not in _VALID_KINDS:
        raise CheckpointError(f"invalid checkpoint kind: {kind!r}")
    if _manifest_identity(manifest) != identity:
        raise CheckpointError("checkpoint fingerprints do not match the current run")
    step, epoch = _validate_step_epoch(
        manifest.get("global_step"), manifest.get("epoch"), kind=kind
    )
    if manifest.get("exclude_frozen_parameters") is not True:
        raise CheckpointError("checkpoint did not record the frozen-base exclusion contract")
    versions = manifest.get("package_versions")
    if versions != _installed_versions():
        raise CheckpointError(
            "checkpoint package versions differ: "
            f"saved={versions!r}, current={_installed_versions()!r}"
        )
    if manifest.get("scheduler_state_name") != SCHEDULER_STATE_NAME:
        raise CheckpointError("checkpoint lacks the external Trainer scheduler state")
    if not isinstance(manifest.get("scheduler_class"), str):
        raise CheckpointError("checkpoint lacks the scheduler class identity")
    return step, epoch, kind  # type: ignore[return-value]


def select_resume_checkpoint(
    roots: Sequence[str | Path], *, identity: CheckpointIdentity
) -> ResumeCheckpoint | None:
    """Select the greatest manifest step, never directory name or mtime.

    For identical steps a later root wins.  Callers should list local rolling
    storage first and durable project storage second, making the durable epoch
    copy the deterministic tie winner.
    """

    candidates: list[tuple[int, int, str, Path, Mapping[str, Any]]] = []
    for root_priority, raw_root in enumerate(roots):
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                manifest = read_complete_manifest(child)
                step, _, _ = _validate_checkpoint_payload(child, manifest, identity)
            except (ArtifactError, CheckpointError, OSError):
                continue
            candidates.append((step, root_priority, child.name, child, manifest))
    if not candidates:
        return None
    _, _, _, path, manifest = max(candidates, key=lambda item: item[:3])
    return ResumeCheckpoint(path=path, manifest=manifest)


class CheckpointManager:
    """Save, select, restore and rotate Janus-TS checkpoints."""

    def __init__(
        self,
        identity: CheckpointIdentity,
        *,
        coordinator: DistributedCoordinator | None = None,
        adapter_name: str = "default",
        expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
    ) -> None:
        self.identity = identity
        self.coordinator = coordinator or TorchDistributedCoordinator()
        self.adapter_name = adapter_name
        self.expected_trainable_parameters = expected_trainable_parameters

    @staticmethod
    def checkpoint_name(global_step: int) -> str:
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise CheckpointError(f"invalid global step: {global_step!r}")
        return f"{CHECKPOINT_DIR_PREFIX}{global_step:09d}"

    def destination(self, root: str | Path, global_step: int) -> Path:
        return Path(root) / self.checkpoint_name(global_step)

    def _assert_engine_world(self, engine: Any) -> None:
        engine_rank = int(getattr(engine, "global_rank", self.coordinator.rank))
        engine_world = int(getattr(engine, "world_size", self.coordinator.world_size))
        if engine_rank != self.coordinator.rank or engine_world != self.coordinator.world_size:
            raise CheckpointError(
                "engine/process-group mismatch: "
                f"engine=({engine_rank},{engine_world}), "
                f"coordinator=({self.coordinator.rank},{self.coordinator.world_size})"
            )

    def _quarantine_checkpoint0(self, target: Path, *, validation_error: str) -> Path:
        """Atomically preserve an invalid checkpoint0 owned by this run.

        The wrapper is hidden from resume selection and rotation.  Its random
        suffix avoids all clock/mtime dependence, while the nested
        ``checkpoint`` directory preserves the original tree byte-for-byte.
        """

        quarantine = target.parent / (
            f".{target.name}.quarantine."
            f"{self.identity.run_fingerprint[:12]}.{uuid.uuid4().hex}"
        )
        quarantine.mkdir(parents=False, exist_ok=False)
        receipt = quarantine / "quarantine.json"
        write_json(
            receipt,
            {
                "schema_version": 1,
                "kind": "checkpoint0-quarantine",
                **self.identity.as_dict(),
                "original_name": target.name,
                "payload_name": "checkpoint",
                "validation_error": validation_error,
            },
        )
        try:
            # The destination is inside a newly-created private wrapper and
            # therefore cannot overwrite another checkpoint. Directory rename
            # on the same filesystem is atomic.
            target.rename(quarantine / "checkpoint")
        except BaseException:
            # The original target still exists when rename fails, so removing
            # only the receipt/wrapper created above is non-destructive.
            if target.exists():
                receipt.unlink(missing_ok=True)
                with suppress(OSError):
                    quarantine.rmdir()
            raise
        return quarantine

    def _prepare_checkpoint0_destination(self, destination: Path) -> Mapping[str, Any]:
        """Choose reuse/save on rank zero and broadcast one immutable decision."""

        decision: dict[str, Any] | None
        if self.coordinator.rank == 0:
            try:
                if not destination.exists():
                    decision = {
                        "action": "save",
                        "quarantine": None,
                        "error": None,
                    }
                else:
                    if not destination.is_dir() or destination.is_symlink():
                        raise CheckpointError(
                            "existing checkpoint0 is not a regular directory; "
                            f"left untouched: {destination}"
                        )
                    try:
                        manifest = read_complete_manifest(destination)
                        existing_identity = _manifest_identity(manifest)
                    except (ArtifactError, CheckpointError, OSError) as exc:
                        raise CheckpointError(
                            "existing checkpoint0 identity cannot be established safely; "
                            f"left untouched: {destination}: {exc}"
                        ) from exc
                    if existing_identity != self.identity:
                        raise CheckpointError(
                            "existing checkpoint0 belongs to a foreign identity and was "
                            f"left untouched: {destination}"
                        )
                    try:
                        _, _, kind = _validate_checkpoint_payload(
                            destination,
                            manifest,
                            self.identity,
                            expected_world_size=self.coordinator.world_size,
                        )
                        if kind != "checkpoint0":
                            raise CheckpointError(
                                f"checkpoint0 destination contains kind={kind!r}"
                            )
                    except (ArtifactError, CheckpointError, OSError) as exc:
                        quarantine = self._quarantine_checkpoint0(
                            destination,
                            validation_error=f"{type(exc).__name__}: {exc}",
                        )
                        decision = {
                            "action": "save",
                            "quarantine": str(quarantine),
                            "error": None,
                        }
                    else:
                        decision = {
                            "action": "reuse",
                            "quarantine": None,
                            "error": None,
                        }
            except BaseException as exc:
                decision = {
                    "action": None,
                    "quarantine": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            decision = None
        shared = self.coordinator.broadcast(decision, source=0)
        if not isinstance(shared, Mapping) or shared.get("error") is not None:
            raise CheckpointError(
                f"checkpoint0 decision failed consistently across ranks: {shared!r}"
            )
        if shared.get("action") not in {"reuse", "save"}:
            raise CheckpointError(f"invalid checkpoint0 decision: {shared!r}")
        return shared

    def save(
        self,
        engine: Any,
        destination: str | Path,
        *,
        kind: CheckpointKind,
        global_step: int,
        epoch: float,
        trainer_state: Any,
        lr_scheduler: Any,
        initial_adapter_dir: str | Path | None = None,
    ) -> ResumeCheckpoint:
        """Atomically save one complete resume checkpoint on all ranks."""

        if kind not in _VALID_KINDS:
            raise CheckpointError(f"invalid checkpoint kind: {kind!r}")
        step, epoch_value = _validate_step_epoch(global_step, epoch, kind=kind)
        self._assert_engine_world(engine)
        engine_step = int(getattr(engine, "global_steps", -1))
        if engine_step != step:
            raise CheckpointError(
                f"Trainer/DeepSpeed step mismatch: trainer={step}, engine={engine_step}"
            )
        state_payload = _trainer_state_dict(trainer_state)
        if state_payload.get("global_step") != step:
            raise CheckpointError(
                f"Trainer state global_step={state_payload.get('global_step')!r}, expected {step}"
            )
        scheduler_payload = _scheduler_state_dict(lr_scheduler, global_step=step)
        if kind == "epoch" and initial_adapter_dir is None:
            raise CheckpointError("durable epoch checkpoints require portable adapter export")
        if kind != "epoch" and initial_adapter_dir is not None:
            raise CheckpointError("portable export is only permitted for durable epoch checkpoints")

        save_signature = inspect.signature(engine.save_checkpoint)
        if "exclude_frozen_parameters" not in save_signature.parameters:
            raise CheckpointError(
                "DeepSpeed engine cannot exclude frozen parameters; refusing to save the 27B base"
            )

        target = Path(destination)
        setup: dict[str, str | None] | None
        if self.coordinator.rank == 0:
            try:
                if target.exists():
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                staging_path = target.parent / f".{target.name}.building.{uuid.uuid4().hex}"
                staging_path.mkdir(parents=False, exist_ok=False)
                setup = {"path": str(staging_path), "error": None}
            except BaseException as exc:
                setup = {"path": None, "error": f"{type(exc).__name__}: {exc}"}
        else:
            setup = None
        setup = self.coordinator.broadcast(setup, source=0)
        if not isinstance(setup, Mapping) or setup.get("error") is not None:
            raise CheckpointError(f"checkpoint staging setup failed: {setup!r}")
        staging_value = setup.get("path")
        if not isinstance(staging_value, str):
            raise CheckpointError(f"checkpoint staging path was not broadcast: {setup!r}")
        staging = Path(staging_value)

        tag = f"global_step{step:09d}"
        rng_state = _capture_rng_state()
        manifest_core: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            **self.identity.as_dict(),
            "global_step": step,
            "epoch": epoch_value,
            "world_size": self.coordinator.world_size,
            "deepspeed_tag": tag,
            "exclude_frozen_parameters": True,
            "frozen_base_restore_policy": (
                "fresh-identical-residual-base-plus-pissa-init-then-nonstrict-lora-overlay"
            ),
            "package_versions": _installed_versions(),
            "scheduler_state_name": SCHEDULER_STATE_NAME,
            "scheduler_class": type(lr_scheduler).__qualname__,
        }
        client_state = {
            "janus_ts_checkpoint": dict(manifest_core),
            "trainer_state": state_payload,
        }

        try:
            saved = engine.save_checkpoint(
                str(staging / DEEPSPEED_SUBDIR),
                tag=tag,
                client_state=client_state,
                save_latest=False,
                exclude_frozen_parameters=True,
            )
            if saved is not True:
                raise CheckpointError(f"DeepSpeed save_checkpoint returned {saved!r}")

            rng_path = staging / RNG_SUBDIR / f"rank-{self.coordinator.rank:05d}.pt"
            _atomic_torch_save(rng_path, rng_state)
            self.coordinator.barrier()

            gathered = gather_lora_state_dict(
                engine,
                self.coordinator,
                expected_trainable_parameters=self.expected_trainable_parameters,
            )
            finalization_error: str | None = None
            if self.coordinator.rank == 0:
                try:
                    assert gathered is not None
                    module = _engine_module(engine)
                    normalized = normalize_lora_state_dict(gathered, adapter_name=self.adapter_name)
                    _write_adapter(
                        staging / RESUME_ADAPTER_SUBDIR,
                        module,
                        normalized,
                        adapter_name=self.adapter_name,
                    )
                    write_json(staging / TRAINER_STATE_NAME, state_payload)
                    _atomic_torch_save(staging / SCHEDULER_STATE_NAME, scheduler_payload)
                    if initial_adapter_dir is not None:
                        manifest_core["pissa_initial_adapter_fingerprint"] = (
                            _write_portable_adapter(
                                staging / PORTABLE_ADAPTER_SUBDIR,
                                module,
                                normalized,
                                initial_adapter_dir=Path(initial_adapter_dir),
                                adapter_name=self.adapter_name,
                            )
                        )
                    _validate_deepspeed_payload(
                        staging, tag=tag, world_size=self.coordinator.world_size
                    )
                    manifest_core["payload_inventory"] = _payload_inventory(staging)
                    mark_complete(staging, manifest_core)
                except BaseException as exc:
                    finalization_error = f"{type(exc).__name__}: {exc}"
            finalization_error = self.coordinator.broadcast(finalization_error, source=0)
            if finalization_error is not None:
                raise CheckpointError(
                    f"rank-zero checkpoint finalization failed: {finalization_error}"
                )

            rename_error: str | None = None
            if self.coordinator.rank == 0:
                try:
                    os.replace(staging, target)
                except BaseException as exc:
                    rename_error = f"{type(exc).__name__}: {exc}"
            rename_error = self.coordinator.broadcast(rename_error, source=0)
            if rename_error is not None:
                raise CheckpointError(f"checkpoint commit failed: {rename_error}")
        except BaseException:
            # Only rank zero owns cleanup.  A failed tree is hidden and has no
            # completion marker; a killed process can leave such a tree for a
            # later explicit stale-build cleanup without being selected.
            if self.coordinator.rank == 0 and staging.exists():
                shutil.rmtree(staging)
            raise

        manifest = read_complete_manifest(target)
        _validate_manifest(manifest, self.identity)
        return ResumeCheckpoint(target, manifest)

    def restore(
        self,
        engine: Any,
        checkpoint: ResumeCheckpoint,
        *,
        lr_scheduler: Any,
    ) -> RestoredTrainingState:
        """Restore a checkpoint into a newly initialized ZeRO-3 engine."""

        self._assert_engine_world(engine)
        if int(getattr(engine, "global_steps", -1)) != 0:
            raise CheckpointError(
                "ZeRO-3 restore requires a freshly initialized engine at global_step=0"
            )
        manifest = read_complete_manifest(checkpoint.path)
        step, _, _ = _validate_checkpoint_payload(
            checkpoint.path,
            manifest,
            self.identity,
            expected_world_size=self.coordinator.world_size,
        )
        tag = manifest["deepspeed_tag"]

        load_path, client_state = engine.load_checkpoint(
            str(checkpoint.path / DEEPSPEED_SUBDIR),
            tag=tag,
            load_module_strict=False,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_only=False,
        )
        if load_path is None or not isinstance(client_state, Mapping):
            raise CheckpointError("DeepSpeed did not load a complete checkpoint")
        saved_core = client_state.get("janus_ts_checkpoint")
        if not isinstance(saved_core, Mapping):
            raise CheckpointError("DeepSpeed client state lacks the Janus-TS manifest")
        for key in (*self.identity.as_dict(), "global_step", "deepspeed_tag"):
            expected = manifest[key]
            if saved_core.get(key) != expected:
                raise CheckpointError(
                    f"DeepSpeed client manifest mismatch for {key}: "
                    f"{saved_core.get(key)!r} != {expected!r}"
                )
        if int(getattr(engine, "global_steps", -1)) != step:
            raise CheckpointError(
                f"DeepSpeed restored global_steps={getattr(engine, 'global_steps', None)!r}, "
                f"expected {step}"
            )

        scheduler_name = manifest.get("scheduler_state_name")
        scheduler_class = manifest.get("scheduler_class")
        if scheduler_name != SCHEDULER_STATE_NAME:
            raise CheckpointError(f"unexpected scheduler payload name: {scheduler_name!r}")
        if scheduler_class != type(lr_scheduler).__qualname__:
            raise CheckpointError(
                "scheduler class differs: "
                f"saved={scheduler_class!r}, current={type(lr_scheduler).__qualname__!r}"
            )
        import torch

        scheduler_path = checkpoint.path / SCHEDULER_STATE_NAME
        expected_scheduler = torch.load(
            scheduler_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(expected_scheduler, Mapping):
            raise CheckpointError("saved scheduler state is not a mapping")
        lr_scheduler.load_state_dict(expected_scheduler)
        actual_scheduler = _scheduler_state_dict(lr_scheduler, global_step=step)
        if not _scheduler_states_equal(expected_scheduler, actual_scheduler):
            raise CheckpointError("restored scheduler state differs from the saved payload")

        gathered = gather_lora_state_dict(
            engine,
            self.coordinator,
            expected_trainable_parameters=self.expected_trainable_parameters,
        )
        comparison_error: str | None = None
        if self.coordinator.rank == 0:
            from safetensors.torch import load_file

            assert gathered is not None
            actual = normalize_lora_state_dict(gathered, adapter_name=self.adapter_name)
            expected_state = load_file(
                checkpoint.path / RESUME_ADAPTER_SUBDIR / "adapter_model.safetensors",
                device="cpu",
            )
            if set(actual) != set(expected_state):
                comparison_error = "restored adapter key set differs from saved adapter"
            else:
                import torch

                unequal = [
                    name for name in actual if not torch.equal(actual[name], expected_state[name])
                ]
                if unequal:
                    comparison_error = f"restored adapter tensors differ: {unequal[:8]!r}"
        comparison_error = self.coordinator.broadcast(comparison_error, source=0)
        if comparison_error is not None:
            raise CheckpointError(comparison_error)

        self.restore_rng(checkpoint)
        trainer_state_path = checkpoint.path / TRAINER_STATE_NAME
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        if trainer_state.get("global_step") != step:
            raise CheckpointError("restored Trainer state step disagrees with manifest")
        return RestoredTrainingState(
            checkpoint=ResumeCheckpoint(checkpoint.path, manifest),
            trainer_state=trainer_state,
            client_state=client_state,
        )

    def restore_rng(self, checkpoint: ResumeCheckpoint) -> None:
        """Replay this rank's saved RNG state at Trainer's resume boundary.

        The complete manifest is revalidated and the small rank-local payload
        is rehashed on every replay. This permits Transformers to restore the
        exact saved state *after* it fast-forwards a mid-epoch dataloader.
        """

        manifest = read_complete_manifest(checkpoint.path)
        _validate_manifest(manifest, self.identity)
        rng_path = (
            checkpoint.path
            / RNG_SUBDIR
            / f"rank-{self.coordinator.rank:05d}.pt"
        )
        relative = rng_path.relative_to(checkpoint.path).as_posix()
        inventory = manifest.get("payload_inventory")
        if not isinstance(inventory, Mapping) or relative not in inventory:
            raise CheckpointError(f"checkpoint manifest lacks RNG payload {relative}")
        entry = inventory[relative]
        if not isinstance(entry, Mapping):
            raise CheckpointError(f"invalid RNG inventory entry: {relative}")
        if not rng_path.is_file() or rng_path.stat().st_size != entry.get("size_bytes"):
            raise CheckpointError(f"checkpoint RNG payload is missing or resized: {rng_path}")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(rng_path) != expected_hash:
            raise CheckpointError(f"checkpoint RNG payload digest differs: {rng_path}")
        _restore_rng_state(rng_path)

    def rotate_local(self, root: str | Path, *, keep: int) -> tuple[Path, ...]:
        """Keep the greatest ``keep`` local steps for this run, by manifest."""

        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise CheckpointError(f"keep must be a positive integer, got {keep!r}")
        if self.coordinator.rank != 0:
            self.coordinator.barrier()
            return ()
        candidates: list[tuple[int, str, Path]] = []
        base = Path(root)
        if base.is_dir():
            for child in base.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                try:
                    manifest = read_complete_manifest(child)
                    step, _, kind = _validate_manifest(manifest, self.identity)
                except (ArtifactError, CheckpointError, OSError):
                    continue
                if kind in {"checkpoint0", "rolling"}:
                    candidates.append((step, child.name, child))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        removed: list[Path] = []
        for _, _, path in candidates[keep:]:
            shutil.rmtree(path)
            removed.append(path)
        self.coordinator.barrier()
        return tuple(removed)


class JanusCheckpointCallback(TrainerCallback):
    """Transformers 5.9-compatible hooks around :class:`CheckpointManager`.

    ``engine_getter`` must return ``trainer.model_wrapped`` after DeepSpeed has
    initialized.  The callback saves checkpoint0 in ``on_train_begin``, local
    rolling checkpoints after completed update steps, and a durable resume +
    portable adapter at each integral epoch end.  It intentionally has no
    mutable callback state; persisted Trainer/DeepSpeed steps are authoritative.
    """

    def __init__(
        self,
        manager: CheckpointManager,
        *,
        engine_getter: Any,
        local_root: str | Path,
        durable_root: str | Path,
        initial_adapter_dir: str | Path,
        checkpoint_steps: int = 50,
        keep_local: int = 2,
    ) -> None:
        super().__init__()
        if checkpoint_steps != 50 or keep_local != 2:
            raise CheckpointError("frozen callback cadence is checkpoint_steps=50, keep_local=2")
        self.manager = manager
        self.engine_getter = engine_getter
        self.local_root = Path(local_root)
        self.durable_root = Path(durable_root)
        self.initial_adapter_dir = Path(initial_adapter_dir)
        self.checkpoint_steps = checkpoint_steps
        self.keep_local = keep_local

    def _engine(self) -> Any:
        engine = self.engine_getter()
        if engine is None:
            raise CheckpointError("DeepSpeed engine is not initialized")
        return engine

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if int(state.global_step) == 0:
            destination = self.manager.destination(self.local_root, 0)
            decision = self.manager._prepare_checkpoint0_destination(destination)
            if decision.get("action") == "save":
                self.manager.save(
                    self._engine(),
                    destination,
                    kind="checkpoint0",
                    global_step=0,
                    epoch=float(state.epoch or 0.0),
                    trainer_state=state,
                    lr_scheduler=kwargs.get("lr_scheduler"),
                )
            self.manager.rotate_local(self.local_root, keep=self.keep_local)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step > 0 and step % self.checkpoint_steps == 0:
            destination = self.manager.destination(self.local_root, step)
            self.manager.save(
                self._engine(),
                destination,
                kind="rolling",
                global_step=step,
                epoch=float(state.epoch or 0.0),
                trainer_state=state,
                lr_scheduler=kwargs.get("lr_scheduler"),
            )
            self.manager.rotate_local(self.local_root, keep=self.keep_local)
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        epoch = float(state.epoch or 0.0)
        if epoch >= 1 and math.isclose(epoch, round(epoch), abs_tol=1e-8):
            destination = self.manager.destination(self.durable_root, int(state.global_step))
            self.manager.save(
                self._engine(),
                destination,
                kind="epoch",
                global_step=int(state.global_step),
                epoch=epoch,
                trainer_state=state,
                lr_scheduler=kwargs.get("lr_scheduler"),
                initial_adapter_dir=self.initial_adapter_dir,
            )
        return control


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointIdentity",
    "CheckpointManager",
    "DistributedCoordinator",
    "JanusCheckpointCallback",
    "RestoredTrainingState",
    "ResumeCheckpoint",
    "TorchDistributedCoordinator",
    "convert_pissa_to_portable_state",
    "gather_lora_state_dict",
    "materialize_train_loss_checkpoint",
    "normalize_lora_state_dict",
    "select_resume_checkpoint",
]
