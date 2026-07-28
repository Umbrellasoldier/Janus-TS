"""Content-addressed gu30 orchestration for the frozen Transition1x run.

This module is deliberately a thin process supervisor, not a second trainer.
Every CUDA-heavy operation runs in an isolated two-rank ``torchrun`` process
under the project GPU lock. Durable
ordering comes only from content identities, explicit global steps, and
completion markers; the gu30 wall clock and filesystem mtimes are never used.
"""

from __future__ import annotations

import errno
import json
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from transformers import TrainerCallback

from .artifacts import (
    ArtifactError,
    atomic_directory,
    mark_complete,
    read_complete_manifest,
    sha256_file,
    sha256_json,
    write_json,
)
from .checkpointing import (
    CheckpointIdentity,
    CheckpointManager,
    JanusCheckpointCallback,
    ResumeCheckpoint,
    select_resume_checkpoint,
)
from .config import ExperimentConfig, load_config
from .constants import MODEL_ID, MODEL_REVISION
from .gates import (
    PISSA_MIN_MEM_AVAILABLE_KIB,
    PISSA_RESOURCE_POLICY,
    pissa_bundle_path,
    prepare_pissa_gate,
)
from .native_stack import audit_native_stack
from .preprocessing import (
    audit_processed_dataset,
    expected_processed_path,
    inventory_sources,
    load_pinned_tokenizer,
)
from .run_state import (
    RunIdentity,
    RunPaths,
    build_run_identity,
    initialize_run,
    resolve_run_paths,
    update_run_state,
)
from .runtime import (
    ResourceUnavailableError,
    assert_host_ready,
    exclusive_lock,
    frozen_distributed_environment,
    install_frozen_environment,
)
from .snapshot import audit_snapshot

TRANSIENT_EXIT_CODE = 75
SMOKE_REJECTED_EXIT_CODE = 42
WORKFLOW_SCHEMA_VERSION = "janus-ts-gu30-workflow-v2"
STAGE_SCHEMA_VERSION = "janus-ts-workflow-stage-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WorkflowError(RuntimeError):
    """A permanent workflow invariant or delegated process failed."""


class TransientWorkflowError(ResourceUnavailableError):
    """A demonstrably transient condition requests supervisor exit 75."""


class DelegatedProcessError(WorkflowError):
    """A delegated command failed without frozen rejection evidence."""

    def __init__(self, phase: str, command: Sequence[str], returncode: int) -> None:
        self.phase = phase
        self.command = tuple(command)
        self.returncode = int(returncode)
        super().__init__(f"{phase} failed permanently with exit {returncode}: {' '.join(command)}")


class _DelegatedSignalInterrupt(BaseException):
    """Interrupt the parent so it can reap only its delegated process group."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"delegated command interrupted by signal {signum}")


@dataclass(frozen=True, slots=True)
class CpuAuditResult:
    config: ExperimentConfig
    config_path: Path
    processed_path: Path
    data_fingerprint: str
    launch_fingerprint: str
    report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedRun:
    config: ExperimentConfig
    config_path: Path
    processed_path: Path
    bundle_dir: Path
    run_identity: RunIdentity
    checkpoint_identity: CheckpointIdentity
    paths: RunPaths
    gate_root: Path
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    phase: str
    command: tuple[str, ...]
    returncode: int


@dataclass(frozen=True, slots=True)
class EpochCheckpoint:
    epoch: int
    global_step: int
    path: Path
    checkpoint_fingerprint: str


Runner = Callable[..., subprocess.CompletedProcess[Any]]


def _run_owned_process_group(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float | None,
) -> subprocess.CompletedProcess[Any]:
    """Run one delegated tree in its own process group.

    A hard-gate timeout terminates only this newly-created process group.  The
    group ID is validated against the child PID before sending a signal, so
    this path can never target an unrelated user's job.
    """

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        start_new_session=True,
    )
    process_group = os.getpgid(process.pid)
    if process_group != process.pid:
        process.kill()
        process.wait()
        raise WorkflowError("delegated process failed to acquire its own process group")

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        return True

    def wait_group_clear(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while group_exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        return not group_exists()

    def terminate_owned_group() -> None:
        if not group_exists():
            return
        os.killpg(process_group, signal.SIGTERM)
        if process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
        if wait_group_clear(30):
            return
        os.killpg(process_group, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=30)
        if not wait_group_clear(30):
            raise WorkflowError("delegated process group survived SIGKILL")

    previous_handlers: dict[int, Any] = {}

    def interrupt_handler(signum: int, frame: Any) -> None:
        del frame
        raise _DelegatedSignalInterrupt(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt_handler)
    try:
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            # This covers timeout, KeyboardInterrupt, SIGINT/SIGTERM converted
            # above, and unexpected wait failures.  The target is the child PID
            # already proven to be this launcher's process-group leader.
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            terminate_owned_group()
            raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    # torchrun is the process-group leader.  If it returned while a worker is
    # still alive, clean only that owned group before interpreting sidecars or
    # allowing a candidate fallback.
    if group_exists():
        terminate_owned_group()
    return subprocess.CompletedProcess(list(command), returncode)


def project_executable(name: str) -> Path:
    """Resolve a required executable from this project's uv environment."""

    executable = (PROJECT_ROOT / ".venv" / "bin" / name).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise WorkflowError(f"required project executable is unavailable: {executable}")
    return executable


def torchrun_command(module: str, *arguments: str) -> tuple[str, ...]:
    """Build the exact two-rank command using the absolute uv torchrun."""

    return (
        str(project_executable("torchrun")),
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        module,
        *map(str, arguments),
    )


def _delegated_training_command(arguments: Sequence[str]) -> tuple[str, ...]:
    """Launch the hidden Typer worker with the absolute console script."""

    return (
        str(project_executable("torchrun")),
        "--standalone",
        "--nproc-per-node=2",
        str(project_executable("janus-ts")),
        "train",
        "worker",
        *map(str, arguments),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"JSON evidence is not an object: {path}")
    return value


def _stage_manifest(
    stage: str,
    fingerprint: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": stage,
        "workflow_fingerprint": fingerprint,
        "status": "complete",
        "report": dict(report),
    }


def seal_stage(
    destination: str | Path,
    *,
    stage: str,
    fingerprint: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically seal immutable stage evidence, or verify an exact restart."""

    target = Path(destination)
    expected = _stage_manifest(stage, fingerprint, report)
    if target.exists():
        stored = read_complete_manifest(target)
        for key, value in expected.items():
            if stored.get(key) != value:
                raise WorkflowError(f"completed stage evidence drift at {target}: {key}")
        report_path = target / "report.json"
        if stored.get("report_sha256") != sha256_file(report_path):
            raise WorkflowError(f"stage report hash mismatch at {target}")
        return stored
    try:
        with atomic_directory(target) as building:
            write_json(building / "report.json", dict(report))
            expected_with_hash = {
                **expected,
                "report_sha256": sha256_file(building / "report.json"),
            }
            mark_complete(building, expected_with_hash)
    except FileExistsError:
        # A concurrent identical launcher may have committed first.  This is
        # safe only when the newly visible artifact validates exactly.
        pass
    stored = read_complete_manifest(target)
    expected_hash = sha256_json(dict(report))
    # ``write_json`` is pretty-printed, so compare the manifest payload first
    # and independently verify the durable file hash it records.
    for key, value in expected.items():
        if stored.get(key) != value:
            raise WorkflowError(f"completed stage evidence drift at {target}: {key}")
    report_path = target / "report.json"
    if stored.get("report_sha256") != sha256_file(report_path):
        raise WorkflowError(f"stage report hash mismatch at {target}")
    if sha256_json(_read_json(report_path)) != expected_hash:
        raise WorkflowError(f"stage report content mismatch at {target}")
    return stored


def load_stage(destination: str | Path, *, stage: str, fingerprint: str) -> dict[str, Any] | None:
    """Load a completed stage by explicit identity; incomplete trees are ignored."""

    target = Path(destination)
    if not target.exists():
        return None
    try:
        manifest = read_complete_manifest(target)
    except ArtifactError:
        return None
    expected = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage": stage,
        "workflow_fingerprint": fingerprint,
        "status": "complete",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise WorkflowError(f"foreign completed stage at {target}")
    report_path = target / "report.json"
    if manifest.get("report_sha256") != sha256_file(report_path):
        raise WorkflowError(f"stage report hash mismatch at {target}")
    report = manifest.get("report")
    if not isinstance(report, dict) or report != _read_json(report_path):
        raise WorkflowError(f"stage report payload mismatch at {target}")
    return report


def _absolute_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise WorkflowError(f"configuration is unavailable: {path}: {exc}") from exc


def run_cpu_audits(config_path: str | Path) -> CpuAuditResult:
    """Run every read-only/data hard gate before requesting a GPU."""

    install_frozen_environment()
    path = _absolute_config_path(config_path)
    config = load_config(path)
    sources = inventory_sources(config)
    processed = expected_processed_path(config, sources)
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed
    processed = processed.resolve(strict=True)
    tokenizer = load_pinned_tokenizer(config, local_files_only=True)
    data_audit = dict(audit_processed_dataset(processed, config, tokenizer=tokenizer, write=False))
    # The underlying audit optionally records when it was run for human-facing
    # audit.json files.  An immutable workflow stage must contain only content
    # evidence, never the unreliable gu30 wall clock.
    data_audit.pop("audited_at_utc", None)
    data_fingerprint = data_audit.get("fingerprint")
    if not isinstance(data_fingerprint, str) or len(data_fingerprint) != 64:
        raise WorkflowError("processed data audit did not return a SHA256 fingerprint")
    snapshot = audit_snapshot(cache_dir=config.model.cache_dir)
    native = audit_native_stack()
    report = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "pass",
        "config_sha256": config.sha256,
        "processed_path": str(processed),
        "data_audit": data_audit,
        "snapshot": snapshot,
        "native_stack": native,
        "host_policy": (
            "GPU sharing is allowed; host memory and whole-device peak GPU memory "
            "are checked independently"
        ),
    }
    launch_fingerprint = sha256_json(
        {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "config_sha256": config.sha256,
            "data_fingerprint": data_fingerprint,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "snapshot_critical_sha256": snapshot["critical_sha256"],
            "native_stack": native,
        }
    )
    return CpuAuditResult(
        config=config,
        config_path=path,
        processed_path=processed,
        data_fingerprint=data_fingerprint,
        launch_fingerprint=launch_fingerprint,
        report=report,
    )


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(frozen_distributed_environment())
    return environment


def run_delegated_command(
    phase: str,
    command: Sequence[str],
    *,
    runner: Runner = subprocess.run,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> CommandOutcome:
    """Run one absolute command and classify only explicit exit 75 as transient."""

    values = tuple(map(str, command))
    if not values or not Path(values[0]).is_absolute():
        raise WorkflowError(f"{phase} command executable must be absolute: {values!r}")
    kwargs: dict[str, Any] = {
        "cwd": PROJECT_ROOT,
        "env": dict(env or _subprocess_environment()),
        "check": False,
    }
    if timeout_seconds is not None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise WorkflowError(f"invalid {phase} timeout {timeout_seconds!r}")
        kwargs["timeout"] = timeout_seconds
    try:
        if runner is subprocess.run:
            result = _run_owned_process_group(
                values,
                cwd=PROJECT_ROOT,
                env=kwargs["env"],
                timeout=timeout_seconds,
            )
        else:
            result = runner(list(values), **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(
            f"{phase} exceeded its explicit {timeout_seconds:g}s hard-gate timeout"
        ) from exc
    returncode = int(result.returncode)
    if returncode == TRANSIENT_EXIT_CODE:
        raise TransientWorkflowError(f"{phase} reported transient exit 75")
    return CommandOutcome(phase, values, returncode)


def run_locked_gpu_command(
    phase: str,
    command: Sequence[str],
    *,
    lock_path: str | Path,
    required_gpu_count: int = 2,
    runner: Runner = subprocess.run,
    timeout_seconds: float | None = None,
    require_clean_after: bool = True,
) -> CommandOutcome:
    """Take the independent GPU flock, recheck resources, then delegate."""

    with exclusive_lock(lock_path):
        assert_host_ready(
            required_gpu_count=required_gpu_count,
        )
        try:
            return run_delegated_command(
                phase,
                command,
                runner=runner,
                timeout_seconds=timeout_seconds,
            )
        finally:
            if require_clean_after:
                # The workflow process itself may retain a small CUDA context
                # from one-time PiSSA preparation. No delegated child may
                # remain on either GPU after torchrun has returned or timed out.
                assert_host_ready(
                    required_gpu_count=required_gpu_count,
                )


def _require_success(outcome: CommandOutcome) -> None:
    if outcome.returncode != 0:
        raise DelegatedProcessError(outcome.phase, outcome.command, outcome.returncode)


def _stage_root(cpu: CpuAuditResult) -> Path:
    path = (
        cpu.config.runtime.artifacts_root / "gates" / cpu.config.data.name / cpu.launch_fingerprint
    )
    return path if path.is_absolute() else PROJECT_ROOT / path


def _absolute_run_paths(paths: RunPaths) -> RunPaths:
    project = paths.project if paths.project.is_absolute() else PROJECT_ROOT / paths.project
    local = paths.local if paths.local.is_absolute() else PROJECT_ROOT / paths.local
    return RunPaths(project=project.resolve(), local=local.resolve())


def _run_kernel_gate(
    cpu: CpuAuditResult,
    gate_root: Path,
    *,
    runner: Runner,
) -> dict[str, Any]:
    stage = gate_root / "01-kernel"
    if report := load_stage(stage, stage="kernel", fingerprint=cpu.launch_fingerprint):
        return report
    raw_report = gate_root / "raw" / "kernel.json"
    environment = _subprocess_environment()
    environment["JANUS_TS_GATE_OUTPUT"] = str(raw_report.resolve())
    with exclusive_lock(cpu.config.runtime.gpu_lock_path):
        assert_host_ready(
            required_gpu_count=cpu.config.runtime.required_gpu_count,
        )
        outcome = run_delegated_command(
            "kernel-gate",
            torchrun_command("janus_ts.gpu_gates"),
            runner=runner,
            env=environment,
            timeout_seconds=15 * 60,
        )
        assert_host_ready(
            required_gpu_count=cpu.config.runtime.required_gpu_count,
        )
    _require_success(outcome)
    report = _read_json(raw_report)
    if report.get("status") != "pass":
        raise WorkflowError("kernel gate did not record status=pass")
    seal_stage(stage, stage="kernel", fingerprint=cpu.launch_fingerprint, report=report)
    return report


def _run_dtype_gate(
    cpu: CpuAuditResult,
    gate_root: Path,
    *,
    runner: Runner,
) -> dict[str, Any]:
    stage = gate_root / "02-dtype"
    if report := load_stage(stage, stage="dtype", fingerprint=cpu.launch_fingerprint):
        return report
    raw_report = (gate_root / "raw" / "dtype.json").resolve()
    command = torchrun_command("janus_ts.dtype_gate", "--output", str(raw_report))
    outcome = run_locked_gpu_command(
        "dtype-gate",
        command,
        lock_path=cpu.config.runtime.gpu_lock_path,
        required_gpu_count=cpu.config.runtime.required_gpu_count,
        runner=runner,
        timeout_seconds=10 * 60,
    )
    _require_success(outcome)
    report = _read_json(raw_report)
    if report.get("status") != "pass":
        raise WorkflowError("dtype gate did not record status=pass")
    seal_stage(stage, stage="dtype", fingerprint=cpu.launch_fingerprint, report=report)
    return report


def _pissa_resource_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    resources = report.get("preparation_resources")
    if not isinstance(resources, Mapping):
        raise WorkflowError("PiSSA gate did not return preparation resource evidence")
    if (
        resources.get("policy") != PISSA_RESOURCE_POLICY
        or resources.get("scope") not in {"full_serialization", "cached_payload_verification"}
        or resources.get("minimum_mem_available_kib") != PISSA_MIN_MEM_AVAILABLE_KIB
        or resources.get("swap_growth_is_diagnostic") is not True
        or resources.get("other_gpu_phases_max_swap_growth_kib") != 256 * 1024
        or type(resources.get("observed_swap_growth_kib")) is not int
        or not isinstance(resources.get("host_before"), Mapping)
        or not isinstance(resources.get("host_after"), Mapping)
    ):
        raise WorkflowError("PiSSA preparation resource evidence violates the frozen policy")
    # HostStatus.to_dict() intentionally preserves its tuple fields. Durable
    # JSON turns those tuples into arrays, so normalize before immutable stage
    # comparison instead of comparing Python container implementation details.
    normalized = json.loads(json.dumps(dict(resources), allow_nan=False))
    if not isinstance(normalized, dict):  # pragma: no cover - guaranteed above
        raise WorkflowError("PiSSA preparation resource evidence is not a JSON object")
    return normalized


def _prepare_pissa(cpu: CpuAuditResult, gate_root: Path) -> tuple[Path, dict[str, Any]]:
    stage = gate_root / "03-pissa"
    bundle = pissa_bundle_path(cpu.config.runtime.local_cache_root)
    prior = load_stage(stage, stage="pissa", fingerprint=cpu.launch_fingerprint)
    if prior is not None:
        _pissa_resource_evidence(prior)
        manifest_sha256 = sha256_file(bundle / "manifest.json")
        files_sha256 = prior.get("files_sha256")
        if not isinstance(files_sha256, str) or len(files_sha256) != 64:
            raise WorkflowError("sealed PiSSA stage has no aggregate payload hash")
        expected = {
            "bundle": str(bundle),
            "manifest_sha256": manifest_sha256,
        }
        drift = {
            key: {"sealed": prior.get(key), "current": value}
            for key, value in expected.items()
            if prior.get(key) != value
        }
        if drift:
            raise WorkflowError(f"cached PiSSA stage drift: {drift}")
        return bundle, prior

    # prepare_pissa_gate owns the same non-reentrant lock internally.  Do not
    # wrap this call in another flock.
    report = prepare_pissa_gate(
        hub_cache_dir=cpu.config.model.cache_dir,
        local_cache_root=cpu.config.runtime.local_cache_root,
        gpu_lock_path=cpu.config.runtime.gpu_lock_path,
    )
    resources = _pissa_resource_evidence(report)
    manifest_sha256 = sha256_file(bundle / "manifest.json")
    files_sha256 = report.get("files_sha256")
    if not isinstance(files_sha256, str) or len(files_sha256) != 64:
        raise WorkflowError("PiSSA gate did not return the aggregate payload hash")
    evidence = {
        "status": "pass",
        "bundle": str(bundle),
        "manifest_sha256": manifest_sha256,
        "files_sha256": files_sha256,
        "preparation_resources": resources,
        "observed_swap_growth_kib": resources["observed_swap_growth_kib"],
    }
    seal_stage(stage, stage="pissa", fingerprint=cpu.launch_fingerprint, report=evidence)
    return bundle, evidence


def derive_checkpoint_identity(
    run_identity: RunIdentity,
    *,
    pissa_manifest_sha256: str,
) -> CheckpointIdentity:
    """Use one run fingerprint across state, checkpoint, generation and eval."""

    model_fingerprint = sha256_json(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "pissa_manifest_sha256": pissa_manifest_sha256,
        }
    )
    return CheckpointIdentity(
        run_fingerprint=run_identity.fingerprint,
        config_fingerprint=run_identity.config_sha256,
        data_fingerprint=run_identity.data_fingerprint,
        model_fingerprint=model_fingerprint,
    )


def _write_checkpoint_identity(paths: RunPaths, identity: CheckpointIdentity) -> Path:
    destination = paths.project / "checkpoint-identity.json"
    payload = identity.as_dict()
    if destination.exists():
        if _read_json(destination) != payload:
            raise WorkflowError(f"checkpoint identity drift at {destination}")
    else:
        write_json(destination, payload)
    return destination


_RUN_STAGE_ORDER = {
    "initialized": 0,
    "training": 1,
    "validation": 2,
    "selected": 3,
    "testing": 4,
    "complete": 5,
}


def _advance_run_state(
    prepared: PreparedRun,
    *,
    stage: str,
    global_step: int,
    **updates: Any,
) -> dict[str, Any]:
    """Advance monotonically, while making post-crash re-entry idempotent."""

    current = _read_json(prepared.paths.project / "state.json")
    if current.get("run_fingerprint") != prepared.run_identity.fingerprint:
        raise WorkflowError("workflow state belongs to a foreign run")
    current_stage = current.get("stage")
    if current_stage == "failed":
        raise WorkflowError("a terminal failed run cannot be resumed silently")
    if current_stage not in _RUN_STAGE_ORDER or stage not in _RUN_STAGE_ORDER:
        raise WorkflowError(f"unknown workflow state transition {current_stage!r}->{stage!r}")
    if _RUN_STAGE_ORDER[current_stage] > _RUN_STAGE_ORDER[stage]:
        if int(current.get("global_step", -1)) != global_step:
            raise WorkflowError("later workflow state has a different global step")
        for key, value in updates.items():
            if key in current and current[key] != value:
                raise WorkflowError(f"later workflow state disagrees on {key}")
        return current
    return update_run_state(
        prepared.paths,
        prepared.run_identity,
        stage=stage,
        global_step=global_step,
        **updates,
    )


def _smoke_report_status(report_path: Path) -> tuple[str | None, dict[str, Any] | None]:
    if report_path.is_file():
        report = _read_json(report_path)
        status = report.get("status")
        return (status if isinstance(status, str) else None), report
    return None, None


def _smoke_gate_binding(prepared: PreparedRun, *, micro_batch_size: int) -> dict[str, str]:
    deepspeed_config = prepared.config.train.deepspeed_config
    if not deepspeed_config.is_absolute():
        deepspeed_config = PROJECT_ROOT / deepspeed_config
    bundle_hash = sha256_file(prepared.bundle_dir / "manifest.json")
    identity = sha256_json(
        {
            "gate": "zero3-worst-case-2048-one-update",
            "config_fingerprint": prepared.config.sha256,
            "bundle_manifest_sha256": bundle_hash,
            "deepspeed_config_sha256": sha256_file(deepspeed_config),
            "micro_batch_size_per_gpu": micro_batch_size,
            "sequence_length": prepared.config.model.max_sequence_length,
            "world_size": prepared.config.runtime.required_gpu_count,
        }
    )
    return {
        "gate_identity": identity,
        "config_fingerprint": prepared.config.sha256,
        "bundle_manifest_sha256": bundle_hash,
    }


def _is_bound_smoke_report(
    payload: Mapping[str, Any],
    binding: Mapping[str, str],
) -> bool:
    return all(payload.get(key) == value for key, value in binding.items())


def _matching_rank_oom_reports(
    report_path: Path,
    *,
    expected_micro_batch_size: int,
    binding: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Find only rank evidence explicitly tied to this frozen 2048 gate."""

    patterns = (
        f"{report_path.name}.rank-*.oom.json",
        f"{report_path.stem}.rank-*.oom.json",
        f"{report_path.stem}.rank-*.json",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(report_path.parent.glob(pattern))
    accepted: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            payload = _read_json(path)
        except WorkflowError:
            continue
        gate = payload.get("gate")
        status = payload.get("status")
        sequence = payload.get("sequence_length", payload.get("max_sequence_length"))
        microbatch = payload.get("micro_batch_size_per_gpu", payload.get("micro_batch_size"))
        if (
            gate == "zero3-worst-case-2048-one-update"
            and status in {"oom", "rejected"}
            and (sequence in {None, 2048})
            and microbatch == expected_micro_batch_size
            and _is_bound_smoke_report(payload, binding)
        ):
            accepted.append(payload)
    return tuple(accepted)


def _validate_smoke_pass(
    report: Mapping[str, Any],
    prepared: PreparedRun,
    *,
    micro_batch_size: int,
) -> None:
    config = prepared.config
    binding = _smoke_gate_binding(prepared, micro_batch_size=micro_batch_size)
    if report.get("status") != "pass" or report.get("gate") != ("zero3-worst-case-2048-one-update"):
        raise WorkflowError("memory smoke report is not a passing frozen gate")
    if not _is_bound_smoke_report(report, binding):
        raise WorkflowError("memory smoke report is bound to different inputs")
    if report.get("world_size") != config.runtime.required_gpu_count:
        raise WorkflowError("memory smoke report has the wrong world size")
    ranks = report.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != config.runtime.required_gpu_count:
        raise WorkflowError("memory smoke report does not contain exactly two ranks")
    observed_ranks = {item.get("rank") for item in ranks if isinstance(item, Mapping)}
    if observed_ranks != set(range(config.runtime.required_gpu_count)):
        raise WorkflowError("memory smoke rank set is not exactly {0,1}")
    expected_accumulation = (
        config.train.gradient_accumulation_steps
        if micro_batch_size == config.train.micro_batch_size_per_gpu
        else config.train.candidate_gradient_accumulation_steps
    )
    for item in ranks:
        if not isinstance(item, Mapping):
            raise WorkflowError("memory smoke rank report is malformed")
        if not _is_bound_smoke_report(item, binding):
            raise WorkflowError("memory smoke rank report is bound to different inputs")
        if item.get("micro_batch_size_per_gpu") != micro_batch_size:
            raise WorkflowError("memory smoke rank report has the wrong microbatch")
        if item.get("gradient_accumulation_steps") != expected_accumulation:
            raise WorkflowError("memory smoke rank report has the wrong accumulation")
        if item.get("sequence_length") != config.model.max_sequence_length:
            raise WorkflowError("memory smoke rank report has the wrong sequence length")
        if (
            item.get("local_rank") != item.get("rank")
            or item.get("global_step") != 1
            or isinstance(item.get("training_loss"), bool)
            or not isinstance(item.get("training_loss"), (int, float))
            or not math.isfinite(float(item["training_loss"]))
        ):
            raise WorkflowError("memory smoke rank did not prove one finite update")
        integer_fields = (
            "peak_allocated_mib",
            "peak_reserved_mib",
            "device_memory_used_mib",
            "swap_growth_kib",
        )
        if any(
            isinstance(item.get(name), bool) or not isinstance(item.get(name), int)
            for name in integer_fields
        ):
            raise WorkflowError("memory smoke rank resource evidence is incomplete")
        if any(int(item[name]) < 0 for name in integer_fields[:3]):
            raise WorkflowError("memory smoke rank contains negative memory evidence")
        peak = max(
            int(item["peak_allocated_mib"]),
            int(item["peak_reserved_mib"]),
            int(item["device_memory_used_mib"]),
        )
        if peak > config.runtime.max_gpu_peak_mib:
            raise WorkflowError(
                f"memory smoke peak {peak}MiB exceeds {config.runtime.max_gpu_peak_mib}MiB"
            )
        if int(item["swap_growth_kib"]) > 256 * 1024:
            raise WorkflowError("memory smoke swap growth exceeds 256MiB")


def _run_memory_smoke(
    prepared: PreparedRun,
    *,
    micro_batch_size: Literal[1, 2],
    runner: Runner,
) -> tuple[bool, dict[str, Any]]:
    label = "04-smoke-micro1" if micro_batch_size == 1 else "05-smoke-candidate2"
    stage_name = "smoke-micro1" if micro_batch_size == 1 else "smoke-candidate2"
    stage = prepared.gate_root / label
    if report := load_stage(
        stage,
        stage=stage_name,
        fingerprint=prepared.run_identity.fingerprint,
    ):
        return report.get("decision") == "selected", report
    raw_report = (prepared.gate_root / "raw" / f"memory-micro{micro_batch_size}.json").resolve()
    binding = _smoke_gate_binding(prepared, micro_batch_size=micro_batch_size)
    prior_status, prior_raw = _smoke_report_status(raw_report)
    prior_rank_rejections = _matching_rank_oom_reports(
        raw_report,
        expected_micro_batch_size=micro_batch_size,
        binding=binding,
    )
    if prior_status == "pass" and prior_raw is not None:
        _validate_smoke_pass(
            prior_raw,
            prepared,
            micro_batch_size=micro_batch_size,
        )
        recovered = {**prior_raw, "decision": "selected"}
        seal_stage(
            stage,
            stage=stage_name,
            fingerprint=prepared.run_identity.fingerprint,
            report=recovered,
        )
        return True, recovered
    prior_microbatch = (
        prior_raw.get("micro_batch_size_per_gpu", prior_raw.get("micro_batch_size"))
        if prior_raw is not None
        else None
    )
    prior_rejected = (
        prior_status in {"oom", "rejected"}
        and prior_raw is not None
        and prior_raw.get("gate") == "zero3-worst-case-2048-one-update"
        and prior_microbatch == micro_batch_size
        and _is_bound_smoke_report(prior_raw, binding)
    ) or bool(prior_rank_rejections)
    if micro_batch_size == 2 and prior_rejected:
        recovered = {
            "gate": "zero3-worst-case-2048-one-update",
            "status": "rejected",
            "decision": "fallback-to-micro1",
            "micro_batch_size_per_gpu": 2,
            "delegated_returncode": None,
            "recovered_after_interruption": True,
            "report": prior_raw,
            "rank_rejections": list(prior_rank_rejections),
        }
        seal_stage(
            stage,
            stage=stage_name,
            fingerprint=prepared.run_identity.fingerprint,
            report=recovered,
        )
        return False, recovered
    if prior_raw is not None or prior_rank_rejections:
        raise WorkflowError(f"stale or invalid memory-smoke evidence at {raw_report}")
    command = torchrun_command(
        "janus_ts.distributed",
        "memory-smoke",
        "--config",
        str(prepared.config_path),
        "--bundle",
        str(prepared.bundle_dir),
        "--output-dir",
        str((prepared.gate_root / "smoke-trainer").resolve()),
        "--micro-batch-size",
        str(micro_batch_size),
        "--report",
        str(raw_report),
    )
    outcome = run_locked_gpu_command(
        f"memory-smoke-micro{micro_batch_size}",
        command,
        lock_path=prepared.config.runtime.gpu_lock_path,
        required_gpu_count=prepared.config.runtime.required_gpu_count,
        runner=runner,
        timeout_seconds=2 * 60 * 60,
    )
    status, raw = _smoke_report_status(raw_report)
    rank_rejections = _matching_rank_oom_reports(
        raw_report,
        expected_micro_batch_size=micro_batch_size,
        binding=binding,
    )
    raw_microbatch = (
        raw.get("micro_batch_size_per_gpu", raw.get("micro_batch_size"))
        if raw is not None
        else None
    )
    rejected = (
        status in {"oom", "rejected"}
        and raw is not None
        and raw.get("gate") == "zero3-worst-case-2048-one-update"
        and raw_microbatch == micro_batch_size
        and _is_bound_smoke_report(raw, binding)
    ) or bool(rank_rejections)
    if outcome.returncode == 0:
        if raw is None:
            raise WorkflowError("successful memory smoke produced no report")
        _validate_smoke_pass(
            raw,
            prepared,
            micro_batch_size=micro_batch_size,
        )
        evidence = {**raw, "decision": "selected"}
        seal_stage(
            stage,
            stage=stage_name,
            fingerprint=prepared.run_identity.fingerprint,
            report=evidence,
        )
        return True, evidence
    # Exit 42 is only a transport code.  It may select the default geometry
    # only when a report/sidecar is cryptographically bound to this exact
    # config, bundle, sequence length and candidate microbatch.
    if micro_batch_size == 2 and rejected:
        evidence = {
            "gate": "zero3-worst-case-2048-one-update",
            "status": "rejected",
            "decision": "fallback-to-micro1",
            "micro_batch_size_per_gpu": 2,
            "delegated_returncode": outcome.returncode,
            "report": raw,
            "rank_rejections": list(rank_rejections),
        }
        seal_stage(
            stage,
            stage=stage_name,
            fingerprint=prepared.run_identity.fingerprint,
            report=evidence,
        )
        return False, evidence
    raise DelegatedProcessError(outcome.phase, outcome.command, outcome.returncode)


def _run_resume_continuity_gate(
    prepared: PreparedRun,
    *,
    runner: Runner,
) -> dict[str, Any]:
    """Exercise real save/restart/next-update ordering before formal training."""

    def validate(report: Mapping[str, Any]) -> None:
        from .continuity_gates import (
            CONTINUITY_GATE_SCHEMA_VERSION,
            RESUME_GATE_NAME,
            make_resume_gate_identity,
        )

        bundle_sha = sha256_file(prepared.bundle_dir / "manifest.json")
        source_fingerprint = sha256_json(
            {
                "expected_source_sha256": prepared.config.data.expected_source_sha256,
                "expected_raw_counts": prepared.config.data.expected_raw_counts,
                "expected_retained_counts": prepared.config.data.expected_retained_counts,
            }
        )
        expected_identity = make_resume_gate_identity(
            config_sha256=prepared.config.sha256,
            source_fingerprint=source_fingerprint,
            bundle_manifest_sha256=bundle_sha,
        ).as_dict()
        exact = {
            "schema_version": CONTINUITY_GATE_SCHEMA_VERSION,
            "gate": RESUME_GATE_NAME,
            "status": "pass",
            "world_size": prepared.config.runtime.required_gpu_count,
            "identity": expected_identity,
            "bundle_manifest_sha256": bundle_sha,
            "micro_batch_size_per_gpu": prepared.micro_batch_size_per_gpu,
            "sequence_length": prepared.config.model.max_sequence_length,
        }
        if any(report.get(key) != value for key, value in exact.items()):
            raise WorkflowError("resume continuity report is bound to different inputs")
        checkpoint = report.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or any(
            checkpoint.get(key) != value
            for key, value in {
                "kind": "rolling",
                "global_step": 1,
                "complete_marker_verified": True,
                "exclude_frozen_parameters": True,
            }.items()
        ):
            raise WorkflowError("resume continuity did not prove a complete step-1 checkpoint")
        manifest_sha = checkpoint.get("manifest_sha256")
        if (
            not isinstance(manifest_sha, str)
            or len(manifest_sha) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha)
        ):
            raise WorkflowError("resume continuity checkpoint digest is invalid")
        comparison = report.get("comparison")
        if not isinstance(comparison, Mapping) or any(
            comparison.get(key) != value
            for key, value in {
                "status": "pass",
                "global_step": 2,
                "optimizer_state_restored_before_next_update": True,
                "scheduler_state_equal": True,
                "rank_local_rng_equal": True,
                "lora_bitwise_equal": True,
            }.items()
        ):
            raise WorkflowError("resume continuity next-update proof is incomplete")
        stream = comparison.get("lora_stream")
        if (
            not isinstance(stream, Mapping)
            or stream.get("logical_parameters")
            != prepared.config.adapter.expected_trainable_parameters
            or stream.get("tensor_count") != prepared.config.adapter.expected_target_modules * 2
        ):
            raise WorkflowError("resume continuity LoRA stream has the wrong coverage")
        ranks = report.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 2:
            raise WorkflowError("resume continuity lacks two rank reports")
        observed: set[int] = set()
        for item in ranks:
            if not isinstance(item, Mapping):
                raise WorkflowError("resume continuity rank report is malformed")
            rank = item.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise WorkflowError("resume continuity rank identity is invalid")
            observed.add(rank)
            continuous = item.get("continuous")
            resumed = item.get("resumed")
            rank_comparison = item.get("comparison")
            if not all(
                isinstance(value, Mapping) for value in (continuous, resumed, rank_comparison)
            ):
                raise WorkflowError("resume continuity rank branches are missing")
            if any(
                branch.get("rank") != rank
                or branch.get("global_step") != 2
                or branch.get("engine_global_step") != 2
                for branch in (continuous, resumed)
            ):
                raise WorkflowError("resume continuity rank did not complete equal step 2")
            restore = resumed.get("restore")
            if not isinstance(restore, Mapping) or any(
                restore.get(key) != value
                for key, value in {
                    "checkpoint_global_step": 1,
                    "trainer_state_global_step": 1,
                    "optimizer_and_scheduler_loaded_by_deepspeed": True,
                    "rng_replayed_at_resume_boundary": True,
                }.items()
            ):
                raise WorkflowError("resume continuity rank lacks pre-update restore evidence")
            if rank_comparison.get("status") != "pass":
                raise WorkflowError("resume continuity rank comparison failed")
        if observed != {0, 1}:
            raise WorkflowError("resume continuity rank set is not {0,1}")

    stage = prepared.gate_root / "07-resume-continuity"
    if report := load_stage(
        stage,
        stage="resume-continuity",
        fingerprint=prepared.run_identity.fingerprint,
    ):
        validate(report)
        return report
    raw_report = (prepared.gate_root / "raw" / "resume-continuity.json").resolve()
    command = torchrun_command(
        "janus_ts.continuity_gates",
        "resume",
        "--config",
        str(prepared.config_path),
        "--bundle",
        str(prepared.bundle_dir),
        "--output",
        str(raw_report),
        "--micro-batch-size",
        str(prepared.micro_batch_size_per_gpu),
    )
    outcome = run_locked_gpu_command(
        "resume-continuity-gate",
        command,
        lock_path=prepared.config.runtime.gpu_lock_path,
        required_gpu_count=prepared.config.runtime.required_gpu_count,
        runner=runner,
        timeout_seconds=2 * 60 * 60,
    )
    _require_success(outcome)
    report = _read_json(raw_report)
    validate(report)
    seal_stage(
        stage,
        stage="resume-continuity",
        fingerprint=prepared.run_identity.fingerprint,
        report=report,
    )
    return report


def prepare_smoke_workflow(
    config_path: str | Path = "configs/transition1x.yaml",
    *,
    runner: Runner = subprocess.run,
) -> PreparedRun:
    """Verify data, native BF16, and PiSSA before direct default training."""

    cpu = run_cpu_audits(config_path)
    gate_root = _stage_root(cpu)
    gate_root.mkdir(parents=True, exist_ok=True)
    cpu_stage = gate_root / "00-cpu"
    if load_stage(cpu_stage, stage="cpu", fingerprint=cpu.launch_fingerprint) is None:
        seal_stage(
            cpu_stage,
            stage="cpu",
            fingerprint=cpu.launch_fingerprint,
            report=cpu.report,
        )
    _run_kernel_gate(cpu, gate_root, runner=runner)
    _run_dtype_gate(cpu, gate_root, runner=runner)
    bundle, _ = _prepare_pissa(cpu, gate_root)

    microbatch = cpu.config.train.micro_batch_size_per_gpu
    accumulation = cpu.config.train.gradient_accumulation_steps
    run_identity = build_run_identity(
        cpu.config,
        processed_data_dir=cpu.processed_path,
        pissa_bundle_dir=bundle,
        micro_batch_size_per_gpu=microbatch,
        gradient_accumulation_steps=accumulation,
    )
    paths = _absolute_run_paths(resolve_run_paths(cpu.config, run_identity))
    initialize_run(paths, run_identity)
    checkpoint_identity = derive_checkpoint_identity(
        run_identity,
        pissa_manifest_sha256=run_identity.pissa_manifest_sha256,
    )
    _write_checkpoint_identity(paths, checkpoint_identity)
    selected = PreparedRun(
        config=cpu.config,
        config_path=cpu.config_path,
        processed_path=cpu.processed_path,
        bundle_dir=bundle,
        run_identity=run_identity,
        checkpoint_identity=checkpoint_identity,
        paths=paths,
        gate_root=gate_root,
        micro_batch_size_per_gpu=microbatch,
        gradient_accumulation_steps=accumulation,
    )
    return selected


def make_checkpoint_callback_factory(
    identity: CheckpointIdentity,
    *,
    local_root: str | Path,
    durable_root: str | Path,
    initial_adapter_dir: str | Path,
    checkpoint_steps: int,
    keep_local: int,
    expected_trainable_parameters: int,
    manager_holder: dict[str, CheckpointManager] | None = None,
) -> Callable[[Callable[[], Any]], JanusCheckpointCallback]:
    """Construct the late-bound manager/callback after torch.distributed init."""

    holder = manager_holder if manager_holder is not None else {}

    def factory(engine_getter: Callable[[], Any]) -> JanusCheckpointCallback:
        if "manager" in holder:
            raise WorkflowError("checkpoint callback factory was invoked more than once")
        manager = CheckpointManager(
            identity,
            expected_trainable_parameters=expected_trainable_parameters,
        )
        holder["manager"] = manager
        return JanusCheckpointCallback(
            manager,
            engine_getter=engine_getter,
            local_root=local_root,
            durable_root=durable_root,
            initial_adapter_dir=initial_adapter_dir,
            checkpoint_steps=checkpoint_steps,
            keep_local=keep_local,
        )

    return factory


class RankZeroJsonlLogCallback(TrainerCallback):
    """Fsync one JSON object per Trainer log event on global rank zero.

    The callback intentionally does not coalesce duplicate global steps.  A
    restart can legitimately log the resumed step again; ``launch_ordinal``
    and ``resume_checkpoint`` make those records unambiguous without consulting
    wall-clock time.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_fingerprint: str,
        resume_checkpoint: str | Path | None,
        log_steps: int,
    ) -> None:
        if log_steps != 10:
            raise WorkflowError(f"JSONL logging cadence must be 10, got {log_steps}")
        self.path = Path(path)
        self.run_fingerprint = run_fingerprint
        self.resume_checkpoint = (
            str(Path(resume_checkpoint).resolve()) if resume_checkpoint is not None else None
        )
        self._rank = int(os.environ.get("RANK", "0"))
        self.launch_ordinal = self._next_launch_ordinal() if self._rank == 0 else -1

    def _next_launch_ordinal(self) -> int:
        if not self.path.exists():
            return 1
        if self.path.is_symlink() or not self.path.is_file():
            raise WorkflowError(f"training JSONL path is not a regular file: {self.path}")
        maximum = 0
        try:
            with self.path.open("r+b") as handle:
                payload = handle.read()
                if payload and not payload.endswith(b"\n"):
                    # A process can die between append writes.  Only the final
                    # unterminated suffix is recoverable: complete records
                    # always include their newline before the fsync.  Any
                    # malformed newline-terminated record remains fatal.
                    last_newline = payload.rfind(b"\n")
                    retained_size = last_newline + 1
                    handle.seek(retained_size)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    payload = payload[:retained_size]
                text_payload = payload.decode("utf-8")
                for line_number, line in enumerate(text_payload.splitlines(), start=1):
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise WorkflowError(f"training JSONL line {line_number} is not an object")
                    if payload.get("run_fingerprint") != self.run_fingerprint:
                        raise WorkflowError("training JSONL contains a foreign run fingerprint")
                    ordinal = payload.get("launch_ordinal")
                    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
                        raise WorkflowError("training JSONL has an invalid launch ordinal")
                    maximum = max(maximum, ordinal)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"training JSONL is malformed: {exc}") from exc
        return maximum + 1

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._rank != 0 or not bool(getattr(state, "is_world_process_zero", True)):
            return control
        payload = {
            "schema_version": "janus-ts-trainer-log-v1",
            "run_fingerprint": self.run_fingerprint,
            "launch_ordinal": self.launch_ordinal,
            "resume_checkpoint": self.resume_checkpoint,
            "global_step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
            "logs": dict(logs or {}),
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise WorkflowError("short write while appending training JSONL")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return control


def make_jsonl_logging_callback_factory(
    path: str | Path,
    *,
    run_fingerprint: str,
    resume_checkpoint: str | Path | None,
    log_steps: int,
) -> Callable[[Callable[[], Any]], RankZeroJsonlLogCallback]:
    """Return a late-bound factory with the same surface as training hooks."""

    def factory(engine_getter: Callable[[], Any]) -> RankZeroJsonlLogCallback:
        del engine_getter
        return RankZeroJsonlLogCallback(
            path,
            run_fingerprint=run_fingerprint,
            resume_checkpoint=resume_checkpoint,
            log_steps=log_steps,
        )

    return factory


def _load_worker_identity(
    config: ExperimentConfig,
    processed_path: Path,
    bundle_dir: Path,
    identity_path: Path,
    *,
    micro_batch_size_per_gpu: int,
    gradient_accumulation_steps: int,
) -> tuple[RunIdentity, CheckpointIdentity, RunPaths]:
    run_identity = build_run_identity(
        config,
        processed_data_dir=processed_path,
        pissa_bundle_dir=bundle_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    checkpoint_identity = derive_checkpoint_identity(
        run_identity,
        pissa_manifest_sha256=run_identity.pissa_manifest_sha256,
    )
    if _read_json(identity_path) != checkpoint_identity.as_dict():
        raise WorkflowError("distributed worker checkpoint identity differs from content inputs")
    return (
        run_identity,
        checkpoint_identity,
        _absolute_run_paths(resolve_run_paths(config, run_identity)),
    )


def run_distributed_training_worker(
    *,
    config_path: str | Path,
    processed_path: str | Path,
    bundle_dir: str | Path,
    identity_path: str | Path,
    output_dir: str | Path,
    micro_batch_size_per_gpu: int,
    gradient_accumulation_steps: int,
    resume_from_checkpoint: str | Path | None = None,
) -> None:
    """Two-rank worker: build Trainer, attach atomic persistence, then train."""

    from .distributed import (
        build_full_distributed_run,
        checkpoint_manager_restore_hook,
    )

    install_frozen_environment()
    config = load_config(config_path)
    processed = Path(processed_path).resolve(strict=True)
    bundle = Path(bundle_dir).resolve(strict=True)
    _, identity, paths = _load_worker_identity(
        config,
        processed,
        bundle,
        Path(identity_path).resolve(strict=True),
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    holder: dict[str, CheckpointManager] = {}
    factory = make_checkpoint_callback_factory(
        identity,
        local_root=paths.local_checkpoints,
        durable_root=paths.durable_checkpoints,
        initial_adapter_dir=bundle / "pissa_init",
        checkpoint_steps=config.train.checkpoint_steps,
        keep_local=config.train.keep_local_checkpoints,
        expected_trainable_parameters=config.adapter.expected_trainable_parameters,
        manager_holder=holder,
    )
    logging_factory = make_jsonl_logging_callback_factory(
        paths.project / "logs" / "train.jsonl",
        run_fingerprint=identity.run_fingerprint,
        resume_checkpoint=resume_from_checkpoint,
        log_steps=config.train.log_steps,
    )
    run = build_full_distributed_run(
        config,
        bundle_dir=bundle,
        processed_path=processed,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        callback_factories=(factory, logging_factory),
        janus_managed_checkpoints=True,
    )
    manager = holder.get("manager")
    if manager is None:
        raise WorkflowError("checkpoint callback factory was not attached")
    resume: ResumeCheckpoint | None = None
    if resume_from_checkpoint is not None:
        selected = select_resume_checkpoint(
            (paths.local_checkpoints, paths.durable_checkpoints),
            identity=identity,
        )
        requested = Path(resume_from_checkpoint).resolve(strict=True)
        if selected is None or selected.path.resolve() != requested:
            raise WorkflowError("worker resume path is not the greatest valid manifest step")
        resume = selected
        run.trainer.janus_restore_hook = checkpoint_manager_restore_hook(manager, resume)
    run.trainer.train(resume_from_checkpoint=str(resume.path) if resume is not None else None)


def _training_command(prepared: PreparedRun, resume: ResumeCheckpoint | None) -> tuple[str, ...]:
    identity_path = prepared.paths.project / "checkpoint-identity.json"
    arguments = [
        "--config",
        str(prepared.config_path),
        "--processed-path",
        str(prepared.processed_path),
        "--bundle-dir",
        str(prepared.bundle_dir),
        "--identity-path",
        str(identity_path),
        "--output-dir",
        str((prepared.paths.project / "trainer").resolve()),
        "--micro-batch-size",
        str(prepared.micro_batch_size_per_gpu),
        "--gradient-accumulation-steps",
        str(prepared.gradient_accumulation_steps),
    ]
    if resume is not None:
        arguments.extend(("--resume-from-checkpoint", str(resume.path.resolve())))
    return _delegated_training_command(arguments)


def enumerate_epoch_checkpoints(prepared: PreparedRun) -> tuple[EpochCheckpoint, ...]:
    """Resolve exactly epochs 1..5 from complete manifests, never mtimes."""

    values: dict[int, EpochCheckpoint] = {}
    root = prepared.paths.durable_checkpoints
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                manifest = read_complete_manifest(child)
            except (ArtifactError, OSError):
                continue
            identity_fields = prepared.checkpoint_identity.as_dict()
            if any(manifest.get(key) != value for key, value in identity_fields.items()):
                continue
            if manifest.get("kind") != "epoch":
                continue
            raw_epoch = manifest.get("epoch")
            if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, (int, float)):
                continue
            epoch = int(raw_epoch)
            if not math.isclose(float(raw_epoch), epoch, abs_tol=1e-8) or epoch <= 0:
                continue
            step = manifest.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                continue
            portable = child / "portable_adapter"
            if not portable.is_dir() or not (portable / "adapter_model.safetensors").is_file():
                continue
            candidate = EpochCheckpoint(
                epoch=epoch,
                global_step=step,
                path=child,
                checkpoint_fingerprint=sha256_file(child / "manifest.json"),
            )
            previous = values.get(epoch)
            if previous is not None and previous.path != child:
                raise WorkflowError(f"multiple durable checkpoints claim epoch {epoch}")
            values[epoch] = candidate
    expected = set(range(1, prepared.config.train.epochs + 1))
    if set(values) != expected:
        raise WorkflowError(f"durable epoch set={sorted(values)}, expected={sorted(expected)}")
    ordered = tuple(values[epoch] for epoch in sorted(values))
    if any(
        left.global_step >= right.global_step
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise WorkflowError("durable epoch global_step values are not strictly increasing")
    return ordered


def run_training_phase(
    prepared: PreparedRun,
    *,
    runner: Runner = subprocess.run,
) -> tuple[EpochCheckpoint, ...]:
    """Resume the greatest valid local/durable step and finish all five epochs."""

    def finalize(checkpoints: tuple[EpochCheckpoint, ...]) -> tuple[EpochCheckpoint, ...]:
        """Reconstruct the immutable training handoff from durable epochs alone."""

        last = checkpoints[-1]
        _advance_run_state(
            prepared,
            stage="validation",
            global_step=last.global_step,
        )
        evidence = {
            "status": "pass",
            "epochs": [
                {
                    "epoch": item.epoch,
                    "global_step": item.global_step,
                    "path": str(item.path),
                    "checkpoint_fingerprint": item.checkpoint_fingerprint,
                }
                for item in checkpoints
            ],
        }
        stage = prepared.paths.project / "workflow" / "training"
        stored = load_stage(
            stage,
            stage="training",
            fingerprint=prepared.run_identity.fingerprint,
        )
        if stored is None:
            seal_stage(
                stage,
                stage="training",
                fingerprint=prepared.run_identity.fingerprint,
                report=evidence,
            )
        elif stored != evidence:
            raise WorkflowError("completed training evidence differs from durable epochs")
        return checkpoints

    try:
        existing = enumerate_epoch_checkpoints(prepared)
    except WorkflowError as exc:
        if not str(exc).startswith("durable epoch set="):
            raise
        existing = ()
    if len(existing) == prepared.config.train.epochs:
        # The process may have died after the fifth atomic epoch checkpoint but
        # before state/provenance were sealed.  Finish that handoff locally;
        # never relaunch a completed five-epoch job just to repair metadata.
        return finalize(existing)
    resume = select_resume_checkpoint(
        (prepared.paths.local_checkpoints, prepared.paths.durable_checkpoints),
        identity=prepared.checkpoint_identity,
    )
    current_step = resume.global_step if resume is not None else 0
    _advance_run_state(
        prepared,
        stage="training",
        global_step=current_step,
    )
    outcome = run_locked_gpu_command(
        "five-epoch-training",
        _training_command(prepared, resume),
        lock_path=prepared.config.runtime.gpu_lock_path,
        required_gpu_count=prepared.config.runtime.required_gpu_count,
        runner=runner,
    )
    _require_success(outcome)
    checkpoints = enumerate_epoch_checkpoints(prepared)
    return finalize(checkpoints)


def _formal_eval_command(
    prepared: PreparedRun,
    checkpoint: EpochCheckpoint,
    *,
    split: Literal["val", "test"],
    selection_proof: Path | None = None,
) -> tuple[str, ...]:
    arguments = [
        split,
        "--config",
        str(prepared.config_path),
        "--processed-path",
        str(prepared.processed_path),
        "--checkpoint-dir",
        str(checkpoint.path.resolve()),
        "--output-dir",
        str(prepared.paths.evaluations.resolve()),
    ]
    if selection_proof is not None:
        arguments.extend(("--selection-proof", str(selection_proof.resolve())))
    return torchrun_command("janus_ts.formal_eval_runtime", *arguments)


def _zero_shot_eval_command(prepared: PreparedRun) -> tuple[str, ...]:
    return torchrun_command(
        "janus_ts.zero_shot_runtime",
        "--config",
        str(prepared.config_path),
        "--processed-path",
        str(prepared.processed_path),
        "--output-dir",
        str(prepared.paths.evaluations.resolve()),
        "--run-fingerprint",
        prepared.run_identity.fingerprint,
    )


def _validate_generation_smoke(value: Any, config: ExperimentConfig) -> None:
    """Validate formal parsing plus the forced-full-512 stress receipt."""

    if not isinstance(value, Mapping):
        raise WorkflowError("epoch-1 portable parity lacks generation evidence")
    from .continuity_gates import (
        ContinuityGateError,
        validate_generation_smoke_evidence,
    )

    try:
        validate_generation_smoke_evidence(
            value,
            max_device_memory_mib=config.runtime.max_gpu_peak_mib,
            max_swap_growth_kib=256 * 1024,
        )
    except ContinuityGateError as exc:
        raise WorkflowError(f"invalid epoch-1 generation smoke: {exc}") from exc


def _run_portable_parity_gate(
    prepared: PreparedRun,
    checkpoint: EpochCheckpoint,
    *,
    runner: Runner,
) -> dict[str, Any]:
    """Verify this epoch's rank-64 portable adapter before it is evaluated."""

    def validate(report: Mapping[str, Any]) -> None:
        from .continuity_gates import (
            CONTINUITY_GATE_SCHEMA_VERSION,
            PORTABLE_GATE_NAME,
        )

        exact = {
            "schema_version": CONTINUITY_GATE_SCHEMA_VERSION,
            "gate": PORTABLE_GATE_NAME,
            "status": "pass",
            "world_size": prepared.config.runtime.required_gpu_count,
        }
        if any(report.get(key) != value for key, value in exact.items()):
            raise WorkflowError("portable parity report has the wrong protocol identity")
        checkpoint_report = report.get("checkpoint")
        expected_checkpoint = {
            "path": str(checkpoint.path.resolve()),
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "run_fingerprint": prepared.run_identity.fingerprint,
            "config_fingerprint": prepared.config.sha256,
            "data_fingerprint": prepared.run_identity.data_fingerprint,
            "model_fingerprint": prepared.checkpoint_identity.model_fingerprint,
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "bundle_manifest_sha256": sha256_file(prepared.bundle_dir / "manifest.json"),
        }
        if not isinstance(checkpoint_report, Mapping) or any(
            checkpoint_report.get(key) != value for key, value in expected_checkpoint.items()
        ):
            raise WorkflowError("portable parity report is bound to another checkpoint")
        residual = report.get("residual_rank32")
        portable = report.get("original_portable_rank64")
        if (
            not isinstance(residual, Mapping)
            or residual.get("adapter_parameters")
            != prepared.config.adapter.expected_trainable_parameters
            or not isinstance(portable, Mapping)
            or portable.get("adapter_parameters")
            != prepared.config.adapter.expected_portable_parameters
        ):
            raise WorkflowError("portable parity adapter coverage is invalid")
        parity = report.get("logit_parity")
        if not isinstance(parity, Mapping) or any(
            parity.get(key) != value
            for key, value in {"status": "pass", "atol": 0.125, "rtol": 0.02}.items()
        ):
            raise WorkflowError("portable parity numeric proof is invalid")
        probe = report.get("probe")
        if (
            not isinstance(probe, Mapping)
            or probe.get("selection") != "first-longest-legal-canonical-validation-prompt"
            or isinstance(probe.get("prompt_tokens"), bool)
            or not isinstance(probe.get("prompt_tokens"), int)
            or not 0 < probe["prompt_tokens"] <= prepared.config.model.max_sequence_length
        ):
            raise WorkflowError("portable parity prompt evidence is invalid")
        ranks = report.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 2:
            raise WorkflowError("portable parity lacks two rank reports")
        observed = {item.get("rank") for item in ranks if isinstance(item, Mapping)}
        if observed != {0, 1}:
            raise WorkflowError("portable parity rank set is not {0,1}")
        generation_smoke = report.get("generation_smoke")
        if checkpoint.epoch == 1:
            _validate_generation_smoke(generation_smoke, prepared.config)
        elif generation_smoke != {
            "status": "not-run",
            "reason": "frozen-full-length-generation-memory-gate-runs-on-epoch-1-only",
            "epoch": checkpoint.epoch,
        }:
            raise WorkflowError("non-epoch-1 checkpoint reran or forged the memory gate")

    stage = (
        prepared.paths.project
        / "workflow"
        / f"portable-parity-epoch-{checkpoint.epoch}-{checkpoint.checkpoint_fingerprint}"
    )
    if report := load_stage(
        stage,
        stage="portable-parity",
        fingerprint=checkpoint.checkpoint_fingerprint,
    ):
        validate(report)
        return report
    raw_report = (
        prepared.paths.evaluations / f"portable-parity.{checkpoint.checkpoint_fingerprint}.json"
    ).resolve()
    command = torchrun_command(
        "janus_ts.continuity_gates",
        "portable-parity",
        "--config",
        str(prepared.config_path),
        "--bundle",
        str(prepared.bundle_dir),
        "--checkpoint",
        str(checkpoint.path.resolve()),
        "--processed-path",
        str(prepared.processed_path),
        "--output",
        str(raw_report),
    )
    outcome = run_locked_gpu_command(
        f"portable-parity-epoch-{checkpoint.epoch}",
        command,
        lock_path=prepared.config.runtime.gpu_lock_path,
        required_gpu_count=prepared.config.runtime.required_gpu_count,
        runner=runner,
        timeout_seconds=2 * 60 * 60,
    )
    _require_success(outcome)
    report = _read_json(raw_report)
    validate(report)
    seal_stage(
        stage,
        stage="portable-parity",
        fingerprint=checkpoint.checkpoint_fingerprint,
        report=report,
    )
    return report


def _generation_identity(prepared: PreparedRun, checkpoint: EpochCheckpoint, split: str) -> Any:
    from .generation import GenerationIdentity

    return GenerationIdentity(
        split=split,
        data_fingerprint=prepared.run_identity.data_fingerprint,
        run_fingerprint=prepared.run_identity.fingerprint,
        checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
    )


def _validate_formal_receipt(
    prepared: PreparedRun,
    checkpoint: EpochCheckpoint,
    *,
    split: Literal["val", "test"],
) -> tuple[Path, Path]:
    from .formal_eval_runtime import runtime_receipt_path, validate_runtime_receipt
    from .generation import merged_predictions_path, metrics_path

    identity = _generation_identity(prepared, checkpoint, split)
    metrics = metrics_path(prepared.paths.evaluations, identity)
    predictions = merged_predictions_path(prepared.paths.evaluations, identity)
    receipt = runtime_receipt_path(prepared.paths.evaluations, identity)
    validate_runtime_receipt(
        receipt,
        checkpoint_dir=checkpoint.path,
        processed_path=prepared.processed_path,
        config=prepared.config,
    )
    if not metrics.is_file() or not predictions.is_file():
        raise WorkflowError(f"formal {split} receipt lacks metrics or predictions")
    return metrics, predictions


def _run_one_formal_evaluation(
    prepared: PreparedRun,
    checkpoint: EpochCheckpoint,
    *,
    split: Literal["val", "test"],
    selection_proof: Path | None,
    runner: Runner,
) -> tuple[Path, Path]:
    _run_portable_parity_gate(prepared, checkpoint, runner=runner)
    stage = (
        prepared.paths.project
        / "workflow"
        / f"{split}-epoch-{checkpoint.epoch}-{checkpoint.checkpoint_fingerprint}"
    )
    if (
        load_stage(
            stage,
            stage=f"formal-{split}",
            fingerprint=checkpoint.checkpoint_fingerprint,
        )
        is not None
    ):
        return _validate_formal_receipt(prepared, checkpoint, split=split)

    # A worker can be interrupted after its content-addressed result/receipt
    # is durable but before this small workflow stage is sealed.  Validate (or,
    # for a locked test completion, reconstruct) that receipt locally before
    # waiting for GPUs or launching torchrun again.
    from .formal_eval_runtime import (
        inspect_durable_checkpoint,
        recover_completed_test,
        runtime_receipt_path,
    )

    identity = _generation_identity(prepared, checkpoint, split)
    receipt = runtime_receipt_path(prepared.paths.evaluations, identity)
    recovered = receipt.is_file()
    if not recovered and split == "test":
        from .generation import load_selection_proof

        if selection_proof is None:
            raise WorkflowError("formal test recovery requires a selection proof")
        durable = inspect_durable_checkpoint(
            checkpoint.path,
            prepared.processed_path,
            config=prepared.config,
        )
        load_selection_proof(
            selection_proof,
            durable.generation_identity("test"),
        )
        recovered = recover_completed_test(prepared.paths.evaluations, durable) is not None
    if recovered:
        metrics, predictions = _validate_formal_receipt(prepared, checkpoint, split=split)
        report = {
            "status": "pass",
            "split": split,
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "metrics": str(metrics),
            "metrics_sha256": sha256_file(metrics),
            "predictions": str(predictions),
            "predictions_sha256": sha256_file(predictions),
        }
        seal_stage(
            stage,
            stage=f"formal-{split}",
            fingerprint=checkpoint.checkpoint_fingerprint,
            report=report,
        )
        return metrics, predictions
    outcome = run_locked_gpu_command(
        f"formal-{split}-epoch-{checkpoint.epoch}",
        _formal_eval_command(
            prepared,
            checkpoint,
            split=split,
            selection_proof=selection_proof,
        ),
        lock_path=prepared.config.runtime.gpu_lock_path,
        required_gpu_count=prepared.config.runtime.required_gpu_count,
        runner=runner,
    )
    _require_success(outcome)
    metrics, predictions = _validate_formal_receipt(prepared, checkpoint, split=split)
    report = {
        "status": "pass",
        "split": split,
        "epoch": checkpoint.epoch,
        "global_step": checkpoint.global_step,
        "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
        "metrics": str(metrics),
        "metrics_sha256": sha256_file(metrics),
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
    }
    seal_stage(
        stage,
        stage=f"formal-{split}",
        fingerprint=checkpoint.checkpoint_fingerprint,
        report=report,
    )
    return metrics, predictions


def _write_or_validate_selection_proof(
    prepared: PreparedRun,
    selected: EpochCheckpoint,
) -> Path:
    from .generation import SelectionProof, write_selection_proof

    proof = SelectionProof(
        data_fingerprint=prepared.run_identity.data_fingerprint,
        run_fingerprint=prepared.run_identity.fingerprint,
        selected_checkpoint_fingerprint=selected.checkpoint_fingerprint,
        selected_epoch=selected.epoch,
    )
    path = prepared.paths.evaluations / "selection-proof.json"
    if path.exists():
        if _read_json(path) != proof.to_json_dict():
            raise WorkflowError("locked selection proof differs from recomputed winner")
        return path
    return write_selection_proof(path, proof)


def _validate_zero_shot_receipt(prepared: PreparedRun) -> tuple[Any, Path, Path]:
    from .generation import merged_predictions_path, metrics_path
    from .zero_shot_runtime import (
        build_zero_shot_baseline,
        validate_zero_shot_completion,
    )

    baseline = build_zero_shot_baseline(
        prepared.config,
        prepared.processed_path,
        run_fingerprint=prepared.run_identity.fingerprint,
    )
    receipt = validate_zero_shot_completion(prepared.paths.evaluations, baseline)
    if receipt is None:
        raise WorkflowError("zero-shot test evaluation has no complete receipt")
    identity = baseline.generation_identity("test")
    metrics = metrics_path(prepared.paths.evaluations, identity)
    predictions = merged_predictions_path(prepared.paths.evaluations, identity)
    if not metrics.is_file() or not predictions.is_file():
        raise WorkflowError("zero-shot receipt lacks metrics or predictions")
    return baseline, metrics, predictions


def _run_zero_shot_baseline(
    prepared: PreparedRun,
    *,
    runner: Runner,
) -> tuple[Any, Path, Path]:
    from .zero_shot_runtime import build_zero_shot_baseline

    baseline = build_zero_shot_baseline(
        prepared.config,
        prepared.processed_path,
        run_fingerprint=prepared.run_identity.fingerprint,
    )
    stage = prepared.paths.project / "workflow" / f"zero-shot-{baseline.model_fingerprint}"
    stored = load_stage(
        stage,
        stage="formal-zero-shot-test",
        fingerprint=baseline.model_fingerprint,
    )
    if stored is not None:
        return _validate_zero_shot_receipt(prepared)

    try:
        baseline, metrics, predictions = _validate_zero_shot_receipt(prepared)
    except WorkflowError as exc:
        if "no complete receipt" not in str(exc):
            raise
        outcome = run_locked_gpu_command(
            "formal-test-zero-shot",
            _zero_shot_eval_command(prepared),
            lock_path=prepared.config.runtime.gpu_lock_path,
            required_gpu_count=prepared.config.runtime.required_gpu_count,
            runner=runner,
        )
        _require_success(outcome)
        baseline, metrics, predictions = _validate_zero_shot_receipt(prepared)

    report = {
        "status": "pass",
        "split": "test",
        "model": "Qwen/Qwen3.6-27B zero-shot",
        "model_fingerprint": baseline.model_fingerprint,
        "metrics": str(metrics),
        "metrics_sha256": sha256_file(metrics),
        "predictions": str(predictions),
        "predictions_sha256": sha256_file(predictions),
    }
    seal_stage(
        stage,
        stage="formal-zero-shot-test",
        fingerprint=baseline.model_fingerprint,
        report=report,
    )
    return baseline, metrics, predictions


def _validated_test_report(path: Path) -> Mapping[str, Any]:
    from .constants import FORMAL_EVAL_K
    from .generation import EXPECTED_FORMAL_SPLIT_COUNTS

    payload = _read_json(path)
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise WorkflowError(f"test metrics lacks evaluation report: {path}")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise WorkflowError(f"test evaluation lacks metrics: {path}")
    expected_labels = {f"@{value}" for value in FORMAL_EVAL_K}
    if set(metrics) != expected_labels:
        raise WorkflowError(f"test evaluation has wrong k values: {sorted(metrics)!r}")
    expected_count = EXPECTED_FORMAL_SPLIT_COUNTS["test"]
    for label in expected_labels:
        summary = metrics[label]
        if not isinstance(summary, Mapping) or summary.get("count") != expected_count:
            raise WorkflowError(f"test {label} count is not {expected_count}")
    return evaluation


def _write_zero_shot_comparison(
    prepared: PreparedRun,
    selected: EpochCheckpoint,
    *,
    selected_metrics: Path,
    selected_predictions: Path,
    baseline: Any,
    baseline_metrics: Path,
    baseline_predictions: Path,
) -> Path:
    output = prepared.paths.evaluations / "zero-shot-comparison.json"
    payload = {
        "schema_version": "janus-ts-zero-shot-comparison-v1",
        "data_fingerprint": prepared.run_identity.data_fingerprint,
        "run_fingerprint": prepared.run_identity.fingerprint,
        "test_count": 996,
        "fine_tuned_qwen": {
            "selected_epoch": selected.epoch,
            "checkpoint_fingerprint": selected.checkpoint_fingerprint,
            "metrics_sha256": sha256_file(selected_metrics),
            "predictions_sha256": sha256_file(selected_predictions),
            "evaluation": _validated_test_report(selected_metrics),
        },
        "qwen_zero_shot": {
            "model_fingerprint": baseline.model_fingerprint,
            "protocol": dict(baseline.protocol),
            "metrics_sha256": sha256_file(baseline_metrics),
            "predictions_sha256": sha256_file(baseline_predictions),
            "evaluation": _validated_test_report(baseline_metrics),
        },
    }
    if output.exists():
        if _read_json(output) != payload:
            raise WorkflowError("zero-shot comparison artifact drift")
    else:
        write_json(output, payload)
    return output


def run_evaluation_phase(
    prepared: PreparedRun,
    checkpoints: Sequence[EpochCheckpoint] | None = None,
    *,
    runner: Runner = subprocess.run,
) -> EpochCheckpoint:
    """Validate five epochs, test the winner, then compare raw-Qwen zero-shot."""

    from .evaluation import select_best_checkpoint
    from .formal_eval_runtime import load_checkpoint_score

    epochs = tuple(checkpoints or enumerate_epoch_checkpoints(prepared))
    if len(epochs) != prepared.config.train.epochs:
        raise WorkflowError("formal selection requires all five epoch checkpoints")
    scores = []
    for checkpoint in epochs:
        metrics, _ = _run_one_formal_evaluation(
            prepared,
            checkpoint,
            split="val",
            selection_proof=None,
            runner=runner,
        )
        scores.append(
            load_checkpoint_score(
                checkpoint.path,
                metrics,
                processed_path=prepared.processed_path,
                config=prepared.config,
            )
        )
    winner_score = select_best_checkpoint(scores)
    matches = [
        checkpoint
        for checkpoint in epochs
        if checkpoint.checkpoint_fingerprint == winner_score.checkpoint_id
        and checkpoint.epoch == winner_score.epoch
    ]
    if len(matches) != 1:
        raise WorkflowError("formal score winner does not map to exactly one epoch checkpoint")
    selected = matches[0]
    proof = _write_or_validate_selection_proof(prepared, selected)
    _advance_run_state(
        prepared,
        stage="selected",
        global_step=epochs[-1].global_step,
        selected_epoch=selected.epoch,
        selected_checkpoint_fingerprint=selected.checkpoint_fingerprint,
    )
    _advance_run_state(
        prepared,
        stage="testing",
        global_step=epochs[-1].global_step,
    )
    selected_metrics, selected_predictions = _run_one_formal_evaluation(
        prepared,
        selected,
        split="test",
        selection_proof=proof,
        runner=runner,
    )
    baseline, baseline_metrics, baseline_predictions = _run_zero_shot_baseline(
        prepared,
        runner=runner,
    )
    comparison = _write_zero_shot_comparison(
        prepared,
        selected,
        selected_metrics=selected_metrics,
        selected_predictions=selected_predictions,
        baseline=baseline,
        baseline_metrics=baseline_metrics,
        baseline_predictions=baseline_predictions,
    )
    _advance_run_state(
        prepared,
        stage="complete",
        global_step=epochs[-1].global_step,
        selected_epoch=selected.epoch,
        selected_checkpoint_fingerprint=selected.checkpoint_fingerprint,
        test_evaluated=True,
        zero_shot_baseline_evaluated=True,
        zero_shot_comparison=str(comparison),
    )
    return selected


def run_full_workflow(
    config_path: str | Path = "configs/transition1x.yaml",
    *,
    runner: Runner = subprocess.run,
) -> PreparedRun:
    """Prepare gates, resume five-epoch training, and finish formal evaluation."""

    prepared = prepare_smoke_workflow(config_path, runner=runner)
    checkpoints = run_training_phase(prepared, runner=runner)
    run_evaluation_phase(prepared, checkpoints, runner=runner)
    return prepared


def is_proven_transient_exception(error: BaseException) -> bool:
    """Recognize only explicit resource errors and narrow transient OS errnos."""

    transient_errnos = {
        errno.ESTALE,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.ECONNRESET,
        errno.ECONNABORTED,
    }
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ResourceUnavailableError, TransientWorkflowError)):
            return True
        if isinstance(current, OSError) and current.errno in transient_errnos:
            return True
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None
    return False


__all__ = [
    "CommandOutcome",
    "CpuAuditResult",
    "DelegatedProcessError",
    "EpochCheckpoint",
    "PreparedRun",
    "PROJECT_ROOT",
    "SMOKE_REJECTED_EXIT_CODE",
    "TRANSIENT_EXIT_CODE",
    "TransientWorkflowError",
    "WorkflowError",
    "derive_checkpoint_identity",
    "enumerate_epoch_checkpoints",
    "is_proven_transient_exception",
    "load_stage",
    "make_checkpoint_callback_factory",
    "prepare_smoke_workflow",
    "project_executable",
    "run_cpu_audits",
    "run_delegated_command",
    "run_distributed_training_worker",
    "run_evaluation_phase",
    "run_full_workflow",
    "run_locked_gpu_command",
    "run_training_phase",
    "seal_stage",
    "torchrun_command",
]
