"""Reboot-safe, non-destructive supervision for the pinned gu30 run.

The supervisor deliberately knows nothing about checkpoint discovery.  It
executes the supplied absolute CLI command verbatim; that command owns resume
selection and must use complete manifests/global steps rather than mtimes.
Its process-singleton lock is intentionally distinct from the delegated
CLI's gu30 GPU-phase lock, so a child can acquire the latter without a
parent/child self-deadlock.

Only exit status :data:`TRANSIENT_EXIT_CODE` requests a retry.  Every other
non-zero status is permanent, which prevents an OOM or protocol error from
silently changing the experiment or entering an unbounded restart loop.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_json, write_json
from .runtime import (
    ResourceUnavailableError,
    RuntimeContractError,
    assert_host_ready,
    descendant_pids,
    install_frozen_environment,
)

DEFAULT_PROJECT_ROOT = Path("/mnt/sto3/caoxiangyu/Janus-TS")
DEFAULT_VENV_DIR = DEFAULT_PROJECT_ROOT / ".venv"
DEFAULT_LOG_DIR = DEFAULT_PROJECT_ROOT / "artifacts" / "logs" / "supervisor"
GPU_PHASE_LOCK_PATH = Path(
    "/home/caoxiangyu/.cache/janus-ts/locks/gu30-gpu0-1.lock"
)
DEFAULT_LOCK_PATH = Path(
    "/home/caoxiangyu/.cache/janus-ts/locks/janus-ts-transition1x-supervisor.lock"
)
DEFAULT_TAG = "janus-ts-transition1x"
DEFAULT_RETRY_DELAYS_SECONDS = (60, 300, 900)
TRANSIENT_EXIT_CODE = 75
PERMANENT_EXIT_CODE = 64
STATE_VERSION = 1
CRONTAB_MARKER_PREFIX = "# JANUS_TS_SUPERVISOR:"
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VALID_STATES = {
    "initialized",
    "waiting_resources",
    "waiting_retry",
    "running",
    "interrupted",
    "complete",
    "permanent_failure",
    "retry_exhausted",
}


class SupervisorContractError(RuntimeError):
    """The persistent supervisor contract is invalid or has drifted."""


class PreviousTerminalStateError(SupervisorContractError):
    """A prior permanent/exhausted run requires explicit operator action."""


@dataclass(frozen=True, slots=True)
class SupervisorSpec:
    """Immutable launch identity and filesystem locations."""

    tag: str
    project_root: Path
    venv_dir: Path
    lock_path: Path
    log_dir: Path
    command: tuple[str, ...]
    retry_delays_seconds: tuple[int, ...] = DEFAULT_RETRY_DELAYS_SECONDS
    poll_seconds: int = 60

    def __post_init__(self) -> None:
        if not _TAG_PATTERN.fullmatch(self.tag):
            raise SupervisorContractError(f"invalid supervisor tag: {self.tag!r}")
        for name in ("project_root", "venv_dir", "lock_path", "log_dir"):
            path = getattr(self, name)
            if not path.is_absolute():
                raise SupervisorContractError(f"{name} must be absolute: {path}")
            if ".." in path.parts:
                raise SupervisorContractError(f"{name} must not contain '..': {path}")
        if not self.log_dir.is_relative_to(self.project_root):
            raise SupervisorContractError("supervisor logs must be inside the project")
        if self.lock_path == GPU_PHASE_LOCK_PATH:
            raise SupervisorContractError(
                "supervisor singleton lock must differ from the delegated CLI GPU-phase lock"
            )
        if not self.command:
            raise SupervisorContractError("a delegated CLI command is required")
        if any("\0" in value or "\n" in value or "\r" in value for value in self.command):
            raise SupervisorContractError("delegated command arguments must be single-line text")
        executable = Path(self.command[0])
        if not executable.is_absolute():
            raise SupervisorContractError(
                f"delegated command executable must be absolute: {self.command[0]!r}"
            )
        if self.retry_delays_seconds != DEFAULT_RETRY_DELAYS_SECONDS:
            raise SupervisorContractError(
                "retry delays are frozen at exactly 1, 5, and 15 minutes"
            )
        if self.poll_seconds <= 0:
            raise SupervisorContractError("poll_seconds must be positive")

    @property
    def state_path(self) -> Path:
        return self.log_dir / "state.json"

    @property
    def event_log_path(self) -> Path:
        return self.log_dir / "supervisor.log"

    @property
    def failure_path(self) -> Path:
        return self.log_dir / "PERMANENT_FAILURE.json"

    @property
    def command_sha256(self) -> str:
        return sha256_json(
            {
                "command": list(self.command),
                "project_root": str(self.project_root),
                "tag": self.tag,
            }
        )


def _decode_mountinfo_field(value: str) -> str:
    """Decode the octal escapes used in Linux mountinfo paths."""

    replacements = {r"\040": " ", r"\011": "\t", r"\012": "\n", r"\134": "\\"}
    for encoded, decoded in replacements.items():
        value = value.replace(encoded, decoded)
    return value


def filesystem_type_for_path(
    path: str | Path, *, mountinfo_path: str | Path = "/proc/self/mountinfo"
) -> str | None:
    """Return the longest-prefix mount's filesystem type without using mtimes."""

    target = Path(path).resolve(strict=False)
    best: tuple[int, str] | None = None
    try:
        lines = Path(mountinfo_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_point = Path(_decode_mountinfo_field(fields[4])).resolve(strict=False)
            fs_type = fields[separator + 1]
        except (IndexError, ValueError):
            continue
        if target == mount_point or target.is_relative_to(mount_point):
            score = len(mount_point.parts)
            if best is None or score > best[0]:
                best = (score, fs_type)
    return None if best is None else best[1]


def boot_readiness_reason(spec: SupervisorSpec) -> str | None:
    """Describe why the NFS project/venv is not ready, or return ``None``."""

    _assert_gu30()
    filesystem_type = filesystem_type_for_path(spec.project_root)
    if filesystem_type not in {"nfs", "nfs4"}:
        return f"project filesystem is not ready as NFS (found {filesystem_type!r})"
    if not spec.project_root.is_dir():
        return f"project directory is unavailable: {spec.project_root}"
    if not (spec.project_root / "pyproject.toml").is_file():
        return f"project sentinel is unavailable: {spec.project_root / 'pyproject.toml'}"
    python = spec.venv_dir / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        return f"virtual-environment Python is unavailable: {python}"
    if not (spec.venv_dir / "pyvenv.cfg").is_file():
        return f"virtual environment is incomplete: {spec.venv_dir}"
    executable = Path(spec.command[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return f"delegated CLI executable is unavailable: {executable}"
    return None


def wait_for_boot_readiness(
    spec: SupervisorSpec,
    *,
    sleep: Callable[[float], None] = time.sleep,
    readiness: Callable[[SupervisorSpec], str | None] = boot_readiness_reason,
    announce: Callable[[str], None] = print,
) -> None:
    """Wait for the NFS project and venv; host mismatch is never retried."""

    previous: str | None = None
    while (reason := readiness(spec)) is not None:
        if reason != previous:
            announce(f"WAIT_BOOT: {reason}")
            previous = reason
        sleep(spec.poll_seconds)


@contextmanager
def tagged_singleton_lock(spec: SupervisorSpec) -> Iterator[None]:
    """Hold a tagged local flock for the complete wait/train lifecycle."""

    spec.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceUnavailableError(
                f"tagged supervisor lock is already held: {spec.lock_path}"
            ) from exc
        metadata = {
            "command_sha256": spec.command_sha256,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "tag": spec.tag,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(metadata, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def wait_for_tagged_singleton_lock(
    spec: SupervisorSpec,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    """Wait for the supervisor-only singleton lock without disturbing its owner."""

    announced = False
    lock_context = tagged_singleton_lock(spec)
    while True:
        try:
            lock_context.__enter__()
            break
        except ResourceUnavailableError:
            if not announced:
                _emit(spec, f"WAIT_LOCK: {spec.lock_path} is held; owner left untouched")
                announced = True
            sleep(spec.poll_seconds)
            lock_context = tagged_singleton_lock(spec)
    try:
        yield
    finally:
        lock_context.__exit__(None, None, None)


def _initial_state(spec: SupervisorSpec) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "tag": spec.tag,
        "command": list(spec.command),
        "command_sha256": spec.command_sha256,
        "status": "initialized",
        "launch_count": 0,
        "transient_failures": 0,
        "last_exit_code": None,
        "pending_retry_delay_seconds": None,
    }


def _load_state(spec: SupervisorSpec) -> dict[str, Any]:
    if not spec.state_path.exists():
        return _initial_state(spec)
    try:
        state = json.loads(spec.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorContractError(f"cannot read supervisor state: {exc}") from exc
    if not isinstance(state, dict):
        raise SupervisorContractError("supervisor state must be a JSON object")
    expected = {
        "version": STATE_VERSION,
        "tag": spec.tag,
        "command": list(spec.command),
        "command_sha256": spec.command_sha256,
    }
    mismatches = [
        f"{key}={state.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if state.get(key) != value
    ]
    if mismatches:
        raise SupervisorContractError(
            "persistent supervisor identity drift: " + "; ".join(mismatches)
        )
    if state.get("status") not in _VALID_STATES:
        raise SupervisorContractError(f"invalid persistent status: {state.get('status')!r}")
    for field in ("launch_count", "transient_failures"):
        if not isinstance(state.get(field), int) or state[field] < 0:
            raise SupervisorContractError(f"invalid persistent {field}: {state.get(field)!r}")
    if state.get("status") in {"permanent_failure", "retry_exhausted"}:
        raise PreviousTerminalStateError(
            f"supervisor is stopped in terminal state {state['status']!r}; "
            f"inspect {spec.failure_path}"
        )
    return state


def _persist_state(spec: SupervisorSpec, state: dict[str, Any]) -> None:
    write_json(spec.state_path, state)


def _emit(spec: SupervisorSpec, message: str) -> None:
    """Append and fsync a numbered-state message; timestamps are not required."""

    rendered = f"[{spec.tag}] {message}"
    print(rendered, flush=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    with spec.event_log_path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def wait_for_host_resources(
    spec: SupervisorSpec,
    *,
    sleep: Callable[[float], None] = time.sleep,
    gate: Callable[..., Any] = assert_host_ready,
) -> None:
    """Wait without killing or signalling any GPU process."""

    last_reason: str | None = None
    while True:
        try:
            gate(
                required_gpu_count=2,
                minimum_mem_available_kib=16 * 1024 * 1024,
                allowed_pids=tuple(sorted(descendant_pids())),
            )
            if last_reason is not None:
                _emit(spec, "READY_GPU: both GPUs are free and MemAvailable >= 16 GiB")
            return
        except ResourceUnavailableError as exc:
            reason = str(exc)
            if reason != last_reason:
                _emit(spec, f"WAIT_GPU: {reason}")
                last_reason = reason
            sleep(spec.poll_seconds)


def _terminal_failure(
    spec: SupervisorSpec,
    state: dict[str, Any],
    *,
    status: str,
    reason: str,
    exit_code: int,
) -> None:
    state.update(
        {
            "status": status,
            "last_exit_code": exit_code,
            "pending_retry_delay_seconds": None,
            "failure_reason": reason,
        }
    )
    _persist_state(spec, state)
    write_json(
        spec.failure_path,
        {
            "command_sha256": spec.command_sha256,
            "exit_code": exit_code,
            "reason": reason,
            "status": status,
            "tag": spec.tag,
        },
    )
    _emit(spec, f"STOPPED_PERMANENTLY: {reason}; inspect {spec.failure_path}")


def _run_delegated_command(
    spec: SupervisorSpec,
    state: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    launch = int(state["launch_count"]) + 1
    state.update(
        {
            "status": "running",
            "launch_count": launch,
            "last_exit_code": None,
            "pending_retry_delay_seconds": None,
        }
    )
    _persist_state(spec, state)
    output_path = spec.log_dir / f"command-attempt-{launch:04d}.log"
    _emit(
        spec,
        f"LAUNCH {launch}: command_sha256={spec.command_sha256} output={output_path}",
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with output_path.open("ab", buffering=0) as output:
        completed = runner(
            list(spec.command),
            cwd=spec.project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def supervise(
    spec: SupervisorSpec,
    *,
    sleep: Callable[[float], None] = time.sleep,
    readiness: Callable[[SupervisorSpec], str | None] = boot_readiness_reason,
    gate: Callable[..., Any] = assert_host_ready,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    """Run the delegated resume-capable CLI until success or a terminal stop."""

    wait_for_boot_readiness(spec, sleep=sleep, readiness=readiness)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    install_frozen_environment()
    with wait_for_tagged_singleton_lock(spec, sleep=sleep):
        state = _load_state(spec)
        if state["status"] == "complete":
            _emit(spec, "ALREADY_COMPLETE: no command launched")
            return 0

        # A reboot while the delegated command was running is not itself a
        # recorded command failure.  The delegated CLI decides which complete
        # checkpoint to resume from.
        if state["status"] == "running":
            state["status"] = "interrupted"
            _persist_state(spec, state)
            _emit(spec, "RECOVER_INTERRUPTED_LAUNCH: delegating resume to the CLI")

        pending = state.get("pending_retry_delay_seconds")
        if state["status"] == "waiting_retry":
            if pending not in DEFAULT_RETRY_DELAYS_SECONDS:
                raise SupervisorContractError(f"invalid pending retry delay: {pending!r}")
            _emit(spec, f"REAPPLY_RETRY_DELAY: {pending} seconds after supervisor restart")
            sleep(int(pending))

        while True:
            state["status"] = "waiting_resources"
            _persist_state(spec, state)
            wait_for_host_resources(spec, sleep=sleep, gate=gate)
            return_code = _run_delegated_command(spec, state, runner=runner)
            state["last_exit_code"] = return_code
            if return_code == 0:
                state.update(
                    {
                        "status": "complete",
                        "pending_retry_delay_seconds": None,
                    }
                )
                _persist_state(spec, state)
                _emit(
                    spec,
                    f"COMPLETE: delegated command succeeded on launch {state['launch_count']}",
                )
                return 0
            if return_code != TRANSIENT_EXIT_CODE:
                reason = (
                    f"delegated command returned permanent exit code {return_code}; "
                    "only exit code 75 is retryable"
                )
                _terminal_failure(
                    spec,
                    state,
                    status="permanent_failure",
                    reason=reason,
                    exit_code=return_code,
                )
                return return_code if 0 < return_code <= 255 else PERMANENT_EXIT_CODE

            failures = int(state["transient_failures"]) + 1
            state["transient_failures"] = failures
            if failures > len(spec.retry_delays_seconds):
                reason = (
                    f"delegated command returned transient exit code 75 {failures} times; "
                    "the 1/5/15-minute retry budget is exhausted"
                )
                _terminal_failure(
                    spec,
                    state,
                    status="retry_exhausted",
                    reason=reason,
                    exit_code=TRANSIENT_EXIT_CODE,
                )
                return TRANSIENT_EXIT_CODE

            delay = spec.retry_delays_seconds[failures - 1]
            state.update(
                {
                    "status": "waiting_retry",
                    "pending_retry_delay_seconds": delay,
                }
            )
            _persist_state(spec, state)
            _emit(
                spec,
                f"TRANSIENT_FAILURE {failures}: retrying after {delay} seconds",
            )
            sleep(delay)


def _common_supervisor_argv(spec: SupervisorSpec, action: str) -> list[str]:
    python = spec.venv_dir / "bin" / "python"
    return [
        str(python),
        "-m",
        "janus_ts.supervisor",
        action,
        "--tag",
        spec.tag,
        "--project-root",
        str(spec.project_root),
        "--venv-dir",
        str(spec.venv_dir),
        "--lock-path",
        str(spec.lock_path),
        "--log-dir",
        str(spec.log_dir),
        "--poll-seconds",
        str(spec.poll_seconds),
        "--",
        *spec.command,
    ]


def launch_tmux(
    spec: SupervisorSpec,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    """Start one detached tagged tmux session, or leave the existing one alone.

    Returns ``True`` when a new session was started and ``False`` when the
    exact tagged session already existed.
    """

    wait_for_boot_readiness(spec)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    target = f"={spec.tag}"
    existing = runner(
        ["/usr/bin/tmux", "has-session", "-t", target],
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0:
        _emit(spec, f"TMUX_EXISTS: session {spec.tag!r} was left untouched")
        return False
    if existing.returncode != 1:
        raise SupervisorContractError(
            f"tmux has-session failed with exit {existing.returncode}: {existing.stderr.strip()}"
        )
    tmux_output = spec.log_dir / "tmux.log"
    supervisor_command = shlex.join(_common_supervisor_argv(spec, "run"))
    shell_command = (
        f"exec {supervisor_command} >> {shlex.quote(str(tmux_output))} 2>&1"
    )
    started = runner(
        [
            "/usr/bin/tmux",
            "new-session",
            "-d",
            "-s",
            spec.tag,
            shell_command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode != 0:
        raise SupervisorContractError(
            f"tmux new-session failed with exit {started.returncode}: {started.stderr.strip()}"
        )
    _emit(spec, f"TMUX_STARTED: detached session {spec.tag!r}")
    return True


def crontab_marker(tag: str) -> str:
    if not _TAG_PATTERN.fullmatch(tag):
        raise SupervisorContractError(f"invalid supervisor tag: {tag!r}")
    return f"{CRONTAB_MARKER_PREFIX}{tag}"


def _assert_gu30() -> None:
    hostname = socket.gethostname()
    if not hostname.startswith("gu30"):
        raise RuntimeContractError(f"formal supervisor is pinned to gu30, found {hostname!r}")


def build_reboot_entry(spec: SupervisorSpec) -> str:
    """Build an absolute, NFS-aware, idempotently tagged ``@reboot`` entry."""

    python = spec.venv_dir / "bin" / "python"
    launcher = shlex.join(_common_supervisor_argv(spec, "launch-tmux"))
    project = shlex.quote(str(spec.project_root))
    python_path = shlex.quote(str(python))
    sentinel = shlex.quote(str(spec.project_root / "pyproject.toml"))
    body = (
        "while :; do "
        f"fs=$(/usr/bin/findmnt -n -o FSTYPE -T {project} 2>/dev/null || true); "
        f"if {{ [ \"$fs\" = nfs ] || [ \"$fs\" = nfs4 ]; }} "
        f"&& [ -x {python_path} ] && [ -f {sentinel} ]; then break; fi; "
        "/usr/bin/sleep 30; "
        "done; "
        f"exec {launcher}"
    )
    return f"@reboot /usr/bin/bash -lc {shlex.quote(body)} {crontab_marker(spec.tag)}"


def merge_tagged_crontab(existing: str, entry: str, *, tag: str) -> str:
    """Replace this project's tagged entry while preserving every other line."""

    marker = crontab_marker(tag)
    if marker not in entry or "\n" in entry or not entry.startswith("@reboot "):
        raise SupervisorContractError("new crontab entry is not a single tagged @reboot line")
    lines = existing.splitlines()
    replacement_index: int | None = None
    preserved: list[str] = []
    for line in lines:
        if marker in line:
            if replacement_index is None:
                replacement_index = len(preserved)
            continue
        preserved.append(line)
    if replacement_index is None:
        if preserved and preserved[-1] != "":
            preserved.append("")
        preserved.append(entry)
    else:
        preserved.insert(replacement_index, entry)
    return "\n".join(preserved).rstrip("\n") + "\n"


def install_reboot_crontab(
    spec: SupervisorSpec,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    """Install/update only this tagged line in the current user's crontab."""

    _assert_gu30()
    listed = runner(
        ["/usr/bin/crontab", "-l"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode == 0:
        existing = listed.stdout
    elif listed.returncode == 1 and (
        not listed.stderr.strip() or "no crontab" in listed.stderr.lower()
    ):
        existing = ""
    else:
        raise SupervisorContractError(
            f"crontab -l failed with exit {listed.returncode}: {listed.stderr.strip()}"
        )
    merged = merge_tagged_crontab(existing, build_reboot_entry(spec), tag=spec.tag)
    if merged == existing:
        return False
    installed = runner(
        ["/usr/bin/crontab", "-"],
        input=merged,
        capture_output=True,
        text=True,
        check=False,
    )
    if installed.returncode != 0:
        raise SupervisorContractError(
            f"crontab install failed with exit {installed.returncode}: "
            f"{installed.stderr.strip()}"
        )
    return True


def _extract_command(values: Sequence[str]) -> tuple[str, ...]:
    command = tuple(values)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SupervisorContractError("supply the delegated CLI command after --")
    return command


def _spec_from_args(args: argparse.Namespace) -> SupervisorSpec:
    return SupervisorSpec(
        tag=args.tag,
        project_root=Path(args.project_root),
        venv_dir=Path(args.venv_dir),
        lock_path=Path(args.lock_path),
        log_dir=Path(args.log_dir),
        command=_extract_command(args.command),
        poll_seconds=args.poll_seconds,
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--venv-dir", default=str(DEFAULT_VENV_DIR))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="absolute resume-capable CLI command, preceded by --",
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "launch-tmux", "render-reboot", "install-reboot"):
        child = subparsers.add_parser(action)
        _add_common_arguments(child)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        spec = _spec_from_args(args)
        if args.action == "run":
            return supervise(spec)
        if args.action == "launch-tmux":
            launch_tmux(spec)
            return 0
        if args.action == "render-reboot":
            print(build_reboot_entry(spec))
            return 0
        if args.action == "install-reboot":
            changed = install_reboot_crontab(spec)
            print("installed" if changed else "already-current")
            return 0
        raise AssertionError(args.action)
    except PreviousTerminalStateError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return PERMANENT_EXIT_CODE
    except (SupervisorContractError, RuntimeContractError) as exc:
        print(f"PERMANENT SUPERVISOR ERROR: {exc}", file=sys.stderr)
        return PERMANENT_EXIT_CODE
    except ResourceUnavailableError as exc:
        print(f"TRANSIENT SUPERVISOR ERROR: {exc}", file=sys.stderr)
        return TRANSIENT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CRONTAB_MARKER_PREFIX",
    "DEFAULT_LOCK_PATH",
    "DEFAULT_LOG_DIR",
    "DEFAULT_PROJECT_ROOT",
    "DEFAULT_RETRY_DELAYS_SECONDS",
    "DEFAULT_TAG",
    "DEFAULT_VENV_DIR",
    "GPU_PHASE_LOCK_PATH",
    "PERMANENT_EXIT_CODE",
    "PreviousTerminalStateError",
    "SupervisorContractError",
    "SupervisorSpec",
    "TRANSIENT_EXIT_CODE",
    "boot_readiness_reason",
    "build_reboot_entry",
    "crontab_marker",
    "filesystem_type_for_path",
    "install_reboot_crontab",
    "launch_tmux",
    "main",
    "merge_tagged_crontab",
    "supervise",
    "tagged_singleton_lock",
    "wait_for_tagged_singleton_lock",
    "wait_for_boot_readiness",
    "wait_for_host_resources",
]
