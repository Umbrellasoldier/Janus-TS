"""Host-resource and process-safety gates for gu30."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXTERNAL_GPU_ALLOWLIST_SCHEMA_VERSION = "janus-ts-external-gpu-allowlist-v1"
DEFAULT_EXTERNAL_GPU_ALLOWLIST_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "gu30_gpu_share_allowlist.json"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ResourceUnavailableError(RuntimeError):
    """A transient host-resource condition prevents a safe GPU launch."""


class RuntimeContractError(RuntimeError):
    """The host differs from the frozen execution contract."""


@dataclass(frozen=True, slots=True)
class GpuStatus:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int


@dataclass(frozen=True, slots=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int


@dataclass(frozen=True, slots=True)
class ExternalGpuProcessIdentity:
    """Immutable identity of a one-time approved external GPU process."""

    pid: int
    start_time_ticks: int
    uid: int
    executable_path: str
    process_name: str
    gpu_uuid: str
    cmdline_sha256: str


@dataclass(frozen=True, slots=True)
class _ProcProcessIdentity:
    start_time_ticks: int
    uids: tuple[int, int, int, int]
    executable_path: str
    process_name: str
    cmdline_sha256: str


@dataclass(frozen=True, slots=True)
class HostStatus:
    hostname: str
    mem_available_kib: int
    swap_free_kib: int
    gpus: tuple[GpuStatus, ...]
    compute_processes: tuple[ComputeProcess, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _nvidia_smi(query: str) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeContractError(f"nvidia-smi query failed: {' '.join(command)}") from exc
    if not result.stdout.strip():
        return []
    return [next(csv.reader([line], skipinitialspace=True)) for line in result.stdout.splitlines()]


def query_gpus() -> tuple[GpuStatus, ...]:
    rows = _nvidia_smi("gpu:index,uuid,name,memory.total,memory.used,utilization.gpu")
    try:
        return tuple(
            GpuStatus(
                index=int(row[0]),
                uuid=row[1].strip(),
                name=row[2].strip(),
                memory_total_mib=int(row[3]),
                memory_used_mib=int(row[4]),
                utilization_percent=int(row[5]),
            )
            for row in rows
        )
    except (IndexError, ValueError) as exc:
        raise RuntimeContractError(f"unexpected nvidia-smi GPU output: {rows!r}") from exc


def query_compute_processes() -> tuple[ComputeProcess, ...]:
    rows = _nvidia_smi("compute-apps:gpu_uuid,pid,process_name,used_memory")
    try:
        return tuple(
            ComputeProcess(
                gpu_uuid=row[0].strip(),
                pid=int(row[1]),
                process_name=row[2].strip(),
                used_memory_mib=int(row[3]),
            )
            for row in rows
        )
    except (IndexError, ValueError) as exc:
        raise RuntimeContractError(f"unexpected nvidia-smi process output: {rows!r}") from exc


def _load_external_gpu_allowlist(
    path: str | Path = DEFAULT_EXTERNAL_GPU_ALLOWLIST_PATH,
) -> tuple[tuple[ExternalGpuProcessIdentity, ...], bytes, Path]:
    requested_path = Path(path)
    try:
        allowlist_path = requested_path.resolve(strict=True)
        raw = allowlist_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            f"cannot read external GPU allowlist {requested_path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "entries"}:
        raise RuntimeContractError("external GPU allowlist must contain schema_version and entries")
    if payload["schema_version"] != EXTERNAL_GPU_ALLOWLIST_SCHEMA_VERSION:
        raise RuntimeContractError(
            f"unexpected external GPU allowlist schema: {payload['schema_version']!r}"
        )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise RuntimeContractError("external GPU allowlist entries must be a list")

    expected_fields = {
        "pid",
        "start_time_ticks",
        "uid",
        "executable_path",
        "process_name",
        "gpu_uuid",
        "cmdline_sha256",
    }
    entries: list[ExternalGpuProcessIdentity] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise RuntimeContractError(
                f"external GPU allowlist entry {index} has unexpected fields"
            )
        for field in ("pid", "start_time_ticks", "uid"):
            if type(item[field]) is not int:  # bool must not pass as an integer identity
                raise RuntimeContractError(
                    f"external GPU allowlist entry {index} {field} must be an integer"
                )
        for field in ("executable_path", "process_name", "gpu_uuid", "cmdline_sha256"):
            if not isinstance(item[field], str) or not item[field]:
                raise RuntimeContractError(
                    f"external GPU allowlist entry {index} {field} must be a non-empty string"
                )
        entry = ExternalGpuProcessIdentity(**item)
        if entry.pid <= 0 or entry.start_time_ticks <= 0 or entry.uid < 0:
            raise RuntimeContractError(
                f"external GPU allowlist entry {index} has an invalid numeric identity"
            )
        if not Path(entry.executable_path).is_absolute():
            raise RuntimeContractError(
                f"external GPU allowlist entry {index} executable_path must be absolute"
            )
        if not entry.gpu_uuid.startswith("GPU-"):
            raise RuntimeContractError(
                f"external GPU allowlist entry {index} has an invalid GPU UUID"
            )
        if _SHA256_PATTERN.fullmatch(entry.cmdline_sha256) is None:
            raise RuntimeContractError(
                f"external GPU allowlist entry {index} has an invalid cmdline SHA-256"
            )
        entries.append(entry)
    if len({entry.pid for entry in entries}) != len(entries):
        raise RuntimeContractError("external GPU allowlist contains duplicate PIDs")
    return tuple(sorted(entries, key=lambda entry: entry.pid)), raw, allowlist_path


def audit_external_gpu_allowlist(
    path: str | Path = DEFAULT_EXTERNAL_GPU_ALLOWLIST_PATH,
) -> dict[str, Any]:
    """Return deterministic, JSON-safe provenance for the tracked allowlist.

    This deliberately does not inspect live processes.  It is suitable for an
    immutable launch fingerprint even after one or all approved processes exit.
    """

    entries, raw, allowlist_path = _load_external_gpu_allowlist(path)
    return {
        "schema_version": EXTERNAL_GPU_ALLOWLIST_SCHEMA_VERSION,
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": hashlib.sha256(raw).hexdigest(),
        "entries": [asdict(entry) for entry in entries],
    }


def _read_proc_process_identity(pid: int, *, proc_root: Path) -> _ProcProcessIdentity:
    process_dir = proc_root / str(pid)

    def read_required(path: Path, *, binary: bool = False) -> str | bytes:
        try:
            return path.read_bytes() if binary else path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise RuntimeContractError(f"cannot verify approved GPU process via {path}") from exc

    def start_time() -> int:
        raw_stat = read_required(process_dir / "stat")
        assert isinstance(raw_stat, str)
        prefix, separator, suffix = raw_stat.rstrip("\n").rpartition(") ")
        if not separator or not prefix.startswith(f"{pid} ("):
            raise RuntimeContractError(f"malformed /proc stat for approved GPU PID {pid}")
        fields = suffix.split()
        try:
            return int(fields[19])  # Linux proc_pid_stat(5), field 22
        except (IndexError, ValueError) as exc:
            raise RuntimeContractError(
                f"malformed start_time in /proc stat for approved GPU PID {pid}"
            ) from exc

    first_start_time = start_time()
    raw_status = read_required(process_dir / "status")
    raw_cmdline = read_required(process_dir / "cmdline", binary=True)
    assert isinstance(raw_status, str)
    assert isinstance(raw_cmdline, bytes)

    status_fields: dict[str, str] = {}
    for line in raw_status.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            status_fields[name] = value.strip()
    try:
        process_name = status_fields["Name"]
        uid_fields = status_fields["Uid"].split()
        if len(uid_fields) != 4:
            raise ValueError("Uid must have four values")
        uids = tuple(int(value) for value in uid_fields)
    except (KeyError, ValueError) as exc:
        raise RuntimeContractError(
            f"malformed /proc status for approved GPU PID {pid}"
        ) from exc
    if len(uids) != 4:  # static typing guard for the fixed-width identity tuple
        raise RuntimeContractError(f"malformed /proc Uid identity for approved GPU PID {pid}")

    raw_argv0 = raw_cmdline.split(b"\0", 1)[0]
    if not raw_argv0:
        raise RuntimeContractError(f"empty /proc cmdline for approved GPU PID {pid}")
    try:
        executable_path = raw_argv0.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeContractError(
            f"non-UTF-8 executable path for approved GPU PID {pid}"
        ) from exc
    second_start_time = start_time()
    if second_start_time != first_start_time:
        raise RuntimeContractError(
            f"approved GPU PID {pid} changed identity while it was being verified"
        )
    return _ProcProcessIdentity(
        start_time_ticks=first_start_time,
        uids=(uids[0], uids[1], uids[2], uids[3]),
        executable_path=executable_path,
        process_name=process_name,
        cmdline_sha256=hashlib.sha256(raw_cmdline).hexdigest(),
    )


def approved_external_gpu_pids(
    path: str | Path = DEFAULT_EXTERNAL_GPU_ALLOWLIST_PATH,
    *,
    compute_processes: Sequence[ComputeProcess] | None = None,
    proc_root: str | Path = "/proc",
) -> tuple[int, ...]:
    """Return only live GPU PIDs matching every frozen identity field.

    An allowlisted process that has exited or stopped using a GPU simply falls
    out of the result.  A currently observed GPU process whose PID is listed
    but whose identity has changed is a contract failure, preventing PID reuse
    from inheriting permission.
    """

    entries, _, _ = _load_external_gpu_allowlist(path)
    by_pid = {entry.pid: entry for entry in entries}
    observed = query_compute_processes() if compute_processes is None else tuple(compute_processes)
    approved: set[int] = set()
    for process in observed:
        entry = by_pid.get(process.pid)
        if entry is None:
            continue
        if process.pid in approved:
            raise RuntimeContractError(
                f"approved GPU PID {process.pid} appears more than once in NVIDIA process data"
            )
        if process.gpu_uuid != entry.gpu_uuid:
            raise RuntimeContractError(
                f"approved GPU PID {process.pid} is attached to an unexpected GPU UUID"
            )
        if process.process_name != entry.executable_path:
            raise RuntimeContractError(
                f"approved GPU PID {process.pid} has an unexpected NVIDIA executable path"
            )
        try:
            live = _read_proc_process_identity(process.pid, proc_root=Path(proc_root))
        except FileNotFoundError:
            # The NVIDIA observation can race with normal process exit.  It is
            # safe to omit a vanished identity; no permission is granted.
            continue
        expected_uids = (entry.uid, entry.uid, entry.uid, entry.uid)
        mismatches = {
            "start_time_ticks": (entry.start_time_ticks, live.start_time_ticks),
            "uids": (expected_uids, live.uids),
            "executable_path": (entry.executable_path, live.executable_path),
            "process_name": (entry.process_name, live.process_name),
            "cmdline_sha256": (entry.cmdline_sha256, live.cmdline_sha256),
        }
        changed = [name for name, (expected, actual) in mismatches.items() if expected != actual]
        if changed:
            raise RuntimeContractError(
                f"approved GPU PID {process.pid} identity mismatch: {', '.join(changed)}"
            )
        approved.add(process.pid)
    return tuple(sorted(approved))


def _memory_status() -> tuple[int, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="ascii") as handle:
        for line in handle:
            name, raw_value = line.split(":", 1)
            first = raw_value.strip().split()[0]
            values[name] = int(first)
    try:
        return values["MemAvailable"], values["SwapFree"]
    except KeyError as exc:
        raise RuntimeContractError("/proc/meminfo lacks MemAvailable or SwapFree") from exc


def host_status() -> HostStatus:
    mem_available, swap_free = _memory_status()
    return HostStatus(
        hostname=socket.gethostname(),
        mem_available_kib=mem_available,
        swap_free_kib=swap_free,
        gpus=query_gpus(),
        compute_processes=query_compute_processes(),
    )


def descendant_pids(root_pid: int | None = None) -> set[int]:
    """Return this launcher's process tree using Linux `/proc` parent IDs."""

    root = os.getpid() if root_pid is None else root_pid
    parents: dict[int, int] = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = {}
            for line in status.read_text(encoding="ascii").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key] = value.strip()
            parents[int(fields["Pid"])] = int(fields["PPid"])
        except (OSError, KeyError, ValueError):
            continue
    allowed = {root}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in allowed and pid not in allowed:
                allowed.add(pid)
                changed = True
    return allowed


def assert_host_ready(
    *,
    required_gpu_count: int = 2,
    minimum_mem_available_kib: int = 16 * 1024 * 1024,
    allowed_pids: Sequence[int] = (),
) -> HostStatus:
    status = host_status()
    if not status.hostname.startswith("gu30"):
        raise RuntimeContractError(f"formal run is pinned to gu30, found {status.hostname!r}")
    if len(status.gpus) != required_gpu_count:
        raise RuntimeContractError(
            f"expected {required_gpu_count} GPUs, found {len(status.gpus)}"
        )
    if any(gpu.name != "NVIDIA GeForce RTX 4090" for gpu in status.gpus):
        raise RuntimeContractError(f"unexpected GPU model(s): {[gpu.name for gpu in status.gpus]}")
    if status.mem_available_kib < minimum_mem_available_kib:
        raise ResourceUnavailableError(
            f"MemAvailable={status.mem_available_kib / 1024**2:.2f} GiB is below 16 GiB"
        )
    permitted = set(allowed_pids)
    foreign = [process for process in status.compute_processes if process.pid not in permitted]
    if foreign:
        details = ", ".join(
            f"pid={item.pid} {item.process_name} {item.used_memory_mib}MiB"
            for item in foreign
        )
        raise ResourceUnavailableError(f"GPUs have foreign compute processes: {details}")
    return status


def frozen_distributed_environment() -> dict[str, str]:
    return {
        # Triton's small CUDA-driver helper is compiled lazily.  gu30 has the
        # Python runtime package but not the matching system -devel headers.
        "CPATH": (
            "/home/caoxiangyu/.cache/micromamba/envs/"
            "janus-ts-cuda128/include/python3.11"
        ),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "42",
        "DS_BUILD_OPS": "0",
    }


def install_frozen_environment() -> None:
    for name, value in frozen_distributed_environment().items():
        existing = os.environ.get(name)
        if existing is not None and existing != value:
            raise RuntimeContractError(
                f"environment override rejected: {name}={existing!r}, expected {value!r}"
            )
        os.environ[name] = value


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    """Take a non-blocking host lock; a second launcher exits transiently."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceUnavailableError(f"GPU lock is already held: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


__all__ = [
    "ComputeProcess",
    "DEFAULT_EXTERNAL_GPU_ALLOWLIST_PATH",
    "EXTERNAL_GPU_ALLOWLIST_SCHEMA_VERSION",
    "ExternalGpuProcessIdentity",
    "GpuStatus",
    "HostStatus",
    "ResourceUnavailableError",
    "RuntimeContractError",
    "approved_external_gpu_pids",
    "assert_host_ready",
    "audit_external_gpu_allowlist",
    "descendant_pids",
    "exclusive_lock",
    "frozen_distributed_environment",
    "host_status",
    "install_frozen_environment",
    "query_compute_processes",
    "query_gpus",
]
