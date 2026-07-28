"""Host-resource and process-safety gates for gu30."""

from __future__ import annotations

import csv
import fcntl
import os
import socket
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


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
    "GpuStatus",
    "HostStatus",
    "ResourceUnavailableError",
    "RuntimeContractError",
    "assert_host_ready",
    "descendant_pids",
    "exclusive_lock",
    "frozen_distributed_environment",
    "host_status",
    "install_frozen_environment",
    "query_compute_processes",
    "query_gpus",
]
