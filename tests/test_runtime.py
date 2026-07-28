from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from janus_ts.runtime import (
    EXTERNAL_GPU_ALLOWLIST_SCHEMA_VERSION,
    ComputeProcess,
    RuntimeContractError,
    approved_external_gpu_pids,
    audit_external_gpu_allowlist,
)

PID = 4321
START_TIME = 987654
UID = 1050
EXECUTABLE = "/opt/lammps/bin/lmp"
PROCESS_NAME = "lmp"
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"
CMDLINE = EXECUTABLE.encode() + b"\0-pk\0gpu\0"


def _entry(*, pid: int = PID, start_time_ticks: int = START_TIME) -> dict[str, object]:
    return {
        "pid": pid,
        "start_time_ticks": start_time_ticks,
        "uid": UID,
        "executable_path": EXECUTABLE,
        "process_name": PROCESS_NAME,
        "gpu_uuid": GPU_UUID,
        "cmdline_sha256": hashlib.sha256(CMDLINE).hexdigest(),
    }


def _write_allowlist(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": EXTERNAL_GPU_ALLOWLIST_SCHEMA_VERSION,
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_proc_identity(
    proc_root: Path,
    *,
    pid: int = PID,
    start_time_ticks: int = START_TIME,
    uid: int = UID,
    process_name: str = PROCESS_NAME,
    cmdline: bytes = CMDLINE,
) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    # The suffix starts at proc_pid_stat(5) field 3; start_time is field 22.
    stat_suffix = ["S", *("0" for _ in range(18)), str(start_time_ticks), "0"]
    (process_dir / "stat").write_text(
        f"{pid} ({process_name}) {' '.join(stat_suffix)}\n", encoding="utf-8"
    )
    (process_dir / "status").write_text(
        f"Name:\t{process_name}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="utf-8",
    )
    (process_dir / "cmdline").write_bytes(cmdline)


def _gpu_process(
    *,
    pid: int = PID,
    gpu_uuid: str = GPU_UUID,
    executable_path: str = EXECUTABLE,
) -> ComputeProcess:
    return ComputeProcess(
        gpu_uuid=gpu_uuid,
        pid=pid,
        process_name=executable_path,
        used_memory_mib=512,
    )


def test_approved_external_gpu_pids_returns_only_observed_exact_identities(tmp_path: Path):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(allowlist, [_entry(), _entry(pid=9876, start_time_ticks=111)])
    proc_root = tmp_path / "proc"
    _write_proc_identity(proc_root)

    assert approved_external_gpu_pids(
        allowlist,
        compute_processes=(_gpu_process(), _gpu_process(pid=9999)),
        proc_root=proc_root,
    ) == (PID,)


def test_exited_approved_process_drops_out_even_if_nvidia_observation_races(tmp_path: Path):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(allowlist, [_entry()])

    assert (
        approved_external_gpu_pids(
            allowlist,
            compute_processes=(_gpu_process(),),
            proc_root=tmp_path / "empty-proc",
        )
        == ()
    )


@pytest.mark.parametrize(
    ("proc_changes", "match"),
    [
        ({"start_time_ticks": START_TIME + 1}, "start_time_ticks"),
        ({"uid": UID + 1}, "uids"),
        ({"process_name": "python"}, "process_name"),
        ({"cmdline": b"/opt/other/lmp\0"}, "executable_path, cmdline_sha256"),
    ],
)
def test_proc_identity_mismatch_fails_closed(
    tmp_path: Path, proc_changes: dict[str, object], match: str
):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(allowlist, [_entry()])
    proc_root = tmp_path / "proc"
    _write_proc_identity(proc_root, **proc_changes)

    with pytest.raises(RuntimeContractError, match=match):
        approved_external_gpu_pids(
            allowlist,
            compute_processes=(_gpu_process(),),
            proc_root=proc_root,
        )


@pytest.mark.parametrize(
    "process",
    [
        _gpu_process(gpu_uuid="GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"),
        _gpu_process(executable_path="/opt/other/lmp"),
    ],
)
def test_nvidia_identity_mismatch_fails_before_permission(process: ComputeProcess, tmp_path: Path):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(allowlist, [_entry()])
    proc_root = tmp_path / "proc"
    _write_proc_identity(proc_root)

    with pytest.raises(RuntimeContractError, match="unexpected"):
        approved_external_gpu_pids(
            allowlist,
            compute_processes=(process,),
            proc_root=proc_root,
        )


def test_duplicate_nvidia_pid_fails_closed(tmp_path: Path):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(allowlist, [_entry()])
    proc_root = tmp_path / "proc"
    _write_proc_identity(proc_root)

    with pytest.raises(RuntimeContractError, match="appears more than once"):
        approved_external_gpu_pids(
            allowlist,
            compute_processes=(_gpu_process(), _gpu_process()),
            proc_root=proc_root,
        )


def test_allowlist_audit_is_static_deterministic_json_provenance(tmp_path: Path):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(
        allowlist,
        [_entry(pid=9876, start_time_ticks=111), _entry()],
    )
    raw = allowlist.read_bytes()

    first = audit_external_gpu_allowlist(allowlist)
    second = audit_external_gpu_allowlist(allowlist)
    assert first == second
    assert first["allowlist_sha256"] == hashlib.sha256(raw).hexdigest()
    assert [item["pid"] for item in first["entries"]] == [PID, 9876]
    assert json.loads(json.dumps(first)) == first


def test_allowlist_rejects_duplicate_pid_and_unknown_fields(tmp_path: Path):
    allowlist = tmp_path / "allowlist.json"
    _write_allowlist(allowlist, [_entry(), _entry()])
    with pytest.raises(RuntimeContractError, match="duplicate PIDs"):
        audit_external_gpu_allowlist(allowlist)

    broken = _entry()
    broken["unreviewed"] = True
    _write_allowlist(allowlist, [broken])
    with pytest.raises(RuntimeContractError, match="unexpected fields"):
        audit_external_gpu_allowlist(allowlist)

    with pytest.raises(RuntimeContractError, match="cannot read"):
        audit_external_gpu_allowlist(tmp_path / "missing.json")


def test_tracked_gu30_allowlist_binds_the_eight_one_time_lammps_processes():
    audit = audit_external_gpu_allowlist()
    assert [item["pid"] for item in audit["entries"]] == [
        1359466,
        1359467,
        1359468,
        1359469,
        1376626,
        1376627,
        1376628,
        1376629,
    ]
    assert {item["process_name"] for item in audit["entries"]} == {"lmp"}
    assert {item["uid"] for item in audit["entries"]} == {1050}
