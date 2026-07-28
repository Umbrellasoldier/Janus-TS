from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from janus_ts import supervisor
from janus_ts.runtime import ResourceUnavailableError, RuntimeContractError


def make_spec(tmp_path: Path, *, command: tuple[str, ...] = ("/bin/true",)):
    project = tmp_path / "project"
    project.mkdir()
    venv = project / ".venv"
    (venv / "bin").mkdir(parents=True)
    return supervisor.SupervisorSpec(
        tag="janus-test",
        project_root=project,
        venv_dir=venv,
        lock_path=tmp_path / "locks" / "janus-test.lock",
        log_dir=project / "artifacts" / "logs" / "supervisor",
        command=command,
        poll_seconds=2,
    )


def completed(returncode: int, *, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_spec_requires_absolute_executable_and_frozen_retry_delays(tmp_path: Path):
    spec = make_spec(tmp_path)
    with pytest.raises(supervisor.SupervisorContractError, match="absolute"):
        supervisor.SupervisorSpec(
            tag=spec.tag,
            project_root=spec.project_root,
            venv_dir=spec.venv_dir,
            lock_path=spec.lock_path,
            log_dir=spec.log_dir,
            command=("python",),
        )
    with pytest.raises(supervisor.SupervisorContractError, match="1, 5, and 15"):
        supervisor.SupervisorSpec(
            tag=spec.tag,
            project_root=spec.project_root,
            venv_dir=spec.venv_dir,
            lock_path=spec.lock_path,
            log_dir=spec.log_dir,
            command=spec.command,
            retry_delays_seconds=(1,),
        )


def test_supervisor_singleton_lock_is_distinct_from_child_gpu_phase_lock(tmp_path: Path):
    assert supervisor.DEFAULT_LOCK_PATH != supervisor.GPU_PHASE_LOCK_PATH
    spec = make_spec(tmp_path)
    with pytest.raises(supervisor.SupervisorContractError, match="must differ"):
        supervisor.SupervisorSpec(
            tag=spec.tag,
            project_root=spec.project_root,
            venv_dir=spec.venv_dir,
            lock_path=supervisor.GPU_PHASE_LOCK_PATH,
            log_dir=spec.log_dir,
            command=spec.command,
        )


def test_filesystem_type_uses_longest_mount_prefix(tmp_path: Path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 0:1 / / rw - ext4 root rw\n"
        "2 1 0:2 / /mnt/sto3 rw - nfs4 server:/home rw\n",
        encoding="utf-8",
    )
    assert (
        supervisor.filesystem_type_for_path(
            "/mnt/sto3/caoxiangyu/Janus-TS", mountinfo_path=mountinfo
        )
        == "nfs4"
    )


def test_boot_wait_retries_paths_but_host_mismatch_is_permanent(tmp_path: Path):
    spec = make_spec(tmp_path)
    answers = iter(["NFS absent", "venv absent", None])
    sleeps: list[float] = []
    messages: list[str] = []
    supervisor.wait_for_boot_readiness(
        spec,
        readiness=lambda unused: next(answers),
        sleep=sleeps.append,
        announce=messages.append,
    )
    assert sleeps == [2, 2]
    assert messages == ["WAIT_BOOT: NFS absent", "WAIT_BOOT: venv absent"]

    with pytest.raises(RuntimeContractError):
        supervisor.wait_for_boot_readiness(
            spec,
            readiness=lambda unused: (_ for _ in ()).throw(RuntimeContractError("not gu30")),
            sleep=sleeps.append,
        )


def test_host_gate_only_waits_and_never_signals_processes(tmp_path: Path, monkeypatch):
    spec = make_spec(tmp_path)
    calls = 0
    sleeps: list[float] = []

    def gate(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["required_gpu_count"] == 2
        assert kwargs["minimum_mem_available_kib"] == 16 * 1024 * 1024
        if calls == 1:
            raise ResourceUnavailableError("foreign pid=123")
        return object()

    def forbidden_kill(*args, **kwargs):
        raise AssertionError("the supervisor must never kill a GPU process")

    monkeypatch.setattr(supervisor.os, "kill", forbidden_kill)
    supervisor.wait_for_host_resources(spec, gate=gate, sleep=sleeps.append)
    assert calls == 2
    assert sleeps == [2]


def test_tagged_lock_is_nonblocking_singleton(tmp_path: Path):
    spec = make_spec(tmp_path)
    with supervisor.tagged_singleton_lock(spec):
        metadata = json.loads(spec.lock_path.read_text(encoding="utf-8"))
        assert metadata["tag"] == spec.tag
        assert metadata["command_sha256"] == spec.command_sha256
        with (
            pytest.raises(ResourceUnavailableError, match="already held"),
            supervisor.tagged_singleton_lock(spec),
        ):
            pass


def test_supervise_retries_only_exit_75_at_frozen_delays(tmp_path: Path, monkeypatch):
    spec = make_spec(tmp_path)
    results = iter([75, 75, 75, 0])
    sleeps: list[float] = []
    launches: list[list[str]] = []

    def runner(command, **kwargs):
        launches.append(command)
        assert kwargs["cwd"] == spec.project_root
        assert kwargs["check"] is False
        return completed(next(results))

    monkeypatch.setattr(supervisor, "install_frozen_environment", lambda: None)
    assert (
        supervisor.supervise(
            spec,
            readiness=lambda unused: None,
            gate=lambda **kwargs: object(),
            sleep=sleeps.append,
            runner=runner,
        )
        == 0
    )
    state = json.loads(spec.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["launch_count"] == 4
    assert state["transient_failures"] == 3
    assert sleeps == [60, 300, 900]
    assert launches == [list(spec.command)] * 4


def test_permanent_command_failure_stops_and_survives_restart(tmp_path: Path, monkeypatch):
    spec = make_spec(tmp_path)
    launches = 0

    def runner(command, **kwargs):
        nonlocal launches
        launches += 1
        return completed(137)

    monkeypatch.setattr(supervisor, "install_frozen_environment", lambda: None)
    kwargs = {
        "readiness": lambda unused: None,
        "gate": lambda **unused: object(),
        "sleep": lambda unused: None,
        "runner": runner,
    }
    assert supervisor.supervise(spec, **kwargs) == 137
    assert launches == 1
    failure = json.loads(spec.failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "permanent_failure"
    assert failure["exit_code"] == 137
    with pytest.raises(supervisor.PreviousTerminalStateError):
        supervisor.supervise(spec, **kwargs)
    assert launches == 1


def test_restart_of_running_state_delegates_resume_without_mtime(
    tmp_path: Path, monkeypatch
):
    spec = make_spec(tmp_path)
    spec.log_dir.mkdir(parents=True)
    state = supervisor._initial_state(spec)
    state.update({"status": "running", "launch_count": 1})
    supervisor._persist_state(spec, state)
    monkeypatch.setattr(supervisor, "install_frozen_environment", lambda: None)
    assert (
        supervisor.supervise(
            spec,
            readiness=lambda unused: None,
            gate=lambda **unused: object(),
            sleep=lambda unused: None,
            runner=lambda command, **kwargs: completed(0),
        )
        == 0
    )
    recovered = json.loads(spec.state_path.read_text(encoding="utf-8"))
    assert recovered["launch_count"] == 2
    assert recovered["status"] == "complete"


def test_crontab_merge_is_idempotent_and_preserves_unrelated_lines(tmp_path: Path):
    spec = make_spec(tmp_path)
    entry = supervisor.build_reboot_entry(spec)
    assert entry.startswith("@reboot /usr/bin/bash -lc ")
    assert supervisor.crontab_marker(spec.tag) in entry
    assert str(spec.project_root) in entry
    assert "/usr/bin/findmnt" in entry
    assert "/usr/bin/sleep 30" in entry
    existing = (
        "MAILTO=user@example.org\n"
        "0 3 * * * /usr/local/bin/backup\n"
        "@reboot /bin/old # JANUS_TS_SUPERVISOR:janus-test\n"
        "@reboot /bin/other # OTHER_PROJECT\n"
    )
    merged = supervisor.merge_tagged_crontab(existing, entry, tag=spec.tag)
    assert "MAILTO=user@example.org" in merged
    assert "0 3 * * * /usr/local/bin/backup" in merged
    assert "# OTHER_PROJECT" in merged
    assert "/bin/old" not in merged
    assert merged.count(supervisor.crontab_marker(spec.tag)) == 1
    assert supervisor.merge_tagged_crontab(merged, entry, tag=spec.tag) == merged


def test_install_crontab_uses_list_then_stdin_and_is_idempotent(tmp_path: Path):
    spec = make_spec(tmp_path)
    calls: list[tuple[list[str], dict]] = []
    installed = ""

    def first_runner(command, **kwargs):
        nonlocal installed
        calls.append((command, kwargs))
        if command[-1] == "-l":
            return completed(0, stdout="1 2 * * * /bin/true\n")
        installed = kwargs["input"]
        return completed(0)

    assert supervisor.install_reboot_crontab(spec, runner=first_runner)
    assert calls[0][0] == ["/usr/bin/crontab", "-l"]
    assert calls[1][0] == ["/usr/bin/crontab", "-"]
    assert "1 2 * * * /bin/true" in installed

    def second_runner(command, **kwargs):
        assert command == ["/usr/bin/crontab", "-l"]
        return completed(0, stdout=installed)

    assert not supervisor.install_reboot_crontab(spec, runner=second_runner)


def test_tmux_existing_session_is_not_replaced(tmp_path: Path, monkeypatch):
    spec = make_spec(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(supervisor, "wait_for_boot_readiness", lambda unused: None)

    def runner(command, **kwargs):
        calls.append(command)
        return completed(0)

    assert not supervisor.launch_tmux(spec, runner=runner)
    assert calls == [["/usr/bin/tmux", "has-session", "-t", f"={spec.tag}"]]


def test_tmux_new_session_contains_exact_delegated_command(tmp_path: Path, monkeypatch):
    command = ("/bin/echo", "argument with spaces", "--resume")
    spec = make_spec(tmp_path, command=command)
    calls: list[list[str]] = []
    monkeypatch.setattr(supervisor, "wait_for_boot_readiness", lambda unused: None)

    def runner(argv, **kwargs):
        calls.append(argv)
        return completed(1 if len(calls) == 1 else 0)

    assert supervisor.launch_tmux(spec, runner=runner)
    assert calls[1][:6] == [
        "/usr/bin/tmux",
        "new-session",
        "-d",
        "-s",
        spec.tag,
        calls[1][5],
    ]
    shell_command = calls[1][5]
    assert "janus_ts.supervisor run" in shell_command
    assert "'argument with spaces'" in shell_command
    assert "--resume" in shell_command
