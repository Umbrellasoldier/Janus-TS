from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from janus_ts import workflow
from janus_ts.cli import app
from janus_ts.config import load_config
from janus_ts.runtime import ResourceUnavailableError


def _input_directories(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed = tmp_path / "processed"
    checkpoint = tmp_path / "epoch-00005"
    output = tmp_path / "exploration"
    processed.mkdir()
    checkpoint.mkdir()
    return processed, checkpoint, output


def test_thinking_cli_builds_absolute_locked_two_rank_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed, checkpoint, output = _input_directories(tmp_path)
    captured: dict[str, object] = {}

    def locked(phase, command, **kwargs):
        captured.update({"phase": phase, "command": tuple(command), **kwargs})
        return workflow.CommandOutcome(phase, tuple(command), 0)

    monkeypatch.setattr(workflow, "run_locked_gpu_command", locked)
    result = CliRunner().invoke(
        app,
        [
            "infer",
            "thinking",
            "--config",
            "configs/transition1x.yaml",
            "--processed-path",
            str(processed),
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--split",
            "test",
            "--reaction-id",
            "rxn0001",
            "--reaction-id",
            "rxn0002",
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    command = captured["command"]
    assert isinstance(command, tuple)
    assert Path(command[0]).is_absolute()
    assert command[1:5] == (
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "janus_ts.thinking_inference",
    )
    assert command[command.index("--config") + 1] == str(
        Path("configs/transition1x.yaml").resolve()
    )
    assert command[command.index("--processed-path") + 1] == str(processed.resolve())
    assert command[command.index("--checkpoint-dir") + 1] == str(checkpoint.resolve())
    assert command[command.index("--output-dir") + 1] == str(output.resolve())
    assert command.count("--reaction-id") == 2
    assert "--selection-proof" not in command
    assert captured["phase"] == "exploratory-thinking-inference"
    config = load_config("configs/transition1x.yaml")
    assert captured["lock_path"] == config.runtime.gpu_lock_path
    assert captured["required_gpu_count"] == 2
    assert captured["timeout_seconds"] is None
    assert captured["require_clean_after"] is True


def test_thinking_cli_maps_busy_gpu_gate_to_transient_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed, checkpoint, output = _input_directories(tmp_path)

    def busy(*args, **kwargs):
        raise ResourceUnavailableError("foreign GPU process")

    monkeypatch.setattr(workflow, "run_locked_gpu_command", busy)
    result = CliRunner().invoke(
        app,
        [
            "infer",
            "thinking",
            "--processed-path",
            str(processed),
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == workflow.TRANSIENT_EXIT_CODE
    assert "TRANSIENT" in result.stderr


def test_thinking_cli_help_labels_the_command_non_formal() -> None:
    result = CliRunner().invoke(app, ["infer", "thinking", "--help"])

    assert result.exit_code == 0
    assert "without touching formal experiment state" in result.stdout
    assert "--processed-path" in result.stdout
    assert "--checkpoint-dir" in result.stdout
