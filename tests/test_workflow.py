from __future__ import annotations

import json
import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from janus_ts import workflow
from janus_ts.artifacts import mark_complete
from janus_ts.checkpointing import CheckpointIdentity
from janus_ts.cli import app
from janus_ts.config import load_config
from janus_ts.run_state import RunIdentity, RunPaths, initialize_run
from janus_ts.runtime import ResourceUnavailableError


def _run_identity() -> RunIdentity:
    return RunIdentity(
        fingerprint="f" * 64,
        experiment="test",
        config_sha256="c" * 64,
        data_fingerprint="d" * 64,
        pissa_manifest_sha256="p" * 64,
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=8,
    )


def _checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        run_fingerprint="f" * 64,
        config_fingerprint="c" * 64,
        data_fingerprint="d" * 64,
        model_fingerprint="e" * 64,
    )


def _prepared(tmp_path: Path) -> workflow.PreparedRun:
    config = load_config("configs/transition1x.yaml")
    paths = RunPaths(project=tmp_path / "project", local=tmp_path / "local")
    paths.project.mkdir(parents=True)
    paths.local.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    (bundle / "pissa_init").mkdir(parents=True)
    mark_complete(bundle, {"artifact_type": "test-pissa"})
    processed = tmp_path / "processed"
    processed.mkdir()
    return workflow.PreparedRun(
        config=config,
        config_path=Path("configs/transition1x.yaml").resolve(),
        processed_path=processed,
        bundle_dir=bundle,
        run_identity=_run_identity(),
        checkpoint_identity=_checkpoint_identity(),
        paths=paths,
        gate_root=tmp_path / "gates",
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=8,
    )


def test_stage_evidence_is_atomic_content_addressed_and_ignores_mtime(tmp_path: Path):
    stage = tmp_path / "stage"
    report = {"status": "pass", "value": 7}
    workflow.seal_stage(stage, stage="unit", fingerprint="a" * 64, report=report)
    manifest = workflow.load_stage(stage, stage="unit", fingerprint="a" * 64)
    assert manifest == report

    os.utime(stage / "manifest.json", (1, 1))
    assert workflow.load_stage(stage, stage="unit", fingerprint="a" * 64) == report
    workflow.seal_stage(stage, stage="unit", fingerprint="a" * 64, report=report)
    with pytest.raises(workflow.WorkflowError, match="drift"):
        workflow.seal_stage(
            stage,
            stage="unit",
            fingerprint="a" * 64,
            report={"status": "pass", "value": 8},
        )


def test_incomplete_stage_is_not_resume_evidence(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text("{}", encoding="utf-8")
    assert workflow.load_stage(stage, stage="unit", fingerprint="a" * 64) is None


def _pissa_resource_report(*, scope: str, swap_growth_kib: int) -> dict[str, object]:
    minimum = workflow.PISSA_MIN_MEM_AVAILABLE_KIB
    host = {
        "mem_available_kib": minimum + 1024,
        "gpus": ({"index": 0},),
        "compute_processes": (),
    }
    return {
        "policy": workflow.PISSA_RESOURCE_POLICY,
        "scope": scope,
        "minimum_mem_available_kib": minimum,
        "swap_growth_is_diagnostic": True,
        "other_gpu_phases_max_swap_growth_kib": 256 * 1024,
        "observed_swap_growth_kib": swap_growth_kib,
        "host_before": host,
        "host_after": {**host, "mem_available_kib": minimum},
    }


def test_prepare_pissa_hashes_once_and_rechecks_a_resumed_stage(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"artifact_type":"pissa"}\n', encoding="utf-8")
    cpu = SimpleNamespace(
        launch_fingerprint="f" * 64,
        config=SimpleNamespace(
            model=SimpleNamespace(cache_dir=tmp_path / "hub"),
            runtime=SimpleNamespace(
                local_cache_root=tmp_path / "cache",
                gpu_lock_path=tmp_path / "gpu.lock",
            ),
        ),
    )
    calls = 0

    def prepare(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["local_cache_root"] == tmp_path / "cache"
        scope = "full_serialization" if calls == 1 else "cached_payload_verification"
        return {
            "files_sha256": "a" * 64,
            "preparation_resources": _pissa_resource_report(
                scope=scope,
                swap_growth_kib=calls * 1024,
            ),
        }

    monkeypatch.setattr(workflow, "pissa_bundle_path", lambda cache: bundle)
    monkeypatch.setattr(workflow, "prepare_pissa_gate", prepare)

    first_bundle, first = workflow._prepare_pissa(cpu, tmp_path / "gates")
    second_bundle, second = workflow._prepare_pissa(cpu, tmp_path / "gates")

    assert first_bundle == second_bundle == bundle
    assert calls == 2
    assert second == first
    assert first["observed_swap_growth_kib"] == 1024
    assert first["preparation_resources"]["scope"] == "full_serialization"
    assert first["preparation_resources"]["host_before"]["gpus"] == [{"index": 0}]


def test_prepare_pissa_never_defaults_missing_resource_evidence(tmp_path: Path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    cpu = SimpleNamespace(
        launch_fingerprint="f" * 64,
        config=SimpleNamespace(
            model=SimpleNamespace(cache_dir=tmp_path / "hub"),
            runtime=SimpleNamespace(
                local_cache_root=tmp_path / "cache",
                gpu_lock_path=tmp_path / "gpu.lock",
            ),
        ),
    )
    monkeypatch.setattr(workflow, "pissa_bundle_path", lambda cache: bundle)
    monkeypatch.setattr(
        workflow,
        "prepare_pissa_gate",
        lambda **kwargs: {"files_sha256": "a" * 64},
    )

    with pytest.raises(workflow.WorkflowError, match="resource evidence"):
        workflow._prepare_pissa(cpu, tmp_path / "gates")


def test_torchrun_is_absolute_two_rank_project_executable():
    command = workflow.torchrun_command("janus_ts.gpu_gates", "--x", "1")
    assert Path(command[0]).is_absolute()
    assert command[1:5] == (
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "janus_ts.gpu_gates",
    )
    assert command[-2:] == ("--x", "1")


def test_delegated_failure_classification_and_gate_timeout(monkeypatch):
    calls: list[dict[str, object]] = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 1)

    outcome = workflow.run_delegated_command(
        "gate", ("/bin/false",), runner=runner, timeout_seconds=123
    )
    assert outcome.returncode == 1
    assert calls[0]["timeout"] == 123
    assert calls[0]["cwd"] == workflow.PROJECT_ROOT

    with pytest.raises(workflow.TransientWorkflowError, match="exit 75"):
        workflow.run_delegated_command(
            "gate",
            ("/bin/false",),
            runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 75),
        )

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(workflow.WorkflowError, match="explicit 10s"):
        workflow.run_delegated_command(
            "gate", ("/bin/false",), runner=timeout, timeout_seconds=10
        )


def test_locked_gpu_phase_checks_host_before_and_after(tmp_path: Path, monkeypatch):
    events: list[object] = []

    @contextmanager
    def lock(path):
        events.append(("lock", Path(path)))
        yield

    def gate(**kwargs):
        events.append(("gate", kwargs["required_gpu_count"]))
        return object()

    monkeypatch.setattr(workflow, "exclusive_lock", lock)
    monkeypatch.setattr(workflow, "assert_host_ready", gate)
    outcome = workflow.run_locked_gpu_command(
        "phase",
        ("/bin/true",),
        lock_path=tmp_path / "gpu.lock",
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )
    assert outcome.returncode == 0
    assert events[1] == ("gate", 2)
    assert events[2] == ("gate", 2)


def test_default_delegation_uses_an_owned_process_group():
    outcome = workflow.run_delegated_command(
        "owned", ("/bin/true",), timeout_seconds=5
    )
    assert outcome.returncode == 0


def test_owned_process_group_is_terminated_on_unexpected_wait_failure(monkeypatch):
    process_state = {"alive": True}
    launches: list[dict[str, object]] = []
    signals: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4242

        def wait(self, timeout=None):
            del timeout
            raise RuntimeError("unexpected wait failure")

        def poll(self):
            return None if process_state["alive"] else 0

        def kill(self):
            process_state["alive"] = False

    def popen(command, **kwargs):
        launches.append({"command": command, **kwargs})
        return FakeProcess()

    def killpg(process_group, signum):
        signals.append((process_group, signum))
        if signum == 0:
            if not process_state["alive"]:
                raise ProcessLookupError
            return
        assert process_group == 4242
        assert signum == signal.SIGTERM
        process_state["alive"] = False

    monkeypatch.setattr(workflow.subprocess, "Popen", popen)
    monkeypatch.setattr(workflow.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(workflow.os, "killpg", killpg)

    with pytest.raises(RuntimeError, match="unexpected wait failure"):
        workflow._run_owned_process_group(
            ("/bin/false",),
            cwd=workflow.PROJECT_ROOT,
            env=os.environ,
            timeout=None,
        )
    assert launches[0]["start_new_session"] is True
    assert (4242, signal.SIGTERM) in signals
    assert process_state["alive"] is False


def test_candidate_rejection_report_falls_back_even_if_torchrun_wraps_exit(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    binding = workflow._smoke_gate_binding(prepared, micro_batch_size=2)

    def delegated(phase, command, **kwargs):
        report = Path(command[command.index("--report") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "gate": "zero3-worst-case-2048-one-update",
                    **binding,
                    "status": "rejected",
                    "micro_batch_size_per_gpu": 2,
                    "sequence_length": 2048,
                }
            ),
            encoding="utf-8",
        )
        return workflow.CommandOutcome(phase, tuple(command), 1)

    monkeypatch.setattr(workflow, "run_locked_gpu_command", delegated)
    selected, evidence = workflow._run_memory_smoke(
        prepared,
        micro_batch_size=2,
        runner=lambda *args, **kwargs: None,
    )
    assert selected is False
    assert evidence["decision"] == "fallback-to-micro1"
    assert workflow.load_stage(
        prepared.gate_root / "05-smoke-candidate2",
        stage="smoke-candidate2",
        fingerprint=prepared.run_identity.fingerprint,
    ) == evidence


def test_unbound_candidate_failure_is_permanent(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)

    def delegated(phase, command, **kwargs):
        report = Path(command[command.index("--report") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        # Missing the frozen microbatch=2 identity, so this is not safe fallback evidence.
        report.write_text(
            json.dumps(
                {
                    "gate": "zero3-worst-case-2048-one-update",
                    "status": "rejected",
                }
            ),
            encoding="utf-8",
        )
        return workflow.CommandOutcome(phase, tuple(command), 1)

    monkeypatch.setattr(workflow, "run_locked_gpu_command", delegated)
    with pytest.raises(workflow.DelegatedProcessError):
        workflow._run_memory_smoke(
            prepared,
            micro_batch_size=2,
            runner=lambda *args, **kwargs: None,
        )


def test_passing_smoke_requires_two_rank_resource_evidence(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    binding = workflow._smoke_gate_binding(prepared, micro_batch_size=1)

    def delegated(phase, command, **kwargs):
        report = Path(command[command.index("--report") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "gate": "zero3-worst-case-2048-one-update",
                    **binding,
                    "status": "pass",
                    "world_size": 2,
                    "ranks": [
                        {
                            **binding,
                            "rank": rank,
                            "local_rank": rank,
                            "micro_batch_size_per_gpu": 1,
                            "gradient_accumulation_steps": 8,
                            "sequence_length": 2048,
                            "global_step": 1,
                            "training_loss": 1.0,
                            "peak_allocated_mib": 44000,
                            "peak_reserved_mib": 44001,
                            "device_memory_used_mib": 44002,
                            "swap_growth_kib": 0,
                        }
                        for rank in (0, 1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return workflow.CommandOutcome(phase, tuple(command), 0)

    monkeypatch.setattr(workflow, "run_locked_gpu_command", delegated)
    selected, evidence = workflow._run_memory_smoke(
        prepared,
        micro_batch_size=1,
        runner=lambda *args, **kwargs: None,
    )
    assert selected is True
    assert evidence["decision"] == "selected"


def test_passing_smoke_rejects_duplicate_rank_identity(tmp_path):
    prepared = _prepared(tmp_path)
    binding = workflow._smoke_gate_binding(prepared, micro_batch_size=1)
    rank = {
        **binding,
        "rank": 0,
        "local_rank": 0,
        "micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 8,
        "sequence_length": 2048,
        "global_step": 1,
        "training_loss": 1.0,
        "peak_allocated_mib": 44000,
        "peak_reserved_mib": 44001,
        "device_memory_used_mib": 44002,
        "swap_growth_kib": 0,
    }
    report = {
        "gate": "zero3-worst-case-2048-one-update",
        **binding,
        "status": "pass",
        "world_size": 2,
        "ranks": [rank, dict(rank)],
    }
    with pytest.raises(workflow.WorkflowError, match="rank set"):
        workflow._validate_smoke_pass(report, prepared, micro_batch_size=1)


def test_checkpoint_identity_binds_pissa_and_shares_run_fingerprint():
    run = _run_identity()
    first = workflow.derive_checkpoint_identity(
        run, pissa_manifest_sha256=run.pissa_manifest_sha256
    )
    second = workflow.derive_checkpoint_identity(run, pissa_manifest_sha256="q" * 64)
    assert first.run_fingerprint == run.fingerprint
    assert first.data_fingerprint == run.data_fingerprint
    assert first.model_fingerprint != second.model_fingerprint


def test_generation_smoke_requires_exact_beam512_and_two_bounded_ranks():
    config = load_config("configs/transition1x.yaml")
    shape = [10, 2200]
    receipt = {
        "status": "pass",
        "reaction_id": "rxn0001",
        "ordinal": 0,
        "prompt_sha256": "a" * 64,
        "prompt_tokens": 1800,
        "num_beams": 10,
        "num_return_sequences": 10,
        "max_new_tokens": 512,
        "output_shape": shape,
        "parsed_beams": 10,
        "valid_parses": 9,
        "swap_growth_kib": 0,
        "max_swap_growth_kib": 262144,
        "max_device_memory_mib": 45056,
        "stress": {
            "status": "pass",
            "purpose": "non-formal-worst-case-memory-only",
            "num_beams": 10,
            "num_return_sequences": 10,
            "min_new_tokens": 512,
            "max_new_tokens": 512,
            "actual_new_tokens": 512,
            "output_shape": [10, 2312],
        },
        "ranks": [
            {
                "rank": rank,
                "local_rank": rank,
                "peak_allocated_mib": 40000,
                "peak_reserved_mib": 41000,
                "device_memory_used_mib": 42000,
                "effective_peak_mib": 42000,
                "output_shape": shape,
                "parsed_beams": 10,
                "valid_parses": 9,
                "stress_output_shape": [10, 2312],
                "stress_actual_new_tokens": 512,
            }
            for rank in (0, 1)
        ],
    }
    workflow._validate_generation_smoke(receipt, config)
    receipt["max_new_tokens"] = 511
    with pytest.raises(workflow.WorkflowError, match="frozen decoding"):
        workflow._validate_generation_smoke(receipt, config)


def test_generation_smoke_rejects_511_token_stress():
    config = load_config("configs/transition1x.yaml")
    prompt_tokens = 1800
    receipt = {
        "status": "pass",
        "reaction_id": "rxn0001",
        "ordinal": 0,
        "prompt_sha256": "a" * 64,
        "prompt_tokens": prompt_tokens,
        "num_beams": 10,
        "num_return_sequences": 10,
        "max_new_tokens": 512,
        "output_shape": [10, 2200],
        "parsed_beams": 10,
        "valid_parses": 9,
        "swap_growth_kib": 0,
        "max_swap_growth_kib": 262144,
        "max_device_memory_mib": 45056,
        "stress": {
            "status": "pass",
            "purpose": "non-formal-worst-case-memory-only",
            "num_beams": 10,
            "num_return_sequences": 10,
            "min_new_tokens": 512,
            "max_new_tokens": 512,
            "actual_new_tokens": 511,
            "output_shape": [10, prompt_tokens + 511],
        },
        "ranks": [],
    }
    with pytest.raises(workflow.WorkflowError, match="full-length stress"):
        workflow._validate_generation_smoke(receipt, config)


def test_later_run_state_makes_selection_reentry_idempotent(tmp_path):
    prepared = _prepared(tmp_path)
    initialize_run(prepared.paths, prepared.run_identity)
    workflow._advance_run_state(prepared, stage="training", global_step=0)
    workflow._advance_run_state(prepared, stage="validation", global_step=500)
    workflow._advance_run_state(
        prepared,
        stage="selected",
        global_step=500,
        selected_epoch=3,
        selected_checkpoint_fingerprint="a" * 64,
    )
    workflow._advance_run_state(prepared, stage="testing", global_step=500)
    state = workflow._advance_run_state(
        prepared,
        stage="selected",
        global_step=500,
        selected_epoch=3,
        selected_checkpoint_fingerprint="a" * 64,
    )
    assert state["stage"] == "testing"


def test_epoch_enumeration_uses_manifest_epoch_and_step_not_mtime(tmp_path):
    prepared = _prepared(tmp_path)
    prepared.paths.durable_checkpoints.mkdir(parents=True)
    for epoch in range(5, 0, -1):
        checkpoint = prepared.paths.durable_checkpoints / f"arbitrary-{epoch}"
        (checkpoint / "portable_adapter").mkdir(parents=True)
        (checkpoint / "portable_adapter" / "adapter_model.safetensors").write_bytes(b"x")
        mark_complete(
            checkpoint,
            {
                "kind": "epoch",
                **prepared.checkpoint_identity.as_dict(),
                "epoch": float(epoch),
                "global_step": epoch * 100,
            },
        )
        os.utime(checkpoint, (100 - epoch, 100 - epoch))
    values = workflow.enumerate_epoch_checkpoints(prepared)
    assert [item.epoch for item in values] == [1, 2, 3, 4, 5]
    assert [item.global_step for item in values] == [100, 200, 300, 400, 500]


def test_complete_epoch_set_repairs_training_handoff_without_gpu(
    tmp_path, monkeypatch
):
    prepared = _prepared(tmp_path)
    initialize_run(prepared.paths, prepared.run_identity)
    prepared.paths.durable_checkpoints.mkdir(parents=True)
    for epoch in range(1, 6):
        checkpoint = prepared.paths.durable_checkpoints / f"epoch-{epoch}"
        (checkpoint / "portable_adapter").mkdir(parents=True)
        (checkpoint / "portable_adapter" / "adapter_model.safetensors").write_bytes(
            b"portable"
        )
        mark_complete(
            checkpoint,
            {
                "kind": "epoch",
                **prepared.checkpoint_identity.as_dict(),
                "epoch": float(epoch),
                "global_step": epoch * 100,
            },
        )

    def unexpected(*args, **kwargs):
        del args, kwargs
        pytest.fail("a complete epoch set must not select resume state or launch a GPU")

    monkeypatch.setattr(workflow, "select_resume_checkpoint", unexpected)
    monkeypatch.setattr(workflow, "run_locked_gpu_command", unexpected)
    checkpoints = workflow.run_training_phase(prepared)

    assert [item.epoch for item in checkpoints] == [1, 2, 3, 4, 5]
    state = json.loads(
        (prepared.paths.project / "state.json").read_text(encoding="utf-8")
    )
    assert state["stage"] == "validation"
    assert state["global_step"] == 500
    evidence = workflow.load_stage(
        prepared.paths.project / "workflow" / "training",
        stage="training",
        fingerprint=prepared.run_identity.fingerprint,
    )
    assert evidence is not None
    assert [item["epoch"] for item in evidence["epochs"]] == [1, 2, 3, 4, 5]


def test_evaluation_phase_is_idempotent_and_runs_locked_test_once(
    tmp_path, monkeypatch
):
    import janus_ts.evaluation as evaluation_module
    import janus_ts.formal_eval_runtime as formal_runtime

    prepared = _prepared(tmp_path)
    initialize_run(prepared.paths, prepared.run_identity)
    workflow._advance_run_state(prepared, stage="validation", global_step=500)
    checkpoints = tuple(
        workflow.EpochCheckpoint(
            epoch=epoch,
            global_step=epoch * 100,
            path=tmp_path / f"checkpoint-{epoch}",
            checkpoint_fingerprint=f"{epoch:064x}",
        )
        for epoch in range(1, 6)
    )
    by_path = {item.path.resolve(): item for item in checkpoints}
    phases: list[str] = []

    monkeypatch.setattr(workflow, "_run_portable_parity_gate", lambda *a, **k: {})

    def delegated(phase, command, **kwargs):
        del kwargs
        phases.append(phase)
        return workflow.CommandOutcome(phase, tuple(command), 0)

    monkeypatch.setattr(workflow, "run_locked_gpu_command", delegated)

    def receipt(_prepared, checkpoint, *, split):
        root = _prepared.paths.evaluations
        root.mkdir(parents=True, exist_ok=True)
        metrics = root / f"{split}-{checkpoint.epoch}.metrics.json"
        predictions = root / f"{split}-{checkpoint.epoch}.predictions.jsonl"
        metrics.write_text("{}\n", encoding="utf-8")
        predictions.write_text("{}\n", encoding="utf-8")
        return metrics, predictions

    monkeypatch.setattr(workflow, "_validate_formal_receipt", receipt)

    def inspect(checkpoint_dir, *args, **kwargs):
        del args, kwargs
        checkpoint = by_path[Path(checkpoint_dir).resolve()]
        return SimpleNamespace(
            generation_identity=lambda split: workflow._generation_identity(
                prepared, checkpoint, split
            )
        )

    monkeypatch.setattr(formal_runtime, "inspect_durable_checkpoint", inspect)
    monkeypatch.setattr(formal_runtime, "recover_completed_test", lambda *a, **k: None)

    scores = {
        item.path.resolve(): SimpleNamespace(
            checkpoint_id=item.checkpoint_fingerprint,
            epoch=item.epoch,
        )
        for item in checkpoints
    }
    monkeypatch.setattr(
        formal_runtime,
        "load_checkpoint_score",
        lambda checkpoint_dir, *args, **kwargs: scores[Path(checkpoint_dir).resolve()],
    )
    monkeypatch.setattr(
        evaluation_module,
        "select_best_checkpoint",
        lambda values: tuple(values)[2],
    )

    def comparison(_prepared, selected, predictions_path):
        del selected, predictions_path
        path = _prepared.paths.evaluations / "v24-comparison.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(workflow, "_write_v24_comparison", comparison)

    first = workflow.run_evaluation_phase(prepared, checkpoints)
    second = workflow.run_evaluation_phase(prepared, checkpoints)
    assert first == second == checkpoints[2]
    assert phases == [
        "formal-val-epoch-1",
        "formal-val-epoch-2",
        "formal-val-epoch-3",
        "formal-val-epoch-4",
        "formal-val-epoch-5",
        "formal-test-epoch-3",
    ]
    state = json.loads(
        (prepared.paths.project / "state.json").read_text(encoding="utf-8")
    )
    assert state["stage"] == "complete"
    assert state["selected_epoch"] == 3
    assert state["test_evaluated"] is True


def test_checkpoint_callback_factory_is_late_bound(tmp_path):
    holder: dict[str, object] = {}
    factory = workflow.make_checkpoint_callback_factory(
        _checkpoint_identity(),
        local_root=tmp_path / "local",
        durable_root=tmp_path / "durable",
        initial_adapter_dir=tmp_path / "pissa",
        checkpoint_steps=50,
        keep_local=2,
        expected_trainable_parameters=4,
        manager_holder=holder,  # type: ignore[arg-type]
    )
    callback = factory(lambda: object())
    assert callback.manager is holder["manager"]
    with pytest.raises(workflow.WorkflowError, match="more than once"):
        factory(lambda: object())


def test_rank_zero_jsonl_log_is_fsynced_and_restart_ordinals_are_explicit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RANK", "0")
    path = tmp_path / "logs" / "train.jsonl"
    state = SimpleNamespace(global_step=10, epoch=0.5, is_world_process_zero=True)
    first = workflow.RankZeroJsonlLogCallback(
        path,
        run_fingerprint="f" * 64,
        resume_checkpoint=None,
        log_steps=10,
    )
    first.on_log(None, state, object(), logs={"loss": 1.25})
    second = workflow.RankZeroJsonlLogCallback(
        path,
        run_fingerprint="f" * 64,
        resume_checkpoint=tmp_path / "checkpoint-10",
        log_steps=10,
    )
    second.on_log(None, state, object(), logs={"loss": 1.0})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["launch_ordinal"] for row in rows] == [1, 2]
    assert rows[0]["resume_checkpoint"] is None
    assert rows[1]["resume_checkpoint"].endswith("checkpoint-10")
    assert [row["global_step"] for row in rows] == [10, 10]


def test_rank_zero_jsonl_recovers_only_an_unterminated_final_record(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RANK", "0")
    path = tmp_path / "train.jsonl"
    complete = {
        "run_fingerprint": "f" * 64,
        "launch_ordinal": 1,
    }
    retained = (json.dumps(complete) + "\n").encode()
    path.write_bytes(retained + b'{"run_fingerprint":"partial')

    callback = workflow.RankZeroJsonlLogCallback(
        path,
        run_fingerprint="f" * 64,
        resume_checkpoint=None,
        log_steps=10,
    )
    assert callback.launch_ordinal == 2
    assert path.read_bytes() == retained


def test_rank_zero_jsonl_rejects_a_malformed_completed_record(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    path = tmp_path / "train.jsonl"
    path.write_bytes(b"{not-json}\n")
    with pytest.raises(workflow.WorkflowError, match="malformed"):
        workflow.RankZeroJsonlLogCallback(
            path,
            run_fingerprint="f" * 64,
            resume_checkpoint=None,
            log_steps=10,
        )


def test_nonzero_rank_never_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    path = tmp_path / "train.jsonl"
    callback = workflow.RankZeroJsonlLogCallback(
        path,
        run_fingerprint="f" * 64,
        resume_checkpoint=None,
        log_steps=10,
    )
    callback.on_log(
        None,
        SimpleNamespace(global_step=10, epoch=1.0, is_world_process_zero=False),
        object(),
        logs={"loss": 1.0},
    )
    assert not path.exists()


def test_proven_transient_walks_wrapped_errno_and_rejects_generic_error():
    wrapped = workflow.WorkflowError("wrapped")
    wrapped.__cause__ = OSError(116, "stale")
    assert workflow.is_proven_transient_exception(wrapped)
    assert workflow.is_proven_transient_exception(ResourceUnavailableError("busy"))
    assert not workflow.is_proven_transient_exception(RuntimeError("unknown"))


def test_train_smoke_typer_command_is_mockable_and_reports_selection(monkeypatch, tmp_path):
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(workflow, "prepare_smoke_workflow", lambda config: prepared)
    result = CliRunner().invoke(
        app,
        ["train", "smoke", "--config", "configs/transition1x.yaml"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_fingerprint"] == "f" * 64
    assert payload["micro_batch_size_per_gpu"] == 1


def test_train_smoke_maps_only_resource_failure_to_exit_75(monkeypatch):
    def busy(config):
        raise ResourceUnavailableError("MemAvailable is low")

    monkeypatch.setattr(workflow, "prepare_smoke_workflow", busy)
    result = CliRunner().invoke(
        app,
        ["train", "smoke", "--config", "configs/transition1x.yaml"],
    )
    assert result.exit_code == 75
    assert "TRANSIENT" in result.stderr
