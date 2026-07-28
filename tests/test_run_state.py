from __future__ import annotations

import json

import pytest

from janus_ts.artifacts import mark_complete
from janus_ts.config import load_config
from janus_ts.run_state import (
    RunIdentity,
    RunPaths,
    RunStateError,
    build_run_identity,
    initialize_run,
    update_run_state,
)


def identity() -> RunIdentity:
    return RunIdentity(
        fingerprint="f" * 64,
        experiment="experiment",
        config_sha256="c" * 64,
        data_fingerprint="d" * 64,
        pissa_manifest_sha256="p" * 64,
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=8,
    )


def test_run_state_is_monotonic(tmp_path):
    paths = RunPaths(project=tmp_path / "project", local=tmp_path / "local")
    item = identity()
    assert initialize_run(paths, item)["stage"] == "initialized"
    state = update_run_state(paths, item, stage="training", global_step=50)
    assert state["global_step"] == 50
    with pytest.raises(RunStateError, match="regress"):
        update_run_state(paths, item, stage="initialized", global_step=50)
    with pytest.raises(RunStateError, match="regress"):
        update_run_state(paths, item, stage="training", global_step=49)


def test_run_identity_drift_is_rejected(tmp_path):
    paths = RunPaths(project=tmp_path / "project", local=tmp_path / "local")
    paths.project.mkdir(parents=True)
    (paths.project / "identity.json").write_text(json.dumps({"wrong": True}), encoding="utf-8")
    with pytest.raises(RunStateError, match="identity drift"):
        initialize_run(paths, identity())


def test_run_identity_fingerprint_binds_selected_batch_geometry(tmp_path):
    config = load_config("configs/transition1x.yaml")
    processed = tmp_path / "processed"
    bundle = tmp_path / "bundle"
    mark_complete(processed, {"fingerprint": "d" * 64})
    mark_complete(bundle, {"artifact_type": "test-pissa"})

    default = build_run_identity(
        config,
        processed_data_dir=processed,
        pissa_bundle_dir=bundle,
        micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=8,
    )
    candidate = build_run_identity(
        config,
        processed_data_dir=processed,
        pissa_bundle_dir=bundle,
        micro_batch_size_per_gpu=2,
        gradient_accumulation_steps=4,
    )

    assert default.fingerprint != candidate.fingerprint
    assert (default.micro_batch_size_per_gpu, default.gradient_accumulation_steps) == (
        1,
        8,
    )
    assert (candidate.micro_batch_size_per_gpu, candidate.gradient_accumulation_steps) == (
        2,
        4,
    )
