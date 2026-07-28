from __future__ import annotations

import json

import pytest

from janus_ts.artifacts import (
    ArtifactError,
    atomic_directory,
    mark_complete,
    newest_complete_checkpoint,
    read_complete_manifest,
)


def test_complete_manifest_and_atomic_directory(tmp_path):
    target = tmp_path / "artifact"
    with atomic_directory(target) as building:
        (building / "payload.txt").write_text("done", encoding="utf-8")
        mark_complete(building, {"run_fingerprint": "abc", "global_step": 3})

    assert read_complete_manifest(target)["global_step"] == 3
    assert (target / "payload.txt").read_text(encoding="utf-8") == "done"
    with pytest.raises(FileExistsError), atomic_directory(target):
        pass


def test_manifest_tamper_is_rejected(tmp_path):
    target = tmp_path / "artifact"
    target.mkdir()
    mark_complete(target, {"global_step": 1})
    (target / "manifest.json").write_text(json.dumps({"global_step": 2}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        read_complete_manifest(target)


def test_resume_selection_uses_step_not_mtime_or_name(tmp_path):
    root = tmp_path / "checkpoints"
    for name, step, fingerprint in (
        ("looks-new", 2, "run"),
        ("looks-old", 50, "run"),
        ("foreign", 100, "other"),
    ):
        path = root / name
        path.mkdir(parents=True)
        mark_complete(path, {"run_fingerprint": fingerprint, "global_step": step})

    selected = newest_complete_checkpoint([root], required_fingerprint="run")
    assert selected is not None
    assert selected[0].name == "looks-old"
    assert selected[1]["global_step"] == 50
