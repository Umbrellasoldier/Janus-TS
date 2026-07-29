from __future__ import annotations

import json
from pathlib import Path

from janus_ts.artifacts import mark_complete, sha256_file, write_json
from janus_ts.config import load_config
from janus_ts.schema import Atom, Edge, MolecularState, ReactionRecord
from janus_ts.thinking_evaluation import (
    THINKING_RUNTIME_RECEIPT_SCHEMA_VERSION,
    THINKING_TEST_REPORT_K,
    THINKING_TEST_SAMPLE_COUNT,
    ThinkingModelBinding,
    build_thinking_binding,
    extract_thinking_answer,
    finalize_thinking_scores,
    thinking_receipt_path,
    validate_thinking_completion,
)
from janus_ts.thinking_inference import ThinkingExplorationReceipt


def _record(reaction_id: str) -> ReactionRecord:
    atoms = (Atom(0, 6, "C"), Atom(1, 8, "O"))
    state = MolecularState(
        atoms=atoms,
        edges=(Edge(0, 1, 1.0),),
        components=((0, 1),),
    )
    return ReactionRecord(
        reaction_id=reaction_id,
        reactant=state,
        product=state,
        ts_edges=(Edge(0, 1, 1.5),),
        split="test",
    )


def _small_config():
    config = load_config("configs/transition1x.yaml")
    counts = {**config.data.expected_retained_counts, "test": 2}
    data = config.data.model_copy(update={"expected_retained_counts": counts})
    return config.model_copy(update={"data": data})


def _raw_receipt(tmp_path: Path) -> ThinkingExplorationReceipt:
    root = tmp_path / "raw"
    root.mkdir()
    correct = "reasoning</think>\n\n<TS_EDGES>\na0 --[bo=1.5]-- a1\n</TS_EDGES><|im_end|>"
    rows = [
        {
            "ordinal": 0,
            "reaction_id": "rxn0001",
            "sample_count": THINKING_TEST_SAMPLE_COUNT,
            "raw_responses": [correct, *(["unfinished reasoning"] * 9)],
        },
        {
            "ordinal": 1,
            "reaction_id": "rxn0002",
            "sample_count": THINKING_TEST_SAMPLE_COUNT,
            "raw_responses": ["unfinished reasoning", correct, *(["unfinished reasoning"] * 8)],
        },
    ]
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return ThinkingExplorationReceipt(
        path=root,
        payload={
            "request_sha256": "a" * 64,
            "selection_count": 2,
            "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
        },
    )


def test_extract_thinking_answer_requires_one_closed_reasoning_block() -> None:
    answer, error = extract_thinking_answer("work\n</think>\n\n<TS_EDGES>\n</TS_EDGES><|im_end|>")
    assert error is None
    assert answer == "<TS_EDGES>\n</TS_EDGES><|im_end|>"
    assert extract_thinking_answer("work")[1] == "missing_closing_think"
    assert extract_thinking_answer("a</think>b</think>c")[1] == "multiple_closing_think"


def test_thinking_scores_use_only_final_answers_and_report_all_k(tmp_path: Path) -> None:
    config = _small_config()
    binding = ThinkingModelBinding(
        role="zero-shot",
        data_fingerprint="d" * 64,
        run_fingerprint="r" * 64,
        model_fingerprint="m" * 64,
        checkpoint_fingerprint="c" * 64,
        epoch=0,
        global_step=0,
        checkpoint_path=None,
    )
    root = finalize_thinking_scores(
        config,
        tmp_path,
        _raw_receipt(tmp_path),
        binding,
        (_record("rxn0001"), _record("rxn0002")),
    )
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["sample_count_per_reaction"] == THINKING_TEST_SAMPLE_COUNT
    assert metrics["report_k"] == list(THINKING_TEST_REPORT_K)
    assert set(metrics["evaluation"]["metrics"]) == {
        "@1",
        "@2",
        "@3",
        "@4",
        "@5",
        "@10",
    }
    assert metrics["evaluation"]["metrics"]["@1"]["count"] == 2
    assert metrics["evaluation"]["metrics"]["@1"]["exact"]["successes"] == 1
    assert metrics["evaluation"]["metrics"]["@2"]["exact"]["successes"] == 2
    assert metrics["evaluation"]["metrics"]["@10"]["exact"]["successes"] == 2
    assert metrics["extraction_errors"] == {"missing_closing_think": 18}
    assert metrics["parse_valid_candidate_count"] == 2


def test_zero_shot_binding_and_runtime_receipt_are_content_verified(tmp_path: Path) -> None:
    config = _small_config()
    processed = tmp_path / "processed"
    mark_complete(processed, {"fingerprint": "d" * 64})
    binding, _ = build_thinking_binding(
        config,
        processed,
        run_fingerprint="f" * 64,
        role="zero-shot",
    )
    raw = _raw_receipt(tmp_path)
    raw_manifest = {
        "schema_version": "janus-ts-thinking-exploration-v2",
        "artifact_class": "exploratory-non-formal",
        "formal_eligible": False,
        "affects_checkpoint_selection": False,
        "metrics_computed": False,
        "writes_formal_test_lease": False,
        "writes_run_state": False,
        "world_size": 2,
        "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
        "request_sha256": "a" * 64,
        "payload_inventory": {
            "rank-00000-of-00002.jsonl": {},
            "rank-00001-of-00002.jsonl": {},
            "predictions.jsonl": {},
        },
    }
    # The dedicated artifact validator is already covered by
    # test_thinking_inference; this test exercises the score and runtime links.
    score = finalize_thinking_scores(
        config,
        tmp_path,
        raw,
        binding,
        (_record("rxn0001"), _record("rxn0002")),
    )
    for name in ("rank-00000-of-00002.jsonl", "rank-00001-of-00002.jsonl"):
        (raw.path / name).write_text("{}\n", encoding="utf-8")
    raw_manifest["payload_inventory"] = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (
            raw.path / "rank-00000-of-00002.jsonl",
            raw.path / "rank-00001-of-00002.jsonl",
            raw.path / "predictions.jsonl",
        )
    }
    mark_complete(raw.path, raw_manifest)
    predictions = score / "scored-predictions.jsonl"
    metrics = score / "metrics.json"
    payload = {
        "schema_version": THINKING_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        **binding.to_json_dict(),
        "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
        "report_k": list(THINKING_TEST_REPORT_K),
        "request_sha256": "a" * 64,
        "raw_artifact_path": str(raw.path.resolve()),
        "raw_predictions_sha256": sha256_file(raw.path / "predictions.jsonl"),
        "score_artifact_path": str(score.resolve()),
        "scored_predictions": {
            "path": str(predictions.resolve()),
            "sha256": sha256_file(predictions),
        },
        "metrics": {"path": str(metrics.resolve()), "sha256": sha256_file(metrics)},
    }
    write_json(thinking_receipt_path(tmp_path, binding), payload)
    assert validate_thinking_completion(tmp_path, binding) is not None
