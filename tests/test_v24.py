from __future__ import annotations

import json
from pathlib import Path

import pytest

import janus_ts.v24 as v24
from janus_ts.config import load_config
from janus_ts.preprocessing import expected_processed_path, load_processed_dataset
from janus_ts.v24 import (
    V24_PREDICTIONS_SHA256,
    V24ContractError,
    audit_v24_source,
    load_v24_predictions,
    parse_v24_prediction,
)


def _custom_rows() -> list[dict[str, object]]:
    beam = "<TS_BO>  </TS_BO>"
    rows = [
        {"rxn_id": f"rxn{index:04d}", "pred_k10_str": [beam] * 10}
        for index in range(1006)
    ]
    rows[267]["rxn_id"] = "rxn2679"
    rows[267]["pred_k10_str"] = [beam] * 9
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_v24_parser_accepts_historical_wire_format_and_canonicalizes():
    parsed = parse_v24_prediction(
        "<TS_BO> 1 3 0.5 ; 0 2 2 </TS_BO>", atom_count=4
    )
    assert parsed.valid
    assert [(edge.atom_i, edge.atom_j, edge.bond_order) for edge in parsed.edges] == [
        (0, 2, 2.0),
        (1, 3, 0.5),
    ]


def test_v24_parser_rejects_repairable_but_invalid_forms():
    assert not parse_v24_prediction("<TS_BO> 1 0 1 </TS_BO>", atom_count=2).valid
    assert not parse_v24_prediction("<TS_BO> 0 2 1 </TS_BO>", atom_count=2).valid
    assert not parse_v24_prediction("<TS_BO> 0 1 1; 0 1 2 </TS_BO>", atom_count=2).valid


def test_real_frozen_v24_source_records_the_unrepaired_nine_beam_anomaly():
    audit = audit_v24_source()
    assert audit["status"] == "pass"
    assert audit["frozen_default_source"] is True
    assert audit["sha256"] == V24_PREDICTIONS_SHA256
    assert audit["line_count"] == 1006
    assert audit["unique_reaction_ids"] == 1006
    assert audit["beam_count_histogram"] == {"9": 1, "10": 1005}
    assert audit["beam_count_anomalies"] == {"rxn2679": 9}
    assert audit["historical_anomaly_was_repaired"] is False
    assert audit["checkpoint"]["present"] is True
    assert json.loads(json.dumps(audit, allow_nan=False)) == audit

    rows = load_v24_predictions()
    assert len(rows["rxn2679"]["pred_k10_str"]) == 9


def test_real_processed_test_intersection_has_102_complete_v24_rows():
    config = load_config("configs/transition1x.yaml")
    processed = load_processed_dataset(expected_processed_path(config))
    current_test_ids = processed["test"]["reaction_id"]

    audit = audit_v24_source(current_test_ids=current_test_ids)
    intersection = audit["current_test_intersection"]
    assert intersection["count"] == 102
    assert len(intersection["reaction_ids"]) == 102
    assert intersection["all_rows_have_10_string_beams"] is True
    assert intersection["historical_anomaly_present"] is False
    assert "rxn2679" not in intersection["reaction_ids"]


def test_custom_source_preserves_structural_contract_without_default_sha(tmp_path: Path):
    path = tmp_path / "v24.jsonl"
    _write_rows(path, _custom_rows())

    audit = audit_v24_source(path)
    assert audit["frozen_default_source"] is False
    assert audit["expected_sha256"] is None
    assert audit["beam_count_histogram"] == {"9": 1, "10": 1005}
    assert audit["beam_count_anomalies"] == {"rxn2679": 9}
    assert audit["checkpoint"] == {
        "checked": False,
        "path": str(v24.V24_CHECKPOINT_PATH),
        "present": None,
    }


@pytest.mark.parametrize("fault", ["row_count", "duplicate", "non_string", "anomaly"])
def test_custom_source_contract_faults_fail_closed(tmp_path: Path, fault: str):
    rows = _custom_rows()
    if fault == "row_count":
        rows.pop()
    elif fault == "duplicate":
        rows[1]["rxn_id"] = rows[0]["rxn_id"]
    elif fault == "non_string":
        beams = list(rows[0]["pred_k10_str"])
        beams[0] = 42
        rows[0]["pred_k10_str"] = beams
    elif fault == "anomaly":
        rows[267]["pred_k10_str"] = ["<TS_BO>  </TS_BO>"] * 10
    path = tmp_path / f"{fault}.jsonl"
    _write_rows(path, rows)

    with pytest.raises(V24ContractError):
        audit_v24_source(path)


def test_path_selected_as_frozen_default_requires_the_pinned_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "pretend-default.jsonl"
    _write_rows(path, _custom_rows())
    monkeypatch.setattr(v24, "V24_PREDICTIONS_PATH", path)

    with pytest.raises(V24ContractError, match="prediction hash"):
        audit_v24_source()
