"""Frozen, post-selection comparison with Chemformer/ReactionT5v2 v24."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .constants import ALLOWED_BOND_ORDERS, FORMAL_EVAL_K
from .evaluation import EvaluationReport, ReactionEvaluation, aggregate_evaluations
from .metrics import oracle_at_k, score_candidate
from .parsing import ParseResult
from .schema import Edge, ReactionRecord, canonical_edges

V24_CHECKPOINT_PATH = Path(
    "/mnt/sto3/caoxiangyu/HERMES-TS/var/models/reactiont5v2-tsbo-v24-ckpt413820"
)
V24_PREDICTIONS_PATH = Path(
    "/mnt/sto3/caoxiangyu/GeoDiff/training_data_0.25cutoff/"
    "v24_ts1x_pred_final_ckpt413820.jsonl"
)
V24_PREDICTIONS_SHA256 = "11e4eea54a1f2d5e45af003a3607fca75fc670fdb96b35ef7fa38b10cf34bc3c"
V24_EXPECTED_LINE_COUNT = 1006
V24_EXPECTED_BEAM_COUNT_HISTOGRAM = {9: 1, 10: 1005}
V24_EXPECTED_BEAM_ANOMALIES = {"rxn2679": 9}
EXPECTED_INTERSECTION = 102
_V24_WRAPPER = re.compile(r"^<TS_BO>(?: (.*))? </TS_BO>$")
_TRIPLET = re.compile(r"^(0|[1-9][0-9]*) (0|[1-9][0-9]*) (0\.5|1|1\.5|2|2\.5|3)$")


class V24ContractError(RuntimeError):
    """Frozen v24 asset or its common-test subset changed."""


def parse_v24_prediction(text: str, *, atom_count: int) -> ParseResult:
    """Parse the historical checkpoint's own exact `<TS_BO>` wire format."""

    if not isinstance(text, str):
        return ParseResult(False, (), "not_text", "v24 prediction is not text", repr(text))
    match = _V24_WRAPPER.fullmatch(text)
    if match is None:
        return ParseResult(False, (), "wrapper", "invalid v24 TS_BO wrapper", text)
    body = match.group(1)
    pieces = [] if body in (None, "") else body.split(" ; ")
    edges: list[Edge] = []
    seen: set[tuple[int, int]] = set()
    for piece in pieces:
        triplet = _TRIPLET.fullmatch(piece)
        if triplet is None:
            return ParseResult(False, (), "line_syntax", "invalid v24 triplet", text)
        atom_i, atom_j = int(triplet.group(1)), int(triplet.group(2))
        bond_order = float(triplet.group(3))
        if atom_i >= atom_j:
            return ParseResult(False, (), "edge_orientation", "v24 edge requires I<J", text)
        if atom_j >= atom_count:
            return ParseResult(False, (), "atom_id", "v24 edge atom is out of range", text)
        if bond_order not in ALLOWED_BOND_ORDERS:
            return ParseResult(False, (), "bond_order", "invalid v24 bond order", text)
        if (atom_i, atom_j) in seen:
            return ParseResult(False, (), "duplicate_edge", "duplicate v24 edge", text)
        seen.add((atom_i, atom_j))
        edges.append(Edge(atom_i, atom_j, bond_order))
    return ParseResult(True, canonical_edges(edges), normalized_text=text)


def _load_v24_source(
    path: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load one whole source without repairing its historical nine-beam row."""

    source = V24_PREDICTIONS_PATH if path is None else Path(path)
    is_frozen_default = source.resolve() == V24_PREDICTIONS_PATH.resolve()
    try:
        actual_hash = sha256_file(source)
    except OSError as exc:
        raise V24ContractError(f"cannot hash v24 source {source}: {exc}") from exc
    if is_frozen_default and actual_hash != V24_PREDICTIONS_SHA256:
        raise V24ContractError(
            f"v24 prediction hash={actual_hash}, expected {V24_PREDICTIONS_SHA256}"
        )

    rows: dict[str, dict[str, Any]] = {}
    beam_count_histogram: Counter[int] = Counter()
    line_count = 0
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line_count = line_number
                try:
                    row = json.loads(line)
                    reaction_id = row["rxn_id"]
                    beams = row["pred_k10_str"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise V24ContractError(
                        f"invalid v24 row {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise V24ContractError(f"v24 row {line_number} is not an object")
                if not isinstance(reaction_id, str) or not reaction_id:
                    raise V24ContractError(
                        f"invalid v24 reaction ID on row {line_number}: {reaction_id!r}"
                    )
                if reaction_id in rows:
                    raise V24ContractError(f"duplicate v24 reaction ID {reaction_id!r}")
                if not isinstance(beams, list):
                    raise V24ContractError(
                        f"v24 {reaction_id!r} pred_k10_str is not a list"
                    )
                if not all(isinstance(item, str) for item in beams):
                    raise V24ContractError(
                        f"v24 {reaction_id!r} contains a non-string raw beam"
                    )
                beam_count_histogram[len(beams)] += 1
                rows[reaction_id] = row
    except (OSError, UnicodeError) as exc:
        raise V24ContractError(f"cannot read v24 source {source}: {exc}") from exc

    if line_count != V24_EXPECTED_LINE_COUNT or len(rows) != V24_EXPECTED_LINE_COUNT:
        raise V24ContractError(
            "v24 predictions contain "
            f"{line_count} lines/{len(rows)} unique rows, "
            f"expected {V24_EXPECTED_LINE_COUNT}"
        )
    if dict(beam_count_histogram) != V24_EXPECTED_BEAM_COUNT_HISTOGRAM:
        raise V24ContractError(
            "v24 beam-count histogram changed: "
            f"got={dict(sorted(beam_count_histogram.items()))}, "
            f"expected={V24_EXPECTED_BEAM_COUNT_HISTOGRAM}"
        )
    beam_anomalies = {
        reaction_id: len(row["pred_k10_str"])
        for reaction_id, row in rows.items()
        if len(row["pred_k10_str"]) != 10
    }
    if beam_anomalies != V24_EXPECTED_BEAM_ANOMALIES:
        raise V24ContractError(
            "v24 beam anomaly changed: "
            f"got={beam_anomalies}, expected={V24_EXPECTED_BEAM_ANOMALIES}"
        )
    if is_frozen_default and not V24_CHECKPOINT_PATH.is_dir():
        raise V24ContractError(f"frozen v24 checkpoint is missing: {V24_CHECKPOINT_PATH}")

    audit: dict[str, Any] = {
        "schema_version": "janus-ts-v24-source-audit-v1",
        "status": "pass",
        "source_path": str(source.resolve()),
        "frozen_default_source": is_frozen_default,
        "sha256": actual_hash,
        "expected_sha256": V24_PREDICTIONS_SHA256 if is_frozen_default else None,
        "line_count": line_count,
        "unique_reaction_ids": len(rows),
        # JSON object keys are strings by definition; make that conversion
        # explicit so the in-memory value survives a JSON round trip exactly.
        "beam_count_histogram": {
            str(count): occurrences
            for count, occurrences in sorted(beam_count_histogram.items())
        },
        "beam_count_anomalies": dict(sorted(beam_anomalies.items())),
        "all_beams_are_strings": True,
        "historical_anomaly_was_repaired": False,
        "checkpoint": {
            "checked": is_frozen_default,
            "path": str(V24_CHECKPOINT_PATH),
            "present": V24_CHECKPOINT_PATH.is_dir() if is_frozen_default else None,
        },
    }
    return rows, audit


def load_v24_predictions(
    path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the complete frozen-format source and preserve its nine-beam anomaly."""

    rows, _ = _load_v24_source(path)
    return rows


def _validated_intersection(
    predictions: Mapping[str, Mapping[str, Any]],
    current_test_ids: Sequence[str],
) -> tuple[str, ...]:
    ids = tuple(current_test_ids)
    if not all(isinstance(reaction_id, str) and reaction_id for reaction_id in ids):
        raise V24ContractError("current test reaction IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise V24ContractError("current test reaction IDs are not unique")
    intersection = tuple(sorted(set(ids) & set(predictions)))
    if len(intersection) != EXPECTED_INTERSECTION:
        raise V24ContractError(
            f"v24/current test intersection={len(intersection)}, "
            f"expected {EXPECTED_INTERSECTION}"
        )
    anomaly_ids = set(V24_EXPECTED_BEAM_ANOMALIES) & set(intersection)
    if anomaly_ids:
        raise V24ContractError(
            "historical v24 beam anomaly unexpectedly entered the current intersection: "
            f"{sorted(anomaly_ids)!r}"
        )
    invalid = {
        reaction_id: len(predictions[reaction_id]["pred_k10_str"])
        for reaction_id in intersection
        if len(predictions[reaction_id]["pred_k10_str"]) != 10
    }
    if invalid:
        raise V24ContractError(f"v24 intersection rows lack 10 raw beams: {invalid!r}")
    return intersection


def audit_v24_source(
    path: str | Path | None = None,
    *,
    current_test_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return JSON-safe, fail-closed evidence for the CPU preflight.

    A custom path skips only the pinned default-path SHA/checkpoint identity;
    it still has to reproduce all 1,006 structural rows, including the single
    historical ``rxn2679`` nine-beam anomaly.
    """

    predictions, audit = _load_v24_source(path)
    if current_test_ids is not None:
        intersection = _validated_intersection(predictions, current_test_ids)
        audit["current_test_intersection"] = {
            "count": len(intersection),
            "reaction_ids": list(intersection),
            "all_rows_have_10_string_beams": True,
            "historical_anomaly_present": False,
        }
    return audit


def evaluate_v24_intersection(
    test_records: Mapping[str, ReactionRecord],
    *,
    predictions_path: str | Path = V24_PREDICTIONS_PATH,
) -> tuple[tuple[str, ...], EvaluationReport]:
    predictions = load_v24_predictions(predictions_path)
    intersection = _validated_intersection(predictions, tuple(test_records))
    evaluations: list[ReactionEvaluation] = []
    for reaction_id in intersection:
        record = test_records[reaction_id]
        beams = tuple(predictions[reaction_id]["pred_k10_str"])
        parses = tuple(
            parse_v24_prediction(text, atom_count=record.atom_count) for text in beams
        )
        candidates = tuple(
            score_candidate(parsed, record.ts_edges, atom_count=record.atom_count)
            for parsed in parses
        )
        evaluations.append(
            ReactionEvaluation(
                reaction_id=reaction_id,
                raw_beams=beams,
                parses=parses,
                candidates=candidates,
                oracle_metrics=tuple(
                    (k, oracle_at_k(candidates, k)) for k in FORMAL_EVAL_K
                ),
            )
        )
    return intersection, aggregate_evaluations(evaluations)


def subset_qwen_predictions(
    predictions: Mapping[str, Sequence[str]], intersection: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    missing = sorted(set(intersection) - set(predictions))
    if missing:
        raise V24ContractError(f"selected Qwen test predictions miss IDs: {missing[:8]!r}")
    result = {reaction_id: tuple(predictions[reaction_id]) for reaction_id in intersection}
    invalid = {reaction_id: len(beams) for reaction_id, beams in result.items() if len(beams) != 10}
    if invalid:
        raise V24ContractError(f"Qwen comparison rows lack 10 raw beams: {invalid!r}")
    return result


__all__ = [
    "EXPECTED_INTERSECTION",
    "V24_CHECKPOINT_PATH",
    "V24_EXPECTED_BEAM_ANOMALIES",
    "V24_EXPECTED_BEAM_COUNT_HISTOGRAM",
    "V24_EXPECTED_LINE_COUNT",
    "V24_PREDICTIONS_PATH",
    "V24_PREDICTIONS_SHA256",
    "V24ContractError",
    "audit_v24_source",
    "evaluate_v24_intersection",
    "load_v24_predictions",
    "parse_v24_prediction",
    "subset_qwen_predictions",
]
