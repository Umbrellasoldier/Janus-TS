from __future__ import annotations

from fractions import Fraction

import pytest

from janus_ts.evaluation import (
    CheckpointScore,
    aggregate_evaluations,
    compare_checkpoints,
    evaluate_reaction,
    select_best_checkpoint,
    wilson_interval,
)
from janus_ts.metrics import AggregateMetrics
from janus_ts.schema import Edge

EMPTY = "<TS_EDGES>\n</TS_EDGES>"
ONE = "<TS_EDGES>\na0 --[bo=1]-- a1\n</TS_EDGES>"
ONE_WRONG_BO = "<TS_EDGES>\na0 --[bo=2]-- a1\n</TS_EDGES>"
TWO = (
    "<TS_EDGES>\n"
    "a0 --[bo=1]-- a1\n"
    "a1 --[bo=1]-- a2\n"
    "</TS_EDGES>"
)


def _aggregate(
    *,
    connectivity: int = 1,
    exact: int = 1,
    edit: int = 0,
    iou: Fraction = Fraction(1),
    f1: Fraction = Fraction(1),
) -> AggregateMetrics:
    return AggregateMetrics(
        count=2,
        connectivity_successes=connectivity,
        exact_successes=exact,
        edit_bond_total=edit,
        edge_iou_sum=2 * iou,
        edge_f1_sum=2 * f1,
    )


def test_raw_duplicate_beams_are_retained_and_oracles_are_independent() -> None:
    beams = ["invalid", ONE_WRONG_BO, ONE, ONE, EMPTY, TWO, EMPTY, EMPTY, EMPTY, EMPTY]
    result = evaluate_reaction(
        "rxn1",
        beams,
        (Edge(0, 1, 1.0),),
        atom_count=3,
    )
    assert result.raw_beams == tuple(beams)
    assert len(result.parses) == len(result.candidates) == 10
    assert result.parses[2].normalized_text == result.parses[3].normalized_text
    assert result.at_k(1).edit_bond == 4  # n(n-1)/2 + 1
    assert result.at_k(2).connectivity == 1
    assert result.at_k(2).exact == 0
    assert result.at_k(2).edit_bond == 1
    assert result.at_k(3).exact == 1


def test_exact_aggregate_json_display_and_deterministic_intervals() -> None:
    first = evaluate_reaction(
        "rxn1",
        [ONE, *([EMPTY] * 9)],
        (Edge(0, 1, 1.0),),
        atom_count=2,
    )
    second = evaluate_reaction(
        "rxn2",
        [EMPTY, ONE, *([EMPTY] * 8)],
        (Edge(0, 1, 1.0),),
        atom_count=2,
    )
    report = aggregate_evaluations((first, second))
    repeated = aggregate_evaluations((first, second))

    at_1 = report.at_k(1)
    assert at_1.metrics.connectivity == Fraction(1, 2)
    assert at_1.metrics.exact == Fraction(1, 2)
    assert at_1.metrics.edit_bond == Fraction(1, 2)
    assert at_1.edit_bond_p50 == 0.5
    assert at_1.edit_bond_p95 == pytest.approx(0.95)
    assert at_1.edit_bond_interval == repeated.at_k(1).edit_bond_interval
    payload = report.to_json_dict()
    assert payload["metrics"]["@1"]["connectivity"]["numerator"] == 1
    assert payload["metrics"]["@1"]["connectivity"]["denominator"] == 2
    assert report.display_rows()[0]["Connectivity"] == "50.00%"
    assert "0.5" in report.to_json()


def test_wilson_bounds_cover_observed_rate() -> None:
    interval = wilson_interval(5, 10)
    assert interval.low < 0.5 < interval.high
    assert interval.method == "Wilson"
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_checkpoint_priority_loss_tolerance_and_earlier_epoch() -> None:
    base = _aggregate()
    better_connectivity = CheckpointScore(
        "connectivity", 5, _aggregate(connectivity=2, exact=0), 99.0
    )
    better_exact = CheckpointScore("exact", 1, base, 1.0)
    assert compare_checkpoints(better_connectivity, better_exact) > 0

    early = CheckpointScore("early", 1, base, 1.0 + 1e-8)
    late = CheckpointScore("late", 2, base, 1.0)
    assert compare_checkpoints(early, late) > 0

    materially_lower_loss = CheckpointScore("lower-loss", 3, base, 0.99999998)
    assert compare_checkpoints(materially_lower_loss, early) > 0
    assert select_best_checkpoint((late, early, materially_lower_loss)) == materially_lower_loss


def test_formal_evaluation_requires_exactly_ten_raw_beams() -> None:
    with pytest.raises(ValueError, match="exactly 10 raw beams"):
        evaluate_reaction("rxn1", [EMPTY] * 9, (), atom_count=1)
