from __future__ import annotations

from fractions import Fraction

import pytest

from janus_ts.metrics import aggregate, oracle_at_k, score_candidate, selection_key
from janus_ts.parsing import parse_ts_edges
from janus_ts.schema import Edge


def test_empty_graph_and_terminal_im_end_are_valid() -> None:
    result = parse_ts_edges("  <TS_EDGES>\r\n</TS_EDGES>\r\n<|im_end|>  ", atom_count=2)
    assert result.valid
    assert result.edges == ()


@pytest.mark.parametrize(
    "text,code",
    [
        ("answer:\n<TS_EDGES>\n</TS_EDGES>", "wrapper"),
        ("<TS_EDGES>\na1 --[bo=1]-- a0\n</TS_EDGES>", "edge_orientation"),
        ("<TS_EDGES>\na0 --[bo=1.0]-- a1\n</TS_EDGES>", "line_syntax"),
        (
            "<TS_EDGES>\na0 --[bo=1]-- a2\na0 --[bo=1]-- a1\n</TS_EDGES>",
            "edge_order",
        ),
        ("<TS_EDGES>\na0 --[bo=1]-- a9\n</TS_EDGES>", "atom_id"),
    ],
)
def test_strict_parser_rejects_repairs(text: str, code: str) -> None:
    result = parse_ts_edges(text, atom_count=3)
    assert not result.valid
    assert result.error_code == code


def test_wrong_bond_order_on_same_pair_is_one_edit() -> None:
    predicted = parse_ts_edges(
        "<TS_EDGES>\na0 --[bo=2]-- a1\n</TS_EDGES>", atom_count=2
    )
    score = score_candidate(predicted, (Edge(0, 1, 1.0),), atom_count=2)
    assert score.connectivity == 1
    assert score.exact == 0
    assert score.edit_bond == 1
    assert score.edge_iou == 1
    assert score.edge_f1 == 1


def test_invalid_penalty_and_independent_oracle() -> None:
    invalid = parse_ts_edges("bad", atom_count=3)
    exact = parse_ts_edges(
        "<TS_EDGES>\na0 --[bo=1]-- a1\n</TS_EDGES>", atom_count=3
    )
    first = score_candidate(invalid, (Edge(0, 1, 1.0),), atom_count=3)
    second = score_candidate(exact, (Edge(0, 1, 1.0),), atom_count=3)
    assert first.edit_bond == 4
    oracle = oracle_at_k([first, second], 2)
    assert oracle.connectivity == oracle.exact == 1
    assert oracle.edit_bond == 0


def test_exact_fraction_aggregation_and_selection() -> None:
    gold = (Edge(0, 1, 1.0), Edge(1, 2, 1.0))
    p1 = parse_ts_edges(
        "<TS_EDGES>\na0 --[bo=1]-- a1\n</TS_EDGES>", atom_count=3
    )
    p2 = parse_ts_edges(
        "<TS_EDGES>\na0 --[bo=1]-- a1\na1 --[bo=1]-- a2\n</TS_EDGES>",
        atom_count=3,
    )
    result = aggregate(
        [score_candidate(p1, gold, atom_count=3), score_candidate(p2, gold, atom_count=3)]
    )
    assert result.edge_iou == Fraction(3, 4)
    assert result.edge_f1 == Fraction(5, 6)
    assert selection_key(result, 1.0) > selection_key(result, 2.0)
