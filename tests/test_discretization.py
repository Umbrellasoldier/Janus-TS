from __future__ import annotations

import math

import pytest

from janus_ts.discretization import normalize_ts_edge_weight


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.0, 0.0),
        (0.099999, 0.0),
        (0.1, 0.5),
        (0.749999, 0.5),
        (0.75, 1.0),
        (1.25, 1.5),
        (1.75, 2.0),
        (2.25, 2.5),
        (2.75, 3.0),
    ],
)
def test_frozen_boundaries(value: float, expected: float) -> None:
    assert normalize_ts_edge_weight(value) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        normalize_ts_edge_weight(value)
