"""Transition-state Wiberg bond-order discretization."""

from __future__ import annotations

import math

from .constants import MIN_TS_EDGE_WEIGHT


def normalize_ts_edge_weight(
    weight: float, *, min_edge_weight: float = MIN_TS_EDGE_WEIGHT
) -> float:
    """Apply the frozen GeoDiff ``utils/datasets.py:604`` rule.

    Boundaries are deliberately left-closed on the higher bin exactly as in
    the source implementation. Non-finite values are rejected rather than
    silently normalized.
    """

    value = float(weight)
    if not math.isfinite(value):
        raise ValueError(f"non-finite WBO: {weight!r}")
    if min_edge_weight < 0:
        raise ValueError("min_edge_weight must be non-negative")
    if value < min_edge_weight:
        return 0.0
    if value < 0.75:
        return 0.5
    if value < 1.25:
        return 1.0
    if value < 1.75:
        return 1.5
    if value < 2.25:
        return 2.0
    if value < 2.75:
        return 2.5
    return 3.0


def format_bond_order(value: float) -> str:
    """Render one of the six output bins without an unnecessary decimal."""

    number = float(value)
    if number not in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}:
        raise ValueError(f"unsupported bond-order bin: {value!r}")
    return str(int(number)) if number.is_integer() else str(number)
