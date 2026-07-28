"""Exact reaction-level graph metrics and deterministic aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .parsing import ParseResult
from .schema import Edge, canonical_edges


@dataclass(frozen=True)
class CandidateMetrics:
    valid: bool
    connectivity: int
    exact: int
    edit_bond: int
    edge_iou: Fraction
    edge_f1: Fraction


@dataclass(frozen=True)
class AggregateMetrics:
    count: int
    connectivity_successes: int
    exact_successes: int
    edit_bond_total: int
    edge_iou_sum: Fraction
    edge_f1_sum: Fraction

    @property
    def connectivity(self) -> Fraction:
        return Fraction(self.connectivity_successes, self.count)

    @property
    def exact(self) -> Fraction:
        return Fraction(self.exact_successes, self.count)

    @property
    def edit_bond(self) -> Fraction:
        return Fraction(self.edit_bond_total, self.count)

    @property
    def edge_iou(self) -> Fraction:
        return self.edge_iou_sum / self.count

    @property
    def edge_f1(self) -> Fraction:
        return self.edge_f1_sum / self.count


def _edge_map(edges: Iterable[Edge]) -> dict[tuple[int, int], float]:
    canonical = canonical_edges(edges)
    return {edge.pair: edge.bond_order for edge in canonical}


def score_candidate(
    prediction: ParseResult,
    gold_edges: Iterable[Edge],
    *,
    atom_count: int,
) -> CandidateMetrics:
    """Score one candidate; a wrong BO on an existing pair is one edit."""

    gold = _edge_map(gold_edges)
    if not prediction.valid:
        return CandidateMetrics(
            valid=False,
            connectivity=0,
            exact=0,
            edit_bond=atom_count * (atom_count - 1) // 2 + 1,
            edge_iou=Fraction(0),
            edge_f1=Fraction(0),
        )

    predicted = _edge_map(prediction.edges)
    gold_pairs = set(gold)
    predicted_pairs = set(predicted)
    intersection = len(gold_pairs & predicted_pairs)
    union = len(gold_pairs | predicted_pairs)
    connectivity = int(predicted_pairs == gold_pairs)
    exact = int(predicted == gold)
    edit_bond = sum(
        predicted.get(pair, 0.0) != gold.get(pair, 0.0)
        for pair in predicted_pairs | gold_pairs
    )
    edge_iou = Fraction(intersection, union) if union else Fraction(1)
    denominator = len(gold_pairs) + len(predicted_pairs)
    edge_f1 = Fraction(2 * intersection, denominator) if denominator else Fraction(1)
    return CandidateMetrics(
        valid=True,
        connectivity=connectivity,
        exact=exact,
        edit_bond=edit_bond,
        edge_iou=edge_iou,
        edge_f1=edge_f1,
    )


def oracle_at_k(candidates: Sequence[CandidateMetrics], k: int) -> CandidateMetrics:
    """Take each metric's independent oracle over the first ``k`` raw beams."""

    if k <= 0:
        raise ValueError("k must be positive")
    prefix = candidates[:k]
    if len(prefix) != k:
        raise ValueError(f"expected at least {k} candidates, got {len(candidates)}")
    return CandidateMetrics(
        valid=any(item.valid for item in prefix),
        connectivity=max(item.connectivity for item in prefix),
        exact=max(item.exact for item in prefix),
        edit_bond=min(item.edit_bond for item in prefix),
        edge_iou=max(item.edge_iou for item in prefix),
        edge_f1=max(item.edge_f1 for item in prefix),
    )


def aggregate(metrics: Iterable[CandidateMetrics]) -> AggregateMetrics:
    values = tuple(metrics)
    if not values:
        raise ValueError("cannot aggregate an empty metric sequence")
    return AggregateMetrics(
        count=len(values),
        connectivity_successes=sum(item.connectivity for item in values),
        exact_successes=sum(item.exact for item in values),
        edit_bond_total=sum(item.edit_bond for item in values),
        edge_iou_sum=sum((item.edge_iou for item in values), start=Fraction(0)),
        edge_f1_sum=sum((item.edge_f1 for item in values), start=Fraction(0)),
    )


def selection_key(metrics_at_10: AggregateMetrics, eval_loss: float) -> tuple[object, ...]:
    """Lexicographic checkpoint key; larger tuples are better."""

    return (
        metrics_at_10.connectivity,
        metrics_at_10.exact,
        -metrics_at_10.edit_bond,
        metrics_at_10.edge_iou,
        metrics_at_10.edge_f1,
        -float(eval_loss),
    )
