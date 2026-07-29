"""Dataset-level evaluation, uncertainty estimates, and checkpoint selection.

The exact graph totals in this module remain integers or
:class:`fractions.Fraction` objects.  Floating-point conversion is restricted
to confidence intervals, percentile summaries, and JSON display values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from statistics import NormalDist

import numpy as np

from .constants import FORMAL_EVAL_K, SEED
from .metrics import (
    AggregateMetrics,
    CandidateMetrics,
    aggregate,
    oracle_at_k,
    score_candidate,
)
from .parsing import ParseResult, parse_ts_edges
from .schema import Edge

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
LOSS_TIE_TOLERANCE = 1e-8
_BOOTSTRAP_BATCH_SIZE = 256
_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    confidence_level: float
    method: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError("confidence interval endpoints must be finite")
        if self.low > self.high:
            raise ValueError("confidence interval low endpoint exceeds high endpoint")

    def to_json_dict(self) -> dict[str, float | str]:
        return {
            "low": self.low,
            "high": self.high,
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass(frozen=True)
class ReactionEvaluation:
    """All ten raw candidates and independent oracle scores for one reaction."""

    reaction_id: str
    raw_beams: tuple[str, ...]
    parses: tuple[ParseResult, ...]
    candidates: tuple[CandidateMetrics, ...]
    oracle_metrics: tuple[tuple[int, CandidateMetrics], ...]

    def at_k(self, k: int) -> CandidateMetrics:
        for stored_k, value in self.oracle_metrics:
            if stored_k == k:
                return value
        raise KeyError(f"reaction {self.reaction_id!r} has no @{k} result")


@dataclass(frozen=True)
class KSummary:
    k: int
    metrics: AggregateMetrics
    connectivity_interval: ConfidenceInterval
    exact_interval: ConfidenceInterval
    edit_bond_interval: ConfidenceInterval
    edge_iou_interval: ConfidenceInterval
    edge_f1_interval: ConfidenceInterval
    edit_bond_p50: float
    edit_bond_p95: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "count": self.metrics.count,
            "connectivity": _success_metric_json(
                self.metrics.connectivity_successes,
                self.metrics.count,
                self.connectivity_interval,
            ),
            "exact": _success_metric_json(
                self.metrics.exact_successes,
                self.metrics.count,
                self.exact_interval,
            ),
            "edit_bond": {
                **_fraction_json(self.metrics.edit_bond),
                "total": self.metrics.edit_bond_total,
                "interval": self.edit_bond_interval.to_json_dict(),
                "p50": self.edit_bond_p50,
                "p95": self.edit_bond_p95,
            },
            "edge_iou": {
                **_fraction_json(self.metrics.edge_iou),
                "sum": _fraction_json(self.metrics.edge_iou_sum),
                "interval": self.edge_iou_interval.to_json_dict(),
            },
            "edge_f1": {
                **_fraction_json(self.metrics.edge_f1),
                "sum": _fraction_json(self.metrics.edge_f1_sum),
                "interval": self.edge_f1_interval.to_json_dict(),
            },
        }


@dataclass(frozen=True)
class EvaluationReport:
    reaction_count: int
    summaries: tuple[KSummary, ...]
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = SEED

    def at_k(self, k: int) -> KSummary:
        for summary in self.summaries:
            if summary.k == k:
                return summary
        raise KeyError(f"report has no @{k} result")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "reaction_count": self.reaction_count,
            "bootstrap": {
                "method": "BCa reaction resampling",
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
            },
            "metrics": {f"@{item.k}": item.to_json_dict() for item in self.summaries},
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize without rounding any stored numeric value."""

        return json.dumps(
            self.to_json_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    def display_rows(self) -> tuple[dict[str, str], ...]:
        """Return human-facing values; rates are percentages with two decimals."""

        return tuple(
            {
                "k": f"@{item.k}",
                "Connectivity": _format_percent(item.metrics.connectivity),
                "Exact": _format_percent(item.metrics.exact),
                "edit bond": f"{float(item.metrics.edit_bond):.2f}",
                "IoU": _format_percent(item.metrics.edge_iou),
                "F1": _format_percent(item.metrics.edge_f1),
            }
            for item in self.summaries
        )


@dataclass(frozen=True)
class CheckpointScore:
    checkpoint_id: str
    epoch: float
    metrics_at_10: AggregateMetrics
    eval_loss: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.epoch) or self.epoch < 0:
            raise ValueError("epoch must be finite and non-negative")
        if not math.isfinite(self.eval_loss):
            raise ValueError("eval_loss must be finite")


def evaluate_reaction(
    reaction_id: str,
    raw_beams: Sequence[str],
    gold_edges: Iterable[Edge],
    *,
    atom_count: int,
    report_k: Sequence[int] = FORMAL_EVAL_K,
) -> ReactionEvaluation:
    """Parse and score raw beams in rank order, without deduplication."""

    ks = _validate_report_k(report_k)
    if len(raw_beams) != max(ks):
        raise ValueError(
            f"formal evaluation requires exactly {max(ks)} raw beams; "
            f"got {len(raw_beams)} for {reaction_id!r}"
        )
    gold = tuple(gold_edges)
    parses = tuple(parse_ts_edges(text, atom_count=atom_count) for text in raw_beams)
    candidates = tuple(
        score_candidate(parsed, gold, atom_count=atom_count) for parsed in parses
    )
    return ReactionEvaluation(
        reaction_id=reaction_id,
        raw_beams=tuple(raw_beams),
        parses=parses,
        candidates=candidates,
        oracle_metrics=tuple((k, oracle_at_k(candidates, k)) for k in ks),
    )


def aggregate_evaluations(
    evaluations: Iterable[ReactionEvaluation],
    *,
    report_k: Sequence[int] = FORMAL_EVAL_K,
) -> EvaluationReport:
    """Aggregate exact macro metrics and frozen uncertainty summaries."""

    reactions = tuple(evaluations)
    if not reactions:
        raise ValueError("cannot evaluate an empty reaction sequence")
    identifiers = [item.reaction_id for item in reactions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("reaction IDs must be unique in an evaluation report")
    ks = _validate_report_k(report_k)

    values_by_k = {k: tuple(item.at_k(k) for item in reactions) for k in ks}
    aggregates = {k: aggregate(values) for k, values in values_by_k.items()}

    # One frozen stream of reaction-resampling weights is shared by all @k and
    # auxiliary metrics.  This both records the intended resampling unit and
    # avoids accidental metric-specific random seeds.
    column_names: list[tuple[int, str]] = []
    columns: list[list[float]] = []
    for k in ks:
        values = values_by_k[k]
        for name, column in (
            ("edit_bond", [float(item.edit_bond) for item in values]),
            ("edge_iou", [float(item.edge_iou) for item in values]),
            ("edge_f1", [float(item.edge_f1) for item in values]),
        ):
            column_names.append((k, name))
            columns.append(column)
    sample_matrix = np.asarray(columns, dtype=np.float64).T
    bca_intervals = _bca_mean_intervals(sample_matrix)
    interval_by_name = dict(zip(column_names, bca_intervals, strict=True))

    summaries: list[KSummary] = []
    for k in ks:
        metric = aggregates[k]
        edit_values = np.asarray(
            [item.edit_bond for item in values_by_k[k]], dtype=np.float64
        )
        summaries.append(
            KSummary(
                k=k,
                metrics=metric,
                connectivity_interval=wilson_interval(
                    metric.connectivity_successes, metric.count
                ),
                exact_interval=wilson_interval(metric.exact_successes, metric.count),
                edit_bond_interval=interval_by_name[(k, "edit_bond")],
                edge_iou_interval=interval_by_name[(k, "edge_iou")],
                edge_f1_interval=interval_by_name[(k, "edge_f1")],
                edit_bond_p50=float(np.quantile(edit_values, 0.50, method="linear")),
                edit_bond_p95=float(np.quantile(edit_values, 0.95, method="linear")),
            )
        )
    return EvaluationReport(reaction_count=len(reactions), summaries=tuple(summaries))


def wilson_interval(
    successes: int,
    count: int,
    *,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> ConfidenceInterval:
    """Two-sided Wilson score interval for a binomial success rate."""

    if count <= 0:
        raise ValueError("count must be positive")
    if successes < 0 or successes > count:
        raise ValueError("successes must lie between zero and count")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    alpha = 1.0 - confidence_level
    z = _STANDARD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / count
    z_squared = z * z
    denominator = 1.0 + z_squared / count
    center = (proportion + z_squared / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z_squared / (4.0 * count * count)
        )
        / denominator
    )
    return ConfidenceInterval(
        low=max(0.0, center - half_width),
        high=min(1.0, center + half_width),
        confidence_level=confidence_level,
        method="Wilson",
    )


def compare_checkpoints(
    left: CheckpointScore,
    right: CheckpointScore,
    *,
    loss_tolerance: float = LOSS_TIE_TOLERANCE,
) -> int:
    """Return positive when ``left`` is preferred by the frozen comparator."""

    if loss_tolerance < 0.0:
        raise ValueError("loss_tolerance must be non-negative")
    left_metrics = left.metrics_at_10
    right_metrics = right.metrics_at_10
    comparisons = (
        _compare(left_metrics.connectivity, right_metrics.connectivity),
        _compare(left_metrics.exact, right_metrics.exact),
        _compare(right_metrics.edit_bond, left_metrics.edit_bond),
        _compare(left_metrics.edge_iou, right_metrics.edge_iou),
        _compare(left_metrics.edge_f1, right_metrics.edge_f1),
    )
    for result in comparisons:
        if result:
            return result

    loss_difference = left.eval_loss - right.eval_loss
    if abs(loss_difference) > loss_tolerance:
        return 1 if loss_difference < 0.0 else -1
    if left.epoch != right.epoch:
        return 1 if left.epoch < right.epoch else -1
    return 0


def select_best_checkpoint(
    checkpoints: Iterable[CheckpointScore],
) -> CheckpointScore:
    values = tuple(checkpoints)
    if not values:
        raise ValueError("cannot select from an empty checkpoint sequence")
    return sorted(values, key=cmp_to_key(compare_checkpoints), reverse=True)[0]


def _bca_mean_intervals(samples: np.ndarray) -> tuple[ConfidenceInterval, ...]:
    """BCa intervals for columns of reaction-level observations."""

    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("samples must be a non-empty two-dimensional array")
    if not np.isfinite(samples).all():
        raise ValueError("BCa samples must be finite")
    reaction_count, metric_count = samples.shape
    observed = samples.mean(axis=0)

    if reaction_count == 1 or np.all(samples == samples[0], axis=0).all():
        return tuple(_degenerate_bca(value) for value in observed)

    bootstrap_means = np.empty((BOOTSTRAP_RESAMPLES, metric_count), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    probabilities = np.full(reaction_count, 1.0 / reaction_count, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, _BOOTSTRAP_BATCH_SIZE):
        stop = min(start + _BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_RESAMPLES)
        counts = rng.multinomial(
            reaction_count,
            probabilities,
            size=stop - start,
        )
        bootstrap_means[start:stop] = counts @ samples / reaction_count

    if reaction_count == 1:
        accelerations = np.zeros(metric_count, dtype=np.float64)
    else:
        jackknife = (samples.sum(axis=0) - samples) / (reaction_count - 1)
        centered = jackknife.mean(axis=0) - jackknife
        squared_sum = np.square(centered).sum(axis=0)
        cubed_sum = np.power(centered, 3).sum(axis=0)
        denominator = 6.0 * np.power(squared_sum, 1.5)
        accelerations = np.divide(
            cubed_sum,
            denominator,
            out=np.zeros_like(cubed_sum),
            where=denominator != 0.0,
        )

    intervals: list[ConfidenceInterval] = []
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    z_alpha_low = _STANDARD_NORMAL.inv_cdf(alpha)
    z_alpha_high = _STANDARD_NORMAL.inv_cdf(1.0 - alpha)
    for column in range(metric_count):
        values = bootstrap_means[:, column]
        if np.all(samples[:, column] == samples[0, column]):
            intervals.append(_degenerate_bca(observed[column]))
            continue
        less = np.count_nonzero(values < observed[column])
        equal = np.count_nonzero(values == observed[column])
        probability = (less + 0.5 * equal) / BOOTSTRAP_RESAMPLES
        probability = min(
            max(probability, 0.5 / BOOTSTRAP_RESAMPLES),
            1.0 - 0.5 / BOOTSTRAP_RESAMPLES,
        )
        bias_correction = _STANDARD_NORMAL.inv_cdf(probability)
        lower_q = _adjusted_bca_quantile(
            bias_correction, accelerations[column], z_alpha_low
        )
        upper_q = _adjusted_bca_quantile(
            bias_correction, accelerations[column], z_alpha_high
        )
        low, high = np.quantile(
            values,
            [lower_q, upper_q],
            method="linear",
        )
        intervals.append(
            ConfidenceInterval(
                low=float(min(low, high)),
                high=float(max(low, high)),
                confidence_level=CONFIDENCE_LEVEL,
                method="BCa bootstrap",
            )
        )
    return tuple(intervals)


def _adjusted_bca_quantile(
    bias_correction: float,
    acceleration: float,
    z_alpha: float,
) -> float:
    numerator = bias_correction + z_alpha
    denominator = 1.0 - acceleration * numerator
    if denominator == 0.0:
        adjusted_z = math.copysign(math.inf, numerator)
    else:
        adjusted_z = bias_correction + numerator / denominator
    return min(max(_STANDARD_NORMAL.cdf(adjusted_z), 0.0), 1.0)


def _degenerate_bca(value: float) -> ConfidenceInterval:
    return ConfidenceInterval(
        low=float(value),
        high=float(value),
        confidence_level=CONFIDENCE_LEVEL,
        method="BCa bootstrap",
    )


def _validate_report_k(report_k: Sequence[int]) -> tuple[int, ...]:
    values = tuple(report_k)
    if not values or any(k <= 0 for k in values):
        raise ValueError("report_k must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("report_k must be unique and strictly increasing")
    return values


def _compare(left: object, right: object) -> int:
    return (left > right) - (left < right)


def _fraction_json(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _success_metric_json(
    successes: int,
    count: int,
    interval: ConfidenceInterval,
) -> dict[str, object]:
    return {
        **_fraction_json(Fraction(successes, count)),
        "successes": successes,
        "interval": interval.to_json_dict(),
    }


def _format_percent(value: Fraction) -> str:
    return f"{100.0 * float(value):.2f}%"
