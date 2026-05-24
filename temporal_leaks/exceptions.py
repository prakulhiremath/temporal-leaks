"""
temporal_leaks.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~

Custom exception hierarchy for temporal leakage detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnLeakMeta:
    """
    Metadata describing a leakage event in a single feature column.

    Attributes
    ----------
    column_name:
        The name of the feature column that exhibited temporal leakage.
    first_leaky_timestamp:
        The earliest timestamp (as a raw Python/pandas/polars scalar) at which
        the column value changed after future data was perturbed.
    mean_absolute_delta:
        Average absolute difference between baseline and perturbed outputs
        for the "past" partition (rows where timestamp <= T).
    max_delta:
        Maximum absolute difference observed across all past rows.
    pct_rows_changed:
        Fraction (0.0–1.0) of past rows where the feature value changed.
    effect_size:
        Normalised effect size in [0, 1]; computed as
        mean_absolute_delta / (baseline_std + 1e-9).
    severity:
        One of "LOW", "MEDIUM", "HIGH", or "CRITICAL" derived from
        *effect_size* thresholds.
    """

    column_name: str
    first_leaky_timestamp: Any
    mean_absolute_delta: float
    max_delta: float
    pct_rows_changed: float
    effect_size: float
    severity: str

    def __str__(self) -> str:
        return (
            f"[{self.severity}] column='{self.column_name}' "
            f"effect_size={self.effect_size:.4f} "
            f"mean_Δ={self.mean_absolute_delta:.6f} "
            f"max_Δ={self.max_delta:.6f} "
            f"rows_changed={self.pct_rows_changed:.1%} "
            f"first_leak@{self.first_leaky_timestamp}"
        )


class TemporalLeakageError(Exception):
    """
    Raised when the ``TemporalAudit`` engine detects statistically significant
    look-ahead bias in a feature-engineering pipeline.

    Parameters
    ----------
    message:
        Human-readable summary of the leakage.
    leakage_score:
        Aggregate leakage score in [0, 1].
    breached_columns:
        Ordered list of :class:`ColumnLeakMeta` objects, one per leaking column.

    Examples
    --------
    >>> try:
    ...     report = auditor.check(df, "timestamp", bad_pipeline)
    ... except TemporalLeakageError as exc:
    ...     print(exc.leakage_score)
    ...     for col in exc.breached_columns:
    ...         print(col)
    """

    def __init__(
        self,
        message: str,
        leakage_score: float,
        breached_columns: list[ColumnLeakMeta],
    ) -> None:
        super().__init__(message)
        self.leakage_score: float = leakage_score
        self.breached_columns: list[ColumnLeakMeta] = breached_columns

    def __str__(self) -> str:
        lines = [
            f"TemporalLeakageError: leakage_score={self.leakage_score:.4f}",
            f"  Breached columns ({len(self.breached_columns)}):",
        ]
        for meta in self.breached_columns:
            lines.append(f"    • {meta}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TemporalLeakageError("
            f"leakage_score={self.leakage_score!r}, "
            f"breached_columns={self.breached_columns!r})"
        )
