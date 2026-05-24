"""
temporal_leaks.core.lookahead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature-level look-ahead bias detector.

Detects whether a feature computed for index ``t`` contains information
from ``t+k`` (k > 0) by running rolling cross-correlation checks and
shift-mutation tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LookAheadResult:
    """Result for a single column's look-ahead check."""

    column_name: str
    max_forward_correlation: float
    suspicious_lags: list[int]
    is_leaking: bool
    details: str


class LookAheadDetector:
    """
    Detect look-ahead bias in feature columns via cross-correlation analysis.

    For each numeric feature column, this detector computes the cross-correlation
    between the feature and the raw input series shifted by 1..max_lag steps
    *forward in time* (i.e., future values). A high positive correlation at
    forward lags suggests the feature was computed using future data.

    Parameters
    ----------
    max_lag:
        Maximum number of forward steps to test. Default: 10.
    correlation_threshold:
        Pearson correlation magnitude above which a lag is flagged. Default: 0.7.
    """

    def __init__(
        self,
        max_lag: int = 10,
        correlation_threshold: float = 0.7,
    ) -> None:
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if not (0.0 < correlation_threshold <= 1.0):
            raise ValueError(f"correlation_threshold must be in (0, 1], got {correlation_threshold}")
        self.max_lag = max_lag
        self.correlation_threshold = correlation_threshold

    def check(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        raw_col: str,
        feature_cols: Optional[list[str]] = None,
    ) -> list[LookAheadResult]:
        """
        Check for look-ahead bias between ``raw_col`` and each feature column.

        Parameters
        ----------
        df:
            DataFrame sorted by time.
        timestamp_col:
            Name of the timestamp column (used for sorting).
        raw_col:
            The source/raw data column to correlate against.
        feature_cols:
            Columns to inspect. If ``None``, all numeric columns except
            ``timestamp_col`` and ``raw_col`` are used.

        Returns
        -------
        list[LookAheadResult]
            One result per inspected feature column.
        """
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        if raw_col not in df.columns:
            raise KeyError(f"raw_col={raw_col!r} not found in DataFrame")

        if feature_cols is None:
            feature_cols = [
                c for c in df.select_dtypes(include="number").columns
                if c not in (timestamp_col, raw_col)
            ]

        raw_series = pd.to_numeric(df[raw_col], errors="coerce")
        results: list[LookAheadResult] = []

        for col in feature_cols:
            if col not in df.columns:
                logger.warning("Column %r not in DataFrame; skipping.", col)
                continue

            feat = pd.to_numeric(df[col], errors="coerce")
            suspicious: list[int] = []
            max_corr = 0.0

            for lag in range(1, self.max_lag + 1):
                # Shift raw series *forward* by lag (future values relative to past)
                future_raw = raw_series.shift(-lag)
                valid = feat.notna() & future_raw.notna()
                if valid.sum() < 10:
                    continue
                corr_matrix = np.corrcoef(feat[valid].values, future_raw[valid].values)
                corr = float(abs(corr_matrix[0, 1]))
                if corr > max_corr:
                    max_corr = corr
                if corr >= self.correlation_threshold:
                    suspicious.append(lag)

            is_leaking = len(suspicious) > 0
            details = (
                f"max_forward_corr={max_corr:.4f} at lag(s)={suspicious}"
                if is_leaking
                else f"max_forward_corr={max_corr:.4f} — no suspicious lags"
            )

            results.append(
                LookAheadResult(
                    column_name=col,
                    max_forward_correlation=max_corr,
                    suspicious_lags=suspicious,
                    is_leaking=is_leaking,
                    details=details,
                )
            )
            if is_leaking:
                logger.warning("LookAhead | %s | %s", col, details)

        return results
