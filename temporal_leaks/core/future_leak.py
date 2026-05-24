"""
temporal_leaks.core.future_leak
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Future-leak detector for time-series train/test splits.

Validates that:
1. ``max(train_timestamps) < min(test_timestamps)`` — no temporal overlap.
2. Data has not been shuffled or sorted in a way that breaks temporal order.
3. No identical timestamps appear in both train and test partitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitLeakResult:
    """Result of a train/test split validation."""

    is_valid: bool
    train_max_ts: Any
    test_min_ts: Any
    overlap_timestamps: list[Any]
    is_shuffled: bool
    details: str


class FutureLeakDetector:
    """
    Validate that a time-series train/test split is temporally sound.

    Parameters
    ----------
    strict:
        If ``True``, requires a strict gap: ``train_max < test_min``.
        If ``False``, allows ``train_max <= test_min`` (permits equal boundary).
        Default: ``True``.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def check_split(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        timestamp_col: str,
    ) -> SplitLeakResult:
        """
        Validate a single train/test split for temporal leakage.

        Parameters
        ----------
        train:
            Training partition DataFrame.
        test:
            Test/validation partition DataFrame.
        timestamp_col:
            Name of the timestamp column in both frames.

        Returns
        -------
        SplitLeakResult
        """
        for name, frame in (("train", train), ("test", test)):
            if timestamp_col not in frame.columns:
                raise KeyError(f"timestamp_col={timestamp_col!r} not in {name} DataFrame")

        train_ts = train[timestamp_col]
        test_ts = test[timestamp_col]

        train_max = train_ts.max()
        test_min = test_ts.min()

        # Check temporal ordering
        overlap = set(train_ts.tolist()) & set(test_ts.tolist())
        overlap_list = sorted(overlap)

        if self.strict:
            no_overlap = train_max < test_min
        else:
            no_overlap = train_max <= test_min

        # Shuffling check: test whether the train timestamps are monotonically
        # non-decreasing (i.e., in temporal order)
        train_sorted = train_ts.is_monotonic_increasing
        test_sorted = test_ts.is_monotonic_increasing
        is_shuffled = not (train_sorted and test_sorted)

        is_valid = no_overlap and not bool(overlap_list)

        issues = []
        if not no_overlap:
            issues.append(
                f"Temporal overlap: train_max={train_max} >= test_min={test_min}"
            )
        if overlap_list:
            issues.append(f"Shared timestamps: {overlap_list[:5]}{'...' if len(overlap_list) > 5 else ''}")
        if is_shuffled:
            issues.append("Data appears shuffled — temporal monotonicity broken")

        details = "; ".join(issues) if issues else "Split is temporally valid"

        result = SplitLeakResult(
            is_valid=is_valid,
            train_max_ts=train_max,
            test_min_ts=test_min,
            overlap_timestamps=overlap_list,
            is_shuffled=is_shuffled,
            details=details,
        )

        if not is_valid or is_shuffled:
            logger.warning("FutureLeak | %s", details)
        else:
            logger.info("FutureLeak | Split validated OK")

        return result

    def check_splits(
        self,
        splits: list[tuple[pd.DataFrame, pd.DataFrame]],
        timestamp_col: str,
    ) -> list[SplitLeakResult]:
        """
        Validate multiple train/test splits (e.g., from TimeSeriesSplit).

        Parameters
        ----------
        splits:
            List of ``(train_df, test_df)`` tuples.
        timestamp_col:
            Timestamp column name.

        Returns
        -------
        list[SplitLeakResult]
        """
        return [
            self.check_split(train, test, timestamp_col)
            for train, test in splits
        ]
