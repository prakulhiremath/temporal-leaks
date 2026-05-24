"""
temporal_leaks.core.contamination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train-test contamination detector.

Identifies identical or near-identical rows / sub-sequences present in
both training and evaluation sets using hash-based exact matching and
optional distance-based approximate matching for time-series windows.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContaminationResult:
    """Result of a contamination check between train and test sets."""

    n_exact_duplicates: int
    duplicate_indices_train: list[int]
    duplicate_indices_test: list[int]
    contamination_rate: float  # fraction of test rows that are duplicates
    is_contaminated: bool
    details: str


class ContaminationDetector:
    """
    Detect train-test contamination via hash-based row matching and
    optional sliding-window sub-sequence detection.

    Parameters
    ----------
    window_size:
        Length of sub-sequence windows to hash for near-duplicate detection.
        Set to ``None`` to skip window-based detection (row-only). Default: ``None``.
    contamination_threshold:
        Fraction of test rows that must be duplicated before flagging
        contamination. Default: ``0.0`` (flag any duplicate).
    numeric_cols:
        If provided, only these columns are used for hashing. If ``None``,
        all numeric columns are used.
    round_decimals:
        Round numeric values to this many decimal places before hashing
        (suppresses floating-point jitter). Default: ``6``.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        contamination_threshold: float = 0.0,
        numeric_cols: Optional[list[str]] = None,
        round_decimals: int = 6,
    ) -> None:
        self.window_size = window_size
        self.contamination_threshold = contamination_threshold
        self.numeric_cols = numeric_cols
        self.round_decimals = round_decimals

    def _hash_rows(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        """Return a Series of SHA-256 hashes for each row."""
        cols = self.numeric_cols or df.select_dtypes(include="number").columns.tolist()
        rounded = df[cols].round(self.round_decimals)
        # Vectorised: convert each row to bytes and hash
        hashes = rounded.apply(
            lambda row: hashlib.sha256(row.values.tobytes()).hexdigest(), axis=1
        )
        return hashes

    def _hash_windows(self, df: pd.DataFrame, window_size: int) -> set[str]:
        """Return a set of hashes for all sliding windows of ``window_size`` rows."""
        cols = self.numeric_cols or df.select_dtypes(include="number").columns.tolist()
        arr = df[cols].round(self.round_decimals).values
        window_hashes: set[str] = set()
        for i in range(len(arr) - window_size + 1):
            chunk = arr[i : i + window_size]
            h = hashlib.sha256(chunk.tobytes()).hexdigest()
            window_hashes.add(h)
        return window_hashes

    def check(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> ContaminationResult:
        """
        Check for train-test contamination.

        Parameters
        ----------
        train:
            Training partition DataFrame.
        test:
            Test/validation partition DataFrame.

        Returns
        -------
        ContaminationResult
        """
        train_hashes = self._hash_rows(train)
        test_hashes = self._hash_rows(test)

        train_hash_set = set(train_hashes.values)

        dup_test_indices: list[int] = []
        dup_train_indices: list[int] = []

        for idx, h in test_hashes.items():
            if h in train_hash_set:
                dup_test_indices.append(int(idx))
                # Find matching train indices
                matching = train_hashes[train_hashes == h].index.tolist()
                dup_train_indices.extend(int(i) for i in matching)

        n_exact = len(dup_test_indices)
        contamination_rate = n_exact / max(len(test), 1)

        issues = []
        if n_exact > 0:
            issues.append(f"{n_exact} exact duplicate row(s) found in test set")

        # Window-based detection
        if self.window_size is not None:
            train_windows = self._hash_windows(train, self.window_size)
            test_windows = self._hash_windows(test, self.window_size)
            shared = train_windows & test_windows
            if shared:
                issues.append(
                    f"{len(shared)} overlapping sub-sequence window(s) of size {self.window_size}"
                )

        is_contaminated = contamination_rate > self.contamination_threshold

        details = "; ".join(issues) if issues else "No contamination detected"

        result = ContaminationResult(
            n_exact_duplicates=n_exact,
            duplicate_indices_train=sorted(set(dup_train_indices)),
            duplicate_indices_test=dup_test_indices,
            contamination_rate=contamination_rate,
            is_contaminated=is_contaminated,
            details=details,
        )

        if is_contaminated:
            logger.warning("Contamination | %s", details)
        else:
            logger.info("Contamination | %s", details)

        return result
