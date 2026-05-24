"""
temporal_leaks.adapters.sklearn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scikit-learn compatible transformer that wraps ``TemporalAudit``.

Usage
-----
>>> from temporal_leaks.adapters.sklearn import TemporalAuditTransformer
>>> from sklearn.pipeline import Pipeline
>>>
>>> pipe = Pipeline([
...     ("audit", TemporalAuditTransformer(timestamp_col="ts", mode="nullify")),
...     ("model", MyModel()),
... ])
>>> pipe.fit(X_train)   # raises TemporalLeakageError if leakage found
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from temporal_leaks.auditor import AuditReport, PerturbationMode, TemporalAudit
from temporal_leaks.exceptions import TemporalLeakageError

logger = logging.getLogger(__name__)


class TemporalAuditTransformer:
    """
    A scikit-learn–style transformer (``fit`` / ``transform``) that runs
    ``TemporalAudit.check`` on every call to ``fit``.

    The transformer is *passthrough* — it does not alter the data. Its
    purpose is to act as a safety guard in a ``sklearn.pipeline.Pipeline``.

    Parameters
    ----------
    timestamp_col:
        Name of the timestamp column.
    pipeline_fn:
        Optional feature-engineering callable. If ``None``, an identity
        function is used (the transformer checks the input frame as-is).
    mode:
        Perturbation mode for ``TemporalAudit``.
    random_seed:
        Seed for reproducibility.
    delta_threshold:
        Minimum change to count as a difference.
    leakage_threshold:
        Leakage score above which ``TemporalLeakageError`` is raised.
    ignore_columns:
        Columns to skip during comparison.
    """

    def __init__(
        self,
        timestamp_col: str,
        pipeline_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        mode: PerturbationMode = "noise",
        random_seed: int = 42,
        delta_threshold: float = 1e-8,
        leakage_threshold: float = 0.0,
        ignore_columns: Optional[list[str]] = None,
    ) -> None:
        self.timestamp_col = timestamp_col
        self.pipeline_fn = pipeline_fn or (lambda df: df)
        self.mode = mode
        self.random_seed = random_seed
        self.delta_threshold = delta_threshold
        self.leakage_threshold = leakage_threshold
        self.ignore_columns = ignore_columns
        self._auditor = TemporalAudit(
            mode=mode,
            random_seed=random_seed,
            delta_threshold=delta_threshold,
            leakage_threshold=leakage_threshold,
            ignore_columns=ignore_columns,
        )
        self.audit_report_: Optional[AuditReport] = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TemporalAuditTransformer":
        """
        Run the temporal audit on ``X``.

        Raises
        ------
        TemporalLeakageError
            If leakage is detected above ``leakage_threshold``.
        """
        logger.info("TemporalAuditTransformer.fit | auditing input frame …")
        self.audit_report_ = self._auditor.check(
            X, timestamp_col=self.timestamp_col, pipeline_fn=self.pipeline_fn
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Pass-through: returns ``X`` unchanged."""
        return X

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit then transform."""
        return self.fit(X, y).transform(X, y)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "timestamp_col": self.timestamp_col,
            "mode": self.mode,
            "random_seed": self.random_seed,
            "delta_threshold": self.delta_threshold,
            "leakage_threshold": self.leakage_threshold,
            "ignore_columns": self.ignore_columns,
        }

    def set_params(self, **params: Any) -> "TemporalAuditTransformer":
        for key, value in params.items():
            setattr(self, key, value)
        # Rebuild internal auditor
        self._auditor = TemporalAudit(
            mode=self.mode,
            random_seed=self.random_seed,
            delta_threshold=self.delta_threshold,
            leakage_threshold=self.leakage_threshold,
            ignore_columns=self.ignore_columns,
        )
        return self

    def __repr__(self) -> str:
        return (
            f"TemporalAuditTransformer("
            f"timestamp_col={self.timestamp_col!r}, "
            f"mode={self.mode!r}, "
            f"random_seed={self.random_seed!r})"
        )
