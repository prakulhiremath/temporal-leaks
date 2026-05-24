"""
tests/test_adapters.py
~~~~~~~~~~~~~~~~~~~~~~

Tests for the sklearn adapter (TemporalAuditTransformer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from temporal_leaks.adapters.sklearn import TemporalAuditTransformer
from temporal_leaks.exceptions import TemporalLeakageError


@pytest.fixture()
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"ts": np.arange(200), "value": rng.normal(10, 2, size=200)}
    )


class TestTemporalAuditTransformer:
    def test_fit_clean_data_does_not_raise(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["expand"] = out["value"].expanding().mean()
            return out

        t = TemporalAuditTransformer(
            timestamp_col="ts",
            pipeline_fn=pipeline,
            mode="noise",
            leakage_threshold=1.1,
        )
        t.fit(df)
        assert t.audit_report_ is not None
        assert t.audit_report_.leakage_score == 0.0

    def test_fit_leaking_data_raises(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        t = TemporalAuditTransformer(timestamp_col="ts", pipeline_fn=pipeline, mode="noise")
        with pytest.raises(TemporalLeakageError):
            t.fit(df)

    def test_transform_returns_input_unchanged(self, df: pd.DataFrame) -> None:
        t = TemporalAuditTransformer(timestamp_col="ts", leakage_threshold=1.1)
        t.fit(df)
        result = t.transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_fit_transform_chain(self, df: pd.DataFrame) -> None:
        t = TemporalAuditTransformer(timestamp_col="ts", leakage_threshold=1.1)
        result = t.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_get_params(self, df: pd.DataFrame) -> None:
        t = TemporalAuditTransformer(timestamp_col="ts", mode="nullify", random_seed=7)
        params = t.get_params()
        assert params["timestamp_col"] == "ts"
        assert params["mode"] == "nullify"
        assert params["random_seed"] == 7

    def test_set_params(self, df: pd.DataFrame) -> None:
        t = TemporalAuditTransformer(timestamp_col="ts")
        t.set_params(mode="sign_flip", random_seed=99)
        assert t.mode == "sign_flip"
        assert t.random_seed == 99

    def test_repr(self) -> None:
        t = TemporalAuditTransformer(timestamp_col="ts", mode="nullify")
        r = repr(t)
        assert "TemporalAuditTransformer" in r
        assert "ts" in r
        assert "nullify" in r
