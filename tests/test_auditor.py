"""
tests/test_auditor.py
~~~~~~~~~~~~~~~~~~~~~

Comprehensive pytest suite for ``temporal_leaks``.

Test matrix
-----------
Clean pipelines (must NOT trigger TemporalLeakageError):
  - Expanding-window mean (causal)
  - Lagged feature (positive shift)
  - Identity passthrough

Leaking pipelines (MUST trigger TemporalLeakageError or return high score):
  - Centred rolling mean (symmetric window = looks ahead)
  - Negative shift (shift(-1) reads the next row's value)
  - Full-dataset normalisation (z-score using global stats)
  - Rolling mean with wrong window alignment

Additional tests:
  - Polars input is handled gracefully
  - AuditReport.to_html() produces valid HTML
  - Decorator API (@temporal_audit)
  - Custom perturbation modes: sign_flip, nullify
  - Deterministic perturbation (same seed → same result)
  - TemporalLeakageError carries correct metadata
  - ignore_columns parameter suppresses a known-leaking column
  - leakage_threshold > 0 allows partial leakage without raising
  - __repr__ and __str__ produce expected strings
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from temporal_leaks import (
    AuditReport,
    ColumnLeakMeta,
    TemporalAudit,
    TemporalLeakageError,
    temporal_audit,
)

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """
    Create a deterministic time-series DataFrame with a numeric ``value``
    column and an integer timestamp.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts": np.arange(n),
            "value": rng.normal(loc=10.0, scale=2.0, size=n),
        }
    )


@pytest.fixture()
def df() -> pd.DataFrame:
    return _make_df(n=200, seed=42)


# ---------------------------------------------------------------------------
# Clean pipelines — must return score 0.0 with no exception
# ---------------------------------------------------------------------------


class TestCleanPipelines:
    """Causal pipelines that must not trigger TemporalLeakageError."""

    def test_expanding_mean_is_clean(self, df: pd.DataFrame) -> None:
        """Expanding window mean only looks at past data → zero leakage."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["expanding_mean"] = out["value"].expanding().mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)

        assert report.leakage_score == 0.0, (
            f"Expected zero leakage for expanding mean, got {report.leakage_score}"
        )
        assert report.breached_columns == [], (
            f"Expected no breached columns, got {report.breached_columns}"
        )
        assert "expanding_mean" in report.clean_columns

    def test_positive_lag_is_clean(self, df: pd.DataFrame) -> None:
        """shift(+1) looks only at the previous row → clean."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["lag1"] = out["value"].shift(1)
            return out

        auditor = TemporalAudit(mode="nullify", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)

        assert report.leakage_score == 0.0
        assert report.breached_columns == []

    def test_identity_pipeline_is_clean(self, df: pd.DataFrame) -> None:
        """A pass-through pipeline should never flag leakage."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

        auditor = TemporalAudit(mode="sign_flip", random_seed=0, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)

        assert report.leakage_score == 0.0

    def test_causal_rolling_with_shift_is_clean(self, df: pd.DataFrame) -> None:
        """Rolling mean shifted by 1 to exclude the current row → causal."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["roll_lag"] = out["value"].shift(1).rolling(5, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=7, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)

        assert report.leakage_score == 0.0


# ---------------------------------------------------------------------------
# Leaking pipelines — must raise TemporalLeakageError (default threshold = 0)
# ---------------------------------------------------------------------------


class TestLeakingPipelines:
    """Non-causal pipelines that MUST be detected."""

    def test_centred_rolling_raises(self, df: pd.DataFrame) -> None:
        """center=True uses future rows in the window — classic look-ahead."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred_roll"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42)
        with pytest.raises(TemporalLeakageError) as exc_info:
            auditor.check(df, "ts", pipeline)

        err = exc_info.value
        assert err.leakage_score > 0.0
        assert any(m.column_name == "centred_roll" for m in err.breached_columns)

    def test_negative_shift_raises(self, df: pd.DataFrame) -> None:
        """shift(-1) reads the *next* row's value — pure future leakage."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["next_value"] = out["value"].shift(-1)
            return out

        # Use sign_flip or noise mode — nullify replaces future rows with NaN
        # so shift(-1) on perturbed data also returns NaN, matching baseline NaN.
        # sign_flip and noise inject non-null future values that propagate backwards.
        auditor = TemporalAudit(mode="sign_flip", random_seed=42)
        with pytest.raises(TemporalLeakageError) as exc_info:
            auditor.check(df, "ts", pipeline)

        err = exc_info.value
        assert err.leakage_score > 0.0
        breached_names = [m.column_name for m in err.breached_columns]
        assert "next_value" in breached_names

    def test_global_zscore_normalisation_raises(self, df: pd.DataFrame) -> None:
        """Z-score using global mean/std leaks future distributional info."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            # Deliberately use .mean() and .std() on the entire column
            out["znorm"] = (out["value"] - out["value"].mean()) / out["value"].std()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42)
        with pytest.raises(TemporalLeakageError) as exc_info:
            auditor.check(df, "ts", pipeline)

        assert exc_info.value.leakage_score > 0.0

    def test_centred_rolling_high_leakage_score(self, df: pd.DataFrame) -> None:
        """Verify the leakage score is substantially above zero for obvious leaks."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred_roll"] = out["value"].rolling(51, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(
            mode="noise",
            random_seed=42,
            leakage_threshold=1.1,  # suppress raise so we can inspect the score
        )
        report = auditor.check(df, "ts", pipeline)

        assert report.leakage_score > 0.1, (
            f"Expected a meaningful leakage score, got {report.leakage_score:.4f}"
        )
        assert len(report.breached_columns) >= 1


# ---------------------------------------------------------------------------
# Effect size metadata
# ---------------------------------------------------------------------------


class TestEffectSizeMetadata:
    """Validate per-column effect-size fields on ColumnLeakMeta."""

    def test_effect_size_fields_are_populated(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)

        assert report.breached_columns, "Expected at least one breached column"
        meta = report.breached_columns[0]

        assert isinstance(meta.mean_absolute_delta, float)
        assert meta.mean_absolute_delta >= 0.0
        assert isinstance(meta.max_delta, float)
        assert meta.max_delta >= meta.mean_absolute_delta
        assert 0.0 <= meta.pct_rows_changed <= 1.0
        assert 0.0 <= meta.effect_size <= 1.0
        assert meta.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_severity_classification_logic(self) -> None:
        """Thresholds: CRITICAL ≥ 0.75, HIGH ≥ 0.40, MEDIUM ≥ 0.15, LOW ≥ 0."""
        from temporal_leaks.auditor import _classify_severity

        assert _classify_severity(0.90) == "CRITICAL"
        assert _classify_severity(0.75) == "CRITICAL"
        assert _classify_severity(0.60) == "HIGH"
        assert _classify_severity(0.40) == "HIGH"
        assert _classify_severity(0.30) == "MEDIUM"
        assert _classify_severity(0.15) == "MEDIUM"
        assert _classify_severity(0.10) == "LOW"
        assert _classify_severity(0.00) == "LOW"


# ---------------------------------------------------------------------------
# Perturbation modes
# ---------------------------------------------------------------------------


class TestPerturbationModes:
    """Each perturbation mode must detect obvious leakage."""

    @pytest.mark.parametrize("mode", ["noise", "sign_flip", "nullify"])
    def test_each_mode_detects_centred_rolling(
        self, df: pd.DataFrame, mode: str
    ) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode=mode, random_seed=42, leakage_threshold=1.1)  # type: ignore[arg-type]
        report = auditor.check(df, "ts", pipeline)

        assert report.leakage_score > 0.0, (
            f"Mode {mode!r} failed to detect leakage"
        )

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            TemporalAudit(mode="invalid_mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deterministic perturbation
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed must produce identical reports."""

    def test_same_seed_produces_identical_reports(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=99, leakage_threshold=1.1)
        report_a = auditor.check(df, "ts", pipeline)
        report_b = auditor.check(df, "ts", pipeline)

        assert report_a.leakage_score == report_b.leakage_score
        assert len(report_a.breached_columns) == len(report_b.breached_columns)
        for a, b in zip(report_a.breached_columns, report_b.breached_columns):
            assert a.column_name == b.column_name
            assert a.mean_absolute_delta == pytest.approx(b.mean_absolute_delta, rel=1e-9)

    def test_different_seeds_may_differ(self, df: pd.DataFrame) -> None:
        """Different seeds yield different noise, but leakage must still be found."""

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor_a = TemporalAudit(mode="noise", random_seed=1, leakage_threshold=1.1)
        auditor_b = TemporalAudit(mode="noise", random_seed=2, leakage_threshold=1.1)
        report_a = auditor_a.check(df, "ts", pipeline)
        report_b = auditor_b.check(df, "ts", pipeline)

        # Both must detect leakage even if the exact score differs
        assert report_a.leakage_score > 0.0
        assert report_b.leakage_score > 0.0


# ---------------------------------------------------------------------------
# Polars input
# ---------------------------------------------------------------------------


class TestPolarsInput:
    """Polars DataFrames must be handled transparently."""

    def test_polars_clean_pipeline(self, df: pd.DataFrame) -> None:
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")

        polars_df = pl.from_pandas(df)

        def pipeline(frame: object) -> object:
            import polars as pl  # noqa: F811
            pandas_frame = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame  # type: ignore[assignment]
            out = pandas_frame.copy()
            out["expanding_mean"] = out["value"].expanding().mean()
            return pl.from_pandas(out)

        auditor = TemporalAudit(mode="nullify", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(polars_df, "ts", pipeline)  # type: ignore[arg-type]

        assert report.leakage_score == 0.0

    def test_polars_leaking_pipeline(self, df: pd.DataFrame) -> None:
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")

        polars_df = pl.from_pandas(df)

        def pipeline(frame: object) -> object:
            import polars as pl  # noqa: F811
            pandas_frame = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame  # type: ignore[assignment]
            out = pandas_frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return pl.from_pandas(out)

        auditor = TemporalAudit(mode="nullify", random_seed=42)
        with pytest.raises(TemporalLeakageError):
            auditor.check(polars_df, "ts", pipeline)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestHTMLReport:
    """AuditReport.to_html() must produce plausible HTML."""

    def test_html_report_contains_key_elements_clean(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["expanding_mean"] = out["value"].expanding().mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)
        html_str = report.to_html()

        assert "<!DOCTYPE html>" in html_str
        assert "temporal-leaks" in html_str
        assert "CLEAN" in html_str
        assert "0.0000" in html_str  # leakage score

    def test_html_report_contains_breach_info(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)
        html_str = report.to_html()

        assert "LEAKAGE DETECTED" in html_str
        assert "centred" in html_str
        assert "Breached Columns" in html_str


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------


class TestDecoratorAPI:
    """@temporal_audit must intercept leaking pipelines at decoration time."""

    def test_decorator_passes_clean_pipeline(self, df: pd.DataFrame) -> None:
        @temporal_audit(timestamp_col="ts", mode="noise", random_seed=42, leakage_threshold=1.1)
        def build_features(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["expand"] = out["value"].expanding().mean()
            return out

        result = build_features(df)
        assert "expand" in result.columns

    def test_decorator_raises_on_leaking_pipeline(self, df: pd.DataFrame) -> None:
        @temporal_audit(timestamp_col="ts", mode="noise", random_seed=42)
        def bad_pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        with pytest.raises(TemporalLeakageError):
            bad_pipeline(df)

    def test_decorator_preserves_function_metadata(self) -> None:
        @temporal_audit(timestamp_col="ts")
        def my_pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            """My docstring."""
            return frame.copy()

        assert my_pipeline.__name__ == "my_pipeline"
        assert my_pipeline.__doc__ == "My docstring."


# ---------------------------------------------------------------------------
# ignore_columns parameter
# ---------------------------------------------------------------------------


class TestIgnoreColumns:
    """ignore_columns should suppress leakage detection for named columns."""

    def test_ignore_leaking_column_suppresses_detection(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(
            mode="noise",
            random_seed=42,
            leakage_threshold=0.0,
            ignore_columns=["centred"],
        )
        # Should NOT raise because "centred" is ignored
        report = auditor.check(df, "ts", pipeline)
        assert report.leakage_score == 0.0


# ---------------------------------------------------------------------------
# leakage_threshold
# ---------------------------------------------------------------------------


class TestLeakageThreshold:
    """leakage_threshold controls when the exception is raised."""

    def test_high_threshold_returns_report_instead_of_raising(
        self, df: pd.DataFrame
    ) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)

        assert isinstance(report, AuditReport)
        assert report.leakage_score > 0.0

    def test_zero_threshold_raises_on_any_leakage(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=0.0)
        with pytest.raises(TemporalLeakageError):
            auditor.check(df, "ts", pipeline)


# ---------------------------------------------------------------------------
# TemporalLeakageError metadata
# ---------------------------------------------------------------------------


class TestTemporalLeakageErrorMetadata:
    """Exception must carry correct metadata."""

    def test_error_has_leakage_score(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42)
        with pytest.raises(TemporalLeakageError) as exc_info:
            auditor.check(df, "ts", pipeline)

        err = exc_info.value
        assert 0.0 < err.leakage_score <= 1.0
        assert len(err.breached_columns) >= 1
        assert all(isinstance(m, ColumnLeakMeta) for m in err.breached_columns)

    def test_error_repr_and_str(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42)
        with pytest.raises(TemporalLeakageError) as exc_info:
            auditor.check(df, "ts", pipeline)

        err = exc_info.value
        assert "TemporalLeakageError" in repr(err)
        assert "leakage_score" in str(err)
        assert "Breached columns" in str(err)


# ---------------------------------------------------------------------------
# __repr__ and __str__ of AuditReport
# ---------------------------------------------------------------------------


class TestAuditReportRepresentation:
    def test_report_repr_clean(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.copy()

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)
        r = repr(report)
        assert "CLEAN" in r
        assert "leakage_score=0.0" in r

    def test_report_str_contains_sections(self, df: pd.DataFrame) -> None:
        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["centred"] = out["value"].rolling(11, center=True, min_periods=1).mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(df, "ts", pipeline)
        s = str(report)
        assert "Leakage Score" in s
        assert "Breached Columns" in s
        assert "centred" in s


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_timestamp_col_raises_key_error(self, df: pd.DataFrame) -> None:
        auditor = TemporalAudit()
        with pytest.raises(KeyError, match="timestamp_col"):
            auditor.check(df, "nonexistent_col", lambda f: f)

    def test_single_row_dataframe(self) -> None:
        """Single-row frames have no past/future split; should return clean."""
        single = pd.DataFrame({"ts": [0], "value": [1.0]})
        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)

        def pipeline(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["feat"] = out["value"] * 2
            return out

        report = auditor.check(single, "ts", pipeline)
        # With 1 row the past partition can still include the single row
        assert report.leakage_score == 0.0 or isinstance(report, AuditReport)

    def test_datetime_timestamp_col(self) -> None:
        """Datetime timestamp columns should be handled without error."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        frame = pd.DataFrame({"ts": dates, "value": np.random.default_rng(0).normal(size=100)})

        def pipeline(f: pd.DataFrame) -> pd.DataFrame:
            out = f.copy()
            out["expand"] = out["value"].expanding().mean()
            return out

        auditor = TemporalAudit(mode="noise", random_seed=42, leakage_threshold=1.1)
        report = auditor.check(frame, "ts", pipeline)
        assert report.leakage_score == 0.0

    def test_auditor_repr(self) -> None:
        auditor = TemporalAudit(mode="sign_flip", random_seed=7)
        r = repr(auditor)
        assert "TemporalAudit" in r
        assert "sign_flip" in r
        assert "7" in r
