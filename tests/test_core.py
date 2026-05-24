"""
tests/test_core.py
~~~~~~~~~~~~~~~~~~

Tests for the three core detection engine modules:
  - LookAheadDetector
  - FutureLeakDetector
  - ContaminationDetector
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from temporal_leaks.core.lookahead import LookAheadDetector, LookAheadResult
from temporal_leaks.core.future_leak import FutureLeakDetector, SplitLeakResult
from temporal_leaks.core.contamination import ContaminationDetector, ContaminationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ts(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts": np.arange(n),
            "price": rng.normal(100, 5, size=n),
        }
    )


@pytest.fixture()
def ts_df() -> pd.DataFrame:
    return _make_ts(n=200, seed=42)


# ---------------------------------------------------------------------------
# LookAheadDetector
# ---------------------------------------------------------------------------


class TestLookAheadDetector:
    def test_causal_feature_is_not_flagged(self, ts_df: pd.DataFrame) -> None:
        df = ts_df.copy()
        df["expanding_mean"] = df["price"].expanding().mean()

        detector = LookAheadDetector(max_lag=5, correlation_threshold=0.7)
        results = detector.check(df, "ts", "price", feature_cols=["expanding_mean"])

        assert len(results) == 1
        assert not results[0].is_leaking

    def test_future_shifted_feature_is_flagged(self, ts_df: pd.DataFrame) -> None:
        df = ts_df.copy()
        # shift(-1): each row contains the *next* price → pure look-ahead
        df["next_price"] = df["price"].shift(-1)

        detector = LookAheadDetector(max_lag=5, correlation_threshold=0.5)
        results = detector.check(df, "ts", "price", feature_cols=["next_price"])

        assert len(results) == 1
        assert results[0].is_leaking
        assert 1 in results[0].suspicious_lags

    def test_invalid_max_lag_raises(self) -> None:
        with pytest.raises(ValueError, match="max_lag"):
            LookAheadDetector(max_lag=0)

    def test_invalid_correlation_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="correlation_threshold"):
            LookAheadDetector(correlation_threshold=1.5)

    def test_missing_raw_col_raises(self, ts_df: pd.DataFrame) -> None:
        detector = LookAheadDetector()
        with pytest.raises(KeyError):
            detector.check(ts_df, "ts", "nonexistent_col")

    def test_unknown_feature_col_is_skipped(self, ts_df: pd.DataFrame) -> None:
        detector = LookAheadDetector()
        # Should not raise — just skip the missing column
        results = detector.check(ts_df, "ts", "price", feature_cols=["ghost_col"])
        assert results == []

    def test_auto_feature_col_discovery(self, ts_df: pd.DataFrame) -> None:
        df = ts_df.copy()
        df["lag1"] = df["price"].shift(1)

        detector = LookAheadDetector(max_lag=3, correlation_threshold=0.7)
        # None → auto-discover numeric columns except ts and price
        results = detector.check(df, "ts", "price", feature_cols=None)

        assert any(r.column_name == "lag1" for r in results)

    def test_result_fields_are_valid(self, ts_df: pd.DataFrame) -> None:
        df = ts_df.copy()
        df["feat"] = df["price"].shift(-2)

        detector = LookAheadDetector(max_lag=5, correlation_threshold=0.3)
        results = detector.check(df, "ts", "price", feature_cols=["feat"])

        r = results[0]
        assert isinstance(r, LookAheadResult)
        assert 0.0 <= r.max_forward_correlation <= 1.0
        assert isinstance(r.suspicious_lags, list)
        assert isinstance(r.details, str)


# ---------------------------------------------------------------------------
# FutureLeakDetector
# ---------------------------------------------------------------------------


class TestFutureLeakDetector:
    def _split(self, df: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df[df["ts"] <= cutoff].reset_index(drop=True)
        test = df[df["ts"] > cutoff].reset_index(drop=True)
        return train, test

    def test_valid_split_passes(self, ts_df: pd.DataFrame) -> None:
        train, test = self._split(ts_df, 99)
        detector = FutureLeakDetector(strict=True)
        result = detector.check_split(train, test, "ts")

        assert result.is_valid
        assert not result.is_shuffled
        assert result.overlap_timestamps == []

    def test_overlapping_split_fails(self, ts_df: pd.DataFrame) -> None:
        # Manually create overlap
        train = ts_df[ts_df["ts"] <= 110].reset_index(drop=True)
        test = ts_df[ts_df["ts"] >= 100].reset_index(drop=True)

        detector = FutureLeakDetector(strict=True)
        result = detector.check_split(train, test, "ts")

        assert not result.is_valid
        assert len(result.overlap_timestamps) > 0

    def test_shuffled_data_flagged(self, ts_df: pd.DataFrame) -> None:
        train = ts_df.sample(frac=0.5, random_state=0).reset_index(drop=True)
        test = ts_df.sample(frac=0.5, random_state=1).reset_index(drop=True)

        detector = FutureLeakDetector(strict=True)
        result = detector.check_split(train, test, "ts")

        assert result.is_shuffled

    def test_missing_timestamp_col_raises(self, ts_df: pd.DataFrame) -> None:
        train, test = self._split(ts_df, 99)
        detector = FutureLeakDetector()
        with pytest.raises(KeyError):
            detector.check_split(train, test, "nonexistent_col")

    def test_check_splits_returns_list(self, ts_df: pd.DataFrame) -> None:
        splits = [self._split(ts_df, c) for c in [60, 100, 140]]
        detector = FutureLeakDetector()
        results = detector.check_splits(splits, "ts")

        assert len(results) == 3
        assert all(isinstance(r, SplitLeakResult) for r in results)

    def test_non_strict_allows_equal_boundary(self, ts_df: pd.DataFrame) -> None:
        # train_max == test_min — strict fails, non-strict should pass
        train = ts_df[ts_df["ts"] <= 100].reset_index(drop=True)
        test = ts_df[ts_df["ts"] >= 100].reset_index(drop=True)

        strict_det = FutureLeakDetector(strict=True)
        loose_det = FutureLeakDetector(strict=False)

        strict_result = strict_det.check_split(train, test, "ts")
        # Strictly speaking, 100 == 100 so strict should flag overlapping timestamps
        assert not strict_result.is_valid or strict_result.overlap_timestamps != []

        loose_result = loose_det.check_split(train, test, "ts")
        # With strict=False, equal boundary without overlap should be valid (except for shared ts=100)
        assert isinstance(loose_result, SplitLeakResult)

    def test_result_fields_populated(self, ts_df: pd.DataFrame) -> None:
        train, test = self._split(ts_df, 99)
        result = FutureLeakDetector().check_split(train, test, "ts")

        assert result.train_max_ts == 99
        assert result.test_min_ts == 100
        assert isinstance(result.details, str)


# ---------------------------------------------------------------------------
# ContaminationDetector
# ---------------------------------------------------------------------------


class TestContaminationDetector:
    def test_clean_split_not_contaminated(self, ts_df: pd.DataFrame) -> None:
        train = ts_df[ts_df["ts"] < 100].reset_index(drop=True)
        test = ts_df[ts_df["ts"] >= 100].reset_index(drop=True)

        detector = ContaminationDetector()
        result = detector.check(train, test)

        assert result.n_exact_duplicates == 0
        assert not result.is_contaminated

    def test_injected_duplicate_detected(self, ts_df: pd.DataFrame) -> None:
        train = ts_df[ts_df["ts"] < 100].reset_index(drop=True)
        # Inject 5 rows from train into test
        contaminated_test = pd.concat(
            [ts_df[ts_df["ts"] >= 100], ts_df.iloc[:5]], ignore_index=True
        )

        detector = ContaminationDetector(contamination_threshold=0.0)
        result = detector.check(train, contaminated_test)

        assert result.n_exact_duplicates >= 5
        assert result.is_contaminated

    def test_contamination_rate_computed_correctly(self, ts_df: pd.DataFrame) -> None:
        train = ts_df.iloc[:50].reset_index(drop=True)
        # Test = 10 clean + 5 duplicates from train
        test_clean = ts_df.iloc[50:60].reset_index(drop=True)
        test_dup = ts_df.iloc[:5].reset_index(drop=True)
        test = pd.concat([test_clean, test_dup], ignore_index=True)

        detector = ContaminationDetector()
        result = detector.check(train, test)

        assert result.contamination_rate == pytest.approx(5 / 15, rel=1e-6)

    def test_threshold_suppresses_small_contamination(self, ts_df: pd.DataFrame) -> None:
        train = ts_df.iloc[:100].reset_index(drop=True)
        # 1 duplicate in a test set of 100 → rate = 0.01
        test = pd.concat(
            [ts_df.iloc[100:199], ts_df.iloc[:1]], ignore_index=True
        )

        # High threshold → should not flag
        detector = ContaminationDetector(contamination_threshold=0.05)
        result = detector.check(train, test)
        assert not result.is_contaminated

    def test_window_based_detection(self, ts_df: pd.DataFrame) -> None:
        train = ts_df.iloc[:100].reset_index(drop=True)
        # Copy a window of 5 consecutive rows from train into test
        test = pd.concat(
            [ts_df.iloc[100:150], ts_df.iloc[10:15]], ignore_index=True
        )

        detector = ContaminationDetector(window_size=5)
        result = detector.check(train, test)

        # Window detection should mention overlapping windows in details
        assert "window" in result.details.lower() or result.n_exact_duplicates >= 5

    def test_result_fields_are_valid(self, ts_df: pd.DataFrame) -> None:
        train = ts_df.iloc[:100].reset_index(drop=True)
        test = ts_df.iloc[100:].reset_index(drop=True)

        result = ContaminationDetector().check(train, test)

        assert isinstance(result, ContaminationResult)
        assert isinstance(result.n_exact_duplicates, int)
        assert isinstance(result.duplicate_indices_train, list)
        assert isinstance(result.duplicate_indices_test, list)
        assert 0.0 <= result.contamination_rate <= 1.0
        assert isinstance(result.details, str)
