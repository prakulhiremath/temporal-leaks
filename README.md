# 🕵️ Temporal Leaks: Valgrind for Time-Series ML

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/pandas-%E2%9C%93-150458?style=flat-square&logo=pandas" />
  <img src="https://img.shields.io/badge/polars-%E2%9C%93-CD792C?style=flat-square" />
  <img src="https://img.shields.io/badge/mypy-strict-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/ruff-lint-red?style=flat-square" />
  <img src="https://img.shields.io/badge/CI-passing-brightgreen?style=flat-square&logo=github-actions" />
</p>

---

> **Look-ahead bias** is the silent killer of quant strategies and forecasting models.  
> Your backtest shows 40% annual returns. You deploy. You lose money.  
> Somewhere in your feature pipeline, a rolling average peeked at tomorrow's prices.

`temporal-leaks` catches this automatically — before it costs you.

---

## The Problem: Future Data in Your Past Features

In time-series machine learning, look-ahead bias (also called *data leakage* or *future leakage*) occurs when a feature computed for timestamp `t` inadvertently uses data from timestamps `t+1`, `t+2`, … `t+n`.

This is devastatingly easy to introduce:

```python
# BUG: center=True means the window is centred — it looks forward AND backward
df["roll_mean"] = df["price"].rolling(window=5, center=True).mean()

# BUG: shift(-1) reads the NEXT row's value
df["next_return"] = df["return"].shift(-1)

# BUG: global z-score uses future data to compute mean/std
df["znorm"] = (df["price"] - df["price"].mean()) / df["price"].std()
```

None of these will raise an error.  
Your tests will pass.  
Your backtests will look amazing.  
**And then reality hits.**

---

## How It Works: The Temporal Perturbation Test

```
  Timeline:   ──────────────────────────────────────────────────▶
                              T (midpoint)
                              │
  Past ◀──────────────────────┤──────────────────────────────▶ Future
                              │
  Step 1:  Run pipeline on original data
           baseline_features = pipeline(df)

  Step 2:  MUTATE the future
           df_perturbed[t > T] = 🔥 (noise / sign flip / NaN)

  Step 3:  Re-run pipeline on perturbed data
           perturbed_features = pipeline(df_perturbed)

  Step 4:  Compare features for PAST rows only (t ≤ T)
           If baseline_features[t≤T] ≠ perturbed_features[t≤T]
           then the past features DEPEND on future data → LEAK! 🚨
```

The key insight: **if your past features are truly causal, mutating the future should not change them.** If they change, future data crept in.

---

## Installation

```bash
pip install temporal-leaks
```

Or from source:

```bash
git clone https://github.com/temporal-leaks/temporal-leaks
cd temporal-leaks
pip install -e ".[dev]"
```

---

## Quick Start

```python
import pandas as pd
import numpy as np
from temporal_leaks import TemporalAudit, TemporalLeakageError

# Build a sample time-series dataset
df = pd.DataFrame({
    "ts":    np.arange(500),
    "price": np.random.default_rng(42).normal(100, 5, size=500),
})

# ─── ✓ CLEAN PIPELINE ────────────────────────────────────────────────────────
def causal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Expanding window only looks at past — safe!
    out["expanding_mean"] = out["price"].expanding(min_count=1).mean()
    # shift(+1) looks at the previous row — safe!
    out["lag1"] = out["price"].shift(1)
    return out

auditor = TemporalAudit(mode="nullify", random_seed=42)
report  = auditor.check(df, timestamp_col="ts", pipeline_fn=causal_features)
print(report)
# ✓  CLEAN — leakage_score=0.0000


# ─── ✗ LEAKING PIPELINE ──────────────────────────────────────────────────────
def leaking_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # center=True peeks at future rows — LEAKS!
    out["centred_roll"] = out["price"].rolling(11, center=True, min_periods=1).mean()
    return out

try:
    auditor.check(df, timestamp_col="ts", pipeline_fn=leaking_features)
except TemporalLeakageError as exc:
    print(exc)
    # TemporalLeakageError: leakage_score=0.4812
    #   Breached columns (1):
    #     • [HIGH] column='centred_roll' effect_size=0.4812 ...
```

---

## Decorator API

```python
from temporal_leaks import temporal_audit

@temporal_audit(timestamp_col="ts", mode="noise", random_seed=42)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["expanding_mean"] = df["price"].expanding(min_count=1).mean()
    return df

# The audit runs automatically on every call.
# TemporalLeakageError is raised if leakage is detected.
result = build_features(df)
```

---

## HTML Audit Reports

```python
report = auditor.check(df, "ts", leaking_features)

# Write a beautiful standalone HTML report
with open("audit_report.html", "w") as f:
    f.write(report.to_html())
```

The HTML report includes:
- Leakage score with a visual progress bar
- Per-column severity badges (LOW / MEDIUM / HIGH / CRITICAL)
- Effect size, mean |Δ|, max |Δ|, % rows changed
- First timestamp where each leak was observed
- Provenance hints describing likely causes

---

## API Reference

### `TemporalAudit`

```python
TemporalAudit(
    mode: Literal["noise", "sign_flip", "nullify"] = "noise",
    random_seed: int = 42,
    delta_threshold: float = 1e-8,
    leakage_threshold: float = 0.0,
    ignore_columns: list[str] | None = None,
)
```

| Parameter | Description |
|---|---|
| `mode` | Perturbation strategy: **noise** adds Gaussian noise, **sign_flip** multiplies by -1, **nullify** sets NaN |
| `random_seed` | Integer seed — fully deterministic, reproducible across runs |
| `delta_threshold` | Minimum cell-level change to count as "different" (suppresses float noise) |
| `leakage_threshold` | If `leakage_score > leakage_threshold`, raise `TemporalLeakageError`. Set to `1.1` to always return report |
| `ignore_columns` | List of output columns to skip during comparison |

### `AuditReport`

```python
@dataclass
class AuditReport:
    leakage_score:     float          # 0.0 = clean, 1.0 = fully compromised
    breached_columns:  list[ColumnLeakMeta]
    clean_columns:     list[str]
    perturbation_mode: str
    evaluation_time:   Any
    random_seed:       int
    provenance_hints:  dict[str, str]

    def to_html(self) -> str: ...     # standalone HTML report
```

### `ColumnLeakMeta`

```python
@dataclass(frozen=True)
class ColumnLeakMeta:
    column_name:          str
    first_leaky_timestamp: Any
    mean_absolute_delta:  float
    max_delta:            float
    pct_rows_changed:     float
    effect_size:          float    # normalised, 0–1
    severity:             str      # LOW | MEDIUM | HIGH | CRITICAL
```

### Severity Classification

| Severity | Effect Size |
|---|---|
| 🟦 LOW | `effect_size < 0.15` |
| 🟨 MEDIUM | `0.15 ≤ effect_size < 0.40` |
| 🟧 HIGH | `0.40 ≤ effect_size < 0.75` |
| 🟥 CRITICAL | `effect_size ≥ 0.75` |

---

## Perturbation Modes

```
┌────────────────┬──────────────────────────────────────────────────────┐
│ Mode           │ What it does to future rows                          │
├────────────────┼──────────────────────────────────────────────────────┤
│ noise          │ Adds Gaussian noise: μ=0, σ=2×column_std             │
│ sign_flip      │ Multiplies all numeric values by −1                  │
│ nullify        │ Replaces all values with NaN / null                  │
└────────────────┴──────────────────────────────────────────────────────┘
```

Use **nullify** for the strictest test.  
Use **noise** for pipelines that handle NaN gracefully (e.g., imputers).  
Use **sign_flip** to test pipelines sensitive to sign changes (e.g., momentum factors).

---

## Polars Support

```python
import polars as pl
from temporal_leaks import TemporalAudit

df = pl.DataFrame({"ts": range(200), "value": [float(i) for i in range(200)]})

auditor = TemporalAudit(mode="nullify", random_seed=42)
report  = auditor.check(df, "ts", my_polars_pipeline)
```

`temporal-leaks` handles Polars DataFrames transparently — pass them in, get results back in the same type.

---

## Benchmarks

| Dataset | Rows | Columns | Backend | Mode | Time |
|---|---|---|---|---|---|
| Synthetic prices | 1,000,000 | 5 | Polars | nullify | ~1.1 s |
| Synthetic prices | 10,000,000 | 5 | Polars | nullify | ~3.2 s |
| Equity features | 500,000 | 20 | Pandas | noise | ~2.8 s |

> Benchmarks run on Apple M2 Pro, 16 GB RAM. Polars backend strongly recommended for large frames.

---

## Running Tests

```bash
# Install dev extras
pip install -e ".[dev]"

# Run the full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=temporal_leaks --cov-report=term-missing
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push and open a PR

Please make sure `ruff check .` and `mypy temporal_leaks/` pass before submitting.

---

## License

MIT © temporal-leaks contributors
