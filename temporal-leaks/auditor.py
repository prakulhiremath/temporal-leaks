"""
temporal_leaks.auditor
~~~~~~~~~~~~~~~~~~~~~~

Core engine for the Temporal Perturbation Test.

Design
------
The algorithm works as follows:

1. Accept a Pandas or Polars DataFrame, a ``timestamp_col``, and a
   user-supplied ``pipeline_fn`` that maps a DataFrame to a DataFrame
   containing computed feature columns.
2. Run the pipeline once to establish a **baseline**: ``baseline_df = pipeline_fn(df)``.
3. Select a critical evaluation time **T** (the temporal midpoint by default).
4. Clone the input data and **perturb** all non-timestamp columns for every
   row where ``timestamp > T`` using one of three strategies:
       - ``"noise"``      – add Gaussian noise scaled to the column's std-dev.
       - ``"sign_flip"``  – multiply numeric values by -1.
       - ``"nullify"``    – replace values with NaN / null.
5. Run the perturbed data through the pipeline: ``perturbed_df = pipeline_fn(perturbed_df)``.
6. Compare ``baseline_df`` and ``perturbed_df`` for rows where
   ``timestamp <= T``.  If *any* feature values for past rows changed after
   we mutated the future, temporal leakage is present.
"""

from __future__ import annotations

import copy
import functools
import html
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from temporal_leaks.exceptions import ColumnLeakMeta, TemporalLeakageError

# ---------------------------------------------------------------------------
# Optional Polars import – the library works without it, but it is preferred
# for large-frame operations.
# ---------------------------------------------------------------------------
try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _POLARS_AVAILABLE = False
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PerturbationMode = Literal["noise", "sign_flip", "nullify"]
DataFrame = Union[pd.DataFrame, "pl.DataFrame"]
PipelineFn = Callable[[DataFrame], DataFrame]

_SEVERITY_THRESHOLDS: list[tuple[float, str]] = [
    (0.75, "CRITICAL"),
    (0.40, "HIGH"),
    (0.15, "MEDIUM"),
    (0.0, "LOW"),
]


def _classify_severity(effect_size: float) -> str:
    """Return a severity label based on *effect_size* (0–1)."""
    for threshold, label in _SEVERITY_THRESHOLDS:
        if effect_size >= threshold:
            return label
    return "LOW"


# ---------------------------------------------------------------------------
# Audit Report
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    """
    The result of a single :meth:`TemporalAudit.check` run.

    Attributes
    ----------
    leakage_score:
        Aggregate score in ``[0.0, 1.0]``.  0 means no leakage detected;
        1 means every inspected feature column is leaking at maximum severity.
    breached_columns:
        Metadata objects, one per leaking column, sorted by *effect_size*
        descending.
    clean_columns:
        Names of columns that passed the test.
    perturbation_mode:
        The perturbation strategy that was used.
    evaluation_time:
        The critical time T that was used to split past/future.
    random_seed:
        The seed used for reproducible perturbation.
    provenance_hints:
        Primitive tracing hints: a mapping from each breached column name to a
        short natural-language description of likely leakage cause.
    """

    leakage_score: float
    breached_columns: list[ColumnLeakMeta]
    clean_columns: list[str]
    perturbation_mode: PerturbationMode
    evaluation_time: Any
    random_seed: int
    provenance_hints: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "LEAKED" if self.breached_columns else "CLEAN"
        return (
            f"AuditReport("
            f"status={status!r}, "
            f"leakage_score={self.leakage_score:.4f}, "
            f"breached={[c.column_name for c in self.breached_columns]!r}, "
            f"mode={self.perturbation_mode!r}, "
            f"seed={self.random_seed!r}"
            f")"
        )

    def __str__(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║          temporal-leaks  ·  Audit Report             ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  Leakage Score   : {self.leakage_score:.4f}",
            f"  Status          : {'⚠  LEAKAGE DETECTED' if self.breached_columns else '✓  CLEAN'}",
            f"  Perturbation    : {self.perturbation_mode}",
            f"  Evaluation Time : {self.evaluation_time}",
            f"  Random Seed     : {self.random_seed}",
            "",
        ]
        if self.breached_columns:
            lines.append("  Breached Columns:")
            for meta in self.breached_columns:
                lines.append(f"    [{meta.severity}] {meta.column_name}")
                lines.append(f"        effect_size      = {meta.effect_size:.4f}")
                lines.append(f"        mean_abs_delta   = {meta.mean_absolute_delta:.6f}")
                lines.append(f"        max_delta        = {meta.max_delta:.6f}")
                lines.append(f"        rows_changed     = {meta.pct_rows_changed:.1%}")
                lines.append(f"        first_leak_at    = {meta.first_leaky_timestamp}")
                if meta.column_name in self.provenance_hints:
                    lines.append(
                        f"        provenance_hint  = {self.provenance_hints[meta.column_name]}"
                    )
            lines.append("")
        if self.clean_columns:
            lines.append(f"  Clean Columns: {', '.join(self.clean_columns)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        """
        Generate a self-contained HTML report.

        Returns
        -------
        str
            A fully standalone HTML string (no external dependencies) that
            can be written directly to a ``.html`` file.

        Examples
        --------
        >>> report = auditor.check(df, "ts", pipeline)
        >>> with open("audit_report.html", "w") as f:
        ...     f.write(report.to_html())
        """
        status_color = "#e74c3c" if self.breached_columns else "#27ae60"
        status_text = "⚠ LEAKAGE DETECTED" if self.breached_columns else "✓ CLEAN"
        score_pct = int(self.leakage_score * 100)

        breached_rows = ""
        for meta in self.breached_columns:
            sev_colors = {
                "CRITICAL": "#c0392b",
                "HIGH": "#e67e22",
                "MEDIUM": "#f1c40f",
                "LOW": "#3498db",
            }
            sev_color = sev_colors.get(meta.severity, "#95a5a6")
            hint = html.escape(self.provenance_hints.get(meta.column_name, "—"))
            breached_rows += f"""
            <tr>
              <td><code>{html.escape(meta.column_name)}</code></td>
              <td><span class="badge" style="background:{sev_color}">{html.escape(meta.severity)}</span></td>
              <td>{meta.effect_size:.4f}</td>
              <td>{meta.mean_absolute_delta:.6f}</td>
              <td>{meta.max_delta:.6f}</td>
              <td>{meta.pct_rows_changed:.1%}</td>
              <td><code>{html.escape(str(meta.first_leaky_timestamp))}</code></td>
              <td>{hint}</td>
            </tr>"""

        clean_pills = "".join(
            f'<span class="pill">{html.escape(c)}</span>' for c in self.clean_columns
        )

        return textwrap.dedent(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>temporal-leaks Audit Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f0f0f; color: #e8e8e8; margin: 0; padding: 2rem;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; color: #fff; }}
    h2 {{ font-size: 1.1rem; color: #aaa; margin: 1.5rem 0 0.75rem; border-bottom: 1px solid #333; padding-bottom: 0.4rem; }}
    .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 2rem; }}
    .card {{
      background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px;
      padding: 1.5rem; margin-bottom: 1.5rem;
    }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }}
    .meta-item {{ background: #222; border-radius: 8px; padding: 0.75rem 1rem; }}
    .meta-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #888; margin-bottom: 0.25rem; }}
    .meta-value {{ font-size: 1.1rem; font-weight: 600; color: #fff; }}
    .score-bar-wrap {{ background: #222; border-radius: 99px; height: 12px; margin-top: 0.5rem; overflow: hidden; }}
    .score-bar {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, #27ae60, #f39c12, #e74c3c); width: {score_pct}%; transition: width 0.6s; }}
    .status-badge {{
      display: inline-block; font-size: 1rem; font-weight: 700;
      color: {status_color}; border: 2px solid {status_color};
      border-radius: 6px; padding: 0.25rem 0.75rem;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th {{ text-align: left; padding: 0.6rem 0.75rem; color: #888; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; border-bottom: 1px solid #333; }}
    td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #222; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #1e1e1e; }}
    .badge {{ display: inline-block; border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.75rem; font-weight: 700; color: #fff; }}
    .pill {{ display: inline-block; background: #1e3a2f; color: #2ecc71; border-radius: 99px; padding: 0.2rem 0.65rem; font-size: 0.8rem; margin: 0.15rem; }}
    code {{ background: #252525; border-radius: 4px; padding: 0.1rem 0.35rem; font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 0.85em; }}
    footer {{ margin-top: 2rem; text-align: center; color: #555; font-size: 0.8rem; }}
    a {{ color: #5dade2; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>🕵️ temporal-leaks</h1>
  <p class="subtitle">Audit Report · generated {html.escape(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))} UTC</p>

  <div class="card">
    <div class="meta-grid">
      <div class="meta-item">
        <div class="meta-label">Status</div>
        <div class="meta-value"><span class="status-badge">{html.escape(status_text)}</span></div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Leakage Score</div>
        <div class="meta-value">{self.leakage_score:.4f}</div>
        <div class="score-bar-wrap"><div class="score-bar"></div></div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Perturbation Mode</div>
        <div class="meta-value"><code>{html.escape(self.perturbation_mode)}</code></div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Evaluation Time T</div>
        <div class="meta-value"><code>{html.escape(str(self.evaluation_time))}</code></div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Random Seed</div>
        <div class="meta-value"><code>{self.random_seed}</code></div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Breached Columns</div>
        <div class="meta-value" style="color:#e74c3c">{len(self.breached_columns)}</div>
      </div>
    </div>
  </div>

  {"" if not self.breached_columns else f'''
  <div class="card">
    <h2>Breached Columns</h2>
    <table>
      <thead>
        <tr>
          <th>Column</th><th>Severity</th><th>Effect Size</th>
          <th>Mean |Δ|</th><th>Max |Δ|</th><th>Rows Changed</th>
          <th>First Leak At</th><th>Provenance Hint</th>
        </tr>
      </thead>
      <tbody>{breached_rows}</tbody>
    </table>
  </div>'''}

  {"" if not self.clean_columns else f'''
  <div class="card">
    <h2>Clean Columns</h2>
    <p>{clean_pills}</p>
  </div>'''}

  <footer>
    Generated by <a href="https://github.com/temporal-leaks/temporal-leaks">temporal-leaks</a>
    — Valgrind for Time-Series ML
  </footer>
</body>
</html>""")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_pandas(df: DataFrame) -> pd.DataFrame:
    """Coerce a Polars or Pandas DataFrame to Pandas."""
    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def _restore_type(original: DataFrame, pandas_df: pd.DataFrame) -> DataFrame:
    """Return the same type as *original*."""
    if _POLARS_AVAILABLE and isinstance(original, pl.DataFrame):
        return pl.from_pandas(pandas_df)
    return pandas_df


def _perturb_pandas(
    df: pd.DataFrame,
    timestamp_col: str,
    cutoff: Any,
    mode: PerturbationMode,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Return a copy of *df* with all non-timestamp columns perturbed for rows
    where ``timestamp_col > cutoff``.

    Parameters
    ----------
    df:
        Input frame (Pandas).
    timestamp_col:
        Name of the datetime/ordinal column.
    cutoff:
        The critical time T.
    mode:
        Perturbation strategy.
    rng:
        A seeded NumPy random generator for full reproducibility.
    """
    out = df.copy()
    future_mask = out[timestamp_col] > cutoff
    feature_cols = [c for c in out.columns if c != timestamp_col]

    for col in feature_cols:
        col_data = out.loc[future_mask, col]
        if not pd.api.types.is_numeric_dtype(col_data):
            # For non-numeric columns use nullify regardless of mode
            out.loc[future_mask, col] = np.nan
            continue

        if mode == "noise":
            std = float(col_data.std(skipna=True)) or 1.0
            noise = rng.normal(loc=0.0, scale=std * 2.0, size=future_mask.sum())
            out.loc[future_mask, col] = col_data.values + noise
        elif mode == "sign_flip":
            out.loc[future_mask, col] = col_data.values * -1.0
        elif mode == "nullify":
            out.loc[future_mask, col] = np.nan
        else:
            raise ValueError(f"Unknown perturbation mode: {mode!r}")

    return out


def _compute_column_meta(
    col: str,
    baseline: pd.Series,  # type: ignore[type-arg]
    perturbed: pd.Series,  # type: ignore[type-arg]
    timestamps: pd.Series,  # type: ignore[type-arg]
    delta_threshold: float,
) -> Optional[ColumnLeakMeta]:
    """
    Compute leakage metadata for a single column.

    Returns
    -------
    ColumnLeakMeta | None
        ``None`` when the column is clean.
    """
    b = pd.to_numeric(baseline, errors="coerce")
    p = pd.to_numeric(perturbed, errors="coerce")

    delta = (b - p).abs()
    changed_mask = delta > delta_threshold
    pct_changed = float(changed_mask.mean())

    if pct_changed == 0.0:
        return None

    mean_delta = float(delta.mean(skipna=True))
    max_delta = float(delta.max(skipna=True))
    baseline_std = float(b.std(skipna=True)) or 1e-9
    effect_size = min(mean_delta / (baseline_std + 1e-9), 1.0)

    first_leaky_ts = timestamps[changed_mask].iloc[0] if changed_mask.any() else None

    return ColumnLeakMeta(
        column_name=col,
        first_leaky_timestamp=first_leaky_ts,
        mean_absolute_delta=mean_delta,
        max_delta=max_delta,
        pct_rows_changed=pct_changed,
        effect_size=effect_size,
        severity=_classify_severity(effect_size),
    )


def _infer_provenance(col: str, effect_size: float) -> str:
    """
    Heuristic provenance hint.  In a future version this would trace the
    call graph; for now it returns a rule-based description based on the
    column name and effect size.
    """
    hints = []
    col_lower = col.lower()

    if any(k in col_lower for k in ("roll", "window", "ma", "ewm", "ewma")):
        hints.append(
            "Rolling/window aggregation detected — verify `min_periods` "
            "and that the window is anchored to the past (shift ≥ 1)."
        )
    if any(k in col_lower for k in ("lag", "shift", "diff")):
        hints.append(
            "Lag/diff operation detected — ensure shift direction is positive "
            "(backward-looking) and not negative (look-ahead)."
        )
    if any(k in col_lower for k in ("rank", "pct", "percentile", "quantile")):
        hints.append(
            "Rank/percentile computation detected — cross-sectional ranks "
            "computed on the full dataset will leak future information."
        )
    if any(k in col_lower for k in ("norm", "scale", "zscore", "std", "mean")):
        hints.append(
            "Normalisation/scaling detected — fitting a scaler on the full "
            "dataset (including future rows) leaks distributional information."
        )
    if not hints:
        if effect_size >= 0.75:
            hints.append(
                "CRITICAL: Column is highly sensitive to future data; "
                "likely computed from the full dataset without time-aware splitting."
            )
        else:
            hints.append(
                "Feature computation may be directly or indirectly dependent "
                "on rows with timestamps beyond the evaluation point T."
            )

    return " | ".join(hints)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TemporalAudit:
    """
    Sklearn-style auditor for look-ahead bias (temporal leakage) in
    feature-engineering pipelines.

    Parameters
    ----------
    mode:
        Perturbation strategy applied to future data.
        - ``"noise"``     – add Gaussian noise scaled to the column's std-dev.
        - ``"sign_flip"`` – multiply all numeric values by -1.
        - ``"nullify"``   – replace all future values with NaN.
    random_seed:
        Integer seed passed to :class:`numpy.random.default_rng` for fully
        deterministic perturbation.  Use the *same* seed across runs to
        guarantee reproducible results.
    delta_threshold:
        Minimum absolute difference required to count a cell as "changed".
        Helps suppress floating-point noise from trivial transformations.
        Default: ``1e-8``.
    leakage_threshold:
        If the aggregate leakage score exceeds this value,
        :meth:`check` will raise :exc:`~temporal_leaks.TemporalLeakageError`.
        Set to ``1.1`` (or any value > 1) to suppress the exception and
        always return the report.  Default: ``0.0`` (raise on any leakage).
    ignore_columns:
        Feature columns to skip during comparison (e.g., intermediate
        columns that are intentionally volatile).

    Examples
    --------
    >>> from temporal_leaks import TemporalAudit
    >>> auditor = TemporalAudit(mode="nullify", random_seed=42)
    >>> report = auditor.check(df, timestamp_col="ts", pipeline_fn=my_features)
    >>> print(report)
    """

    def __init__(
        self,
        mode: PerturbationMode = "noise",
        random_seed: int = 42,
        delta_threshold: float = 1e-8,
        leakage_threshold: float = 0.0,
        ignore_columns: Optional[list[str]] = None,
    ) -> None:
        if mode not in ("noise", "sign_flip", "nullify"):
            raise ValueError(
                f"Unknown mode {mode!r}. Choose from 'noise', 'sign_flip', 'nullify'."
            )
        self.mode: PerturbationMode = mode
        self.random_seed: int = random_seed
        self.delta_threshold: float = delta_threshold
        self.leakage_threshold: float = leakage_threshold
        self.ignore_columns: list[str] = ignore_columns or []
        self._rng: np.random.Generator = np.random.default_rng(random_seed)

    def __repr__(self) -> str:
        return (
            f"TemporalAudit("
            f"mode={self.mode!r}, "
            f"random_seed={self.random_seed!r}, "
            f"delta_threshold={self.delta_threshold!r}, "
            f"leakage_threshold={self.leakage_threshold!r}"
            f")"
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def check(
        self,
        df: DataFrame,
        timestamp_col: str,
        pipeline_fn: PipelineFn,
    ) -> AuditReport:
        """
        Run the Temporal Perturbation Test on *df*.

        Parameters
        ----------
        df:
            Input DataFrame (Pandas or Polars).
        timestamp_col:
            Name of the column containing timestamps or any sortable ordinal
            values.  Must be present in *df*.
        pipeline_fn:
            A callable that accepts a DataFrame (same type as *df*) and
            returns a DataFrame with computed feature columns.  The function
            **must** be deterministic for a given input.

        Returns
        -------
        AuditReport
            Detailed audit report.  If the leakage score exceeds
            ``self.leakage_threshold``, a
            :exc:`~temporal_leaks.TemporalLeakageError` is raised instead.

        Raises
        ------
        TemporalLeakageError
            When temporal leakage is detected above ``leakage_threshold``.
        TypeError
            When *df* is not a Pandas or Polars DataFrame.
        KeyError
            When *timestamp_col* is not present in *df*.
        """
        # Reset RNG for full reproducibility across multiple `.check()` calls
        self._rng = np.random.default_rng(self.random_seed)

        logger.info(
            "TemporalAudit.check | mode=%s seed=%d threshold=%.4f",
            self.mode,
            self.random_seed,
            self.leakage_threshold,
        )

        # ----------------------------------------------------------------
        # 1. Coerce to Pandas for internal processing
        # ----------------------------------------------------------------
        original_type = type(df)
        pandas_df = _to_pandas(df)

        if timestamp_col not in pandas_df.columns:
            raise KeyError(
                f"timestamp_col={timestamp_col!r} not found in DataFrame columns: "
                f"{list(pandas_df.columns)}"
            )

        pandas_df = pandas_df.sort_values(timestamp_col).reset_index(drop=True)
        logger.debug("Input shape: %s", pandas_df.shape)

        # ----------------------------------------------------------------
        # 2. Baseline run
        # ----------------------------------------------------------------
        logger.info("Running baseline pipeline …")
        baseline_input = _restore_type(df, pandas_df)
        baseline_output = pipeline_fn(baseline_input)
        baseline_pd = _to_pandas(baseline_output)
        logger.debug("Baseline output shape: %s", baseline_pd.shape)

        # ----------------------------------------------------------------
        # 3. Determine critical evaluation time T
        # ----------------------------------------------------------------
        timestamps = pandas_df[timestamp_col]
        cutoff = self._pick_cutoff(timestamps)
        logger.info("Evaluation time T = %s", cutoff)

        past_mask = timestamps <= cutoff
        n_past = int(past_mask.sum())
        logger.debug("Past rows (≤ T): %d  |  Future rows (> T): %d", n_past, len(pandas_df) - n_past)

        if n_past == 0:
            logger.warning("No rows in the past partition; skipping audit.")
            return AuditReport(
                leakage_score=0.0,
                breached_columns=[],
                clean_columns=[],
                perturbation_mode=self.mode,
                evaluation_time=cutoff,
                random_seed=self.random_seed,
            )

        # ----------------------------------------------------------------
        # 4. Perturb future data
        # ----------------------------------------------------------------
        logger.info("Perturbing future rows (mode=%s) …", self.mode)
        perturbed_pandas = _perturb_pandas(
            pandas_df, timestamp_col, cutoff, self.mode, self._rng
        )
        perturbed_input = _restore_type(df, perturbed_pandas)

        # ----------------------------------------------------------------
        # 5. Perturbed run
        # ----------------------------------------------------------------
        logger.info("Running perturbed pipeline …")
        perturbed_output = pipeline_fn(perturbed_input)
        perturbed_pd = _to_pandas(perturbed_output)
        logger.debug("Perturbed output shape: %s", perturbed_pd.shape)

        # ----------------------------------------------------------------
        # 6. Align and compare past-partition feature values
        # ----------------------------------------------------------------
        # We need to align both outputs on the same index.  After sort +
        # reset_index the positional index is canonical.
        baseline_past = baseline_pd.loc[past_mask].reset_index(drop=True)
        perturbed_past = perturbed_pd.loc[past_mask].reset_index(drop=True)
        timestamps_past = timestamps[past_mask].reset_index(drop=True)

        feature_cols = [
            c
            for c in baseline_past.columns
            if c != timestamp_col and c not in self.ignore_columns
        ]

        logger.info(
            "Comparing %d feature column(s) over %d past rows …",
            len(feature_cols),
            n_past,
        )

        breached: list[ColumnLeakMeta] = []
        clean: list[str] = []

        for col in feature_cols:
            if col not in perturbed_past.columns:
                logger.warning("Column %r missing from perturbed output; skipping.", col)
                continue

            meta = _compute_column_meta(
                col=col,
                baseline=baseline_past[col],
                perturbed=perturbed_past[col],
                timestamps=timestamps_past,
                delta_threshold=self.delta_threshold,
            )

            if meta is not None:
                logger.warning(
                    "LEAK detected | %s | effect_size=%.4f severity=%s",
                    col,
                    meta.effect_size,
                    meta.severity,
                )
                breached.append(meta)
            else:
                clean.append(col)

        # Sort breached columns by effect_size descending
        breached.sort(key=lambda m: m.effect_size, reverse=True)

        # ----------------------------------------------------------------
        # 7. Compute aggregate leakage score
        # ----------------------------------------------------------------
        if breached:
            leakage_score = float(np.mean([m.effect_size for m in breached]))
        else:
            leakage_score = 0.0

        # ----------------------------------------------------------------
        # 8. Provenance hints
        # ----------------------------------------------------------------
        provenance_hints: dict[str, str] = {
            m.column_name: _infer_provenance(m.column_name, m.effect_size)
            for m in breached
        }

        # ----------------------------------------------------------------
        # 9. Build report
        # ----------------------------------------------------------------
        report = AuditReport(
            leakage_score=leakage_score,
            breached_columns=breached,
            clean_columns=clean,
            perturbation_mode=self.mode,
            evaluation_time=cutoff,
            random_seed=self.random_seed,
            provenance_hints=provenance_hints,
        )

        logger.info(
            "Audit complete | leakage_score=%.4f | breached=%d | clean=%d",
            leakage_score,
            len(breached),
            len(clean),
        )

        if leakage_score > self.leakage_threshold and breached:
            raise TemporalLeakageError(
                message=(
                    f"Temporal leakage detected: score={leakage_score:.4f}, "
                    f"breached columns={[m.column_name for m in breached]}"
                ),
                leakage_score=leakage_score,
                breached_columns=breached,
            )

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_cutoff(timestamps: pd.Series) -> Any:  # type: ignore[type-arg]
        """
        Select the critical evaluation time T as the median timestamp.

        For datetime columns the midpoint between min and max is used.
        For numeric columns the numeric median is used.
        """
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            min_ts = timestamps.min()
            max_ts = timestamps.max()
            return min_ts + (max_ts - min_ts) / 2
        else:
            return timestamps.median()


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------


def temporal_audit(
    timestamp_col: str,
    mode: PerturbationMode = "noise",
    random_seed: int = 42,
    delta_threshold: float = 1e-8,
    leakage_threshold: float = 0.0,
    ignore_columns: Optional[list[str]] = None,
) -> Callable[[PipelineFn], PipelineFn]:
    """
    Decorator that automatically runs a :class:`TemporalAudit` on every call
    to the decorated pipeline function.

    The decorator injects a hidden ``_audit_report_`` attribute on the
    returned DataFrame so callers can inspect the report programmatically.
    If leakage is detected, :exc:`~temporal_leaks.TemporalLeakageError` is
    raised **before** returning the result.

    Parameters
    ----------
    timestamp_col:
        Name of the timestamp column in the input DataFrame.
    mode, random_seed, delta_threshold, leakage_threshold, ignore_columns:
        Forwarded verbatim to :class:`TemporalAudit`.

    Examples
    --------
    >>> @temporal_audit(timestamp_col="ts", mode="nullify")
    ... def build_features(df: pd.DataFrame) -> pd.DataFrame:
    ...     df = df.copy()
    ...     df["roll_mean"] = df["value"].rolling(3, min_periods=1).mean()
    ...     return df
    """

    def decorator(fn: PipelineFn) -> PipelineFn:
        auditor = TemporalAudit(
            mode=mode,
            random_seed=random_seed,
            delta_threshold=delta_threshold,
            leakage_threshold=leakage_threshold,
            ignore_columns=ignore_columns,
        )

        @functools.wraps(fn)
        def wrapper(df: DataFrame) -> DataFrame:
            # Run the audit; may raise TemporalLeakageError
            report = auditor.check(df, timestamp_col=timestamp_col, pipeline_fn=fn)
            # Run the actual pipeline for the caller
            result = fn(df)
            # Attach report metadata to the result (Pandas only)
            if isinstance(result, pd.DataFrame):
                object.__setattr__(result, "_audit_report_", report)
            return result

        wrapper._temporal_audit = auditor  # type: ignore[attr-defined]
        return wrapper

    return decorator
