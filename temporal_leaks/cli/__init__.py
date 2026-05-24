"""
temporal_leaks.cli
~~~~~~~~~~~~~~~~~~

Command-line interface for temporal-leaks.

Usage
-----
    temporal-leaks check --file data.csv --timestamp-col ts --mode nullify
    temporal-leaks check --file data.parquet --timestamp-col date --output report.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from temporal_leaks.auditor import PerturbationMode, TemporalAudit
from temporal_leaks.exceptions import TemporalLeakageError

logger = logging.getLogger("temporal_leaks.cli")


def _load_file(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet file into a Pandas DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Use .csv or .parquet.")


def _identity_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Default pipeline: return the input frame as-is."""
    return df.copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="temporal-leaks",
        description="Valgrind for Time-Series ML — detect look-ahead bias in your data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Run temporal audit on a data file.")
    check.add_argument("--file", "-f", required=True, type=Path, help="Path to CSV or Parquet file.")
    check.add_argument(
        "--timestamp-col", "-t", required=True, help="Name of the timestamp column."
    )
    check.add_argument(
        "--mode",
        "-m",
        choices=["noise", "sign_flip", "nullify"],
        default="noise",
        help="Perturbation mode (default: noise).",
    )
    check.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    check.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Leakage threshold to trigger error (default: 0.0).",
    )
    check.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write HTML report.",
    )
    check.add_argument("--json", action="store_true", help="Print JSON report to stdout.")
    check.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if args.command == "check":
        try:
            df = _load_file(args.file)
        except Exception as exc:
            print(f"ERROR: Failed to load file: {exc}", file=sys.stderr)
            return 2

        auditor = TemporalAudit(
            mode=args.mode,
            random_seed=args.seed,
            leakage_threshold=args.threshold,
        )

        try:
            report = auditor.check(df, timestamp_col=args.timestamp_col, pipeline_fn=_identity_pipeline)
        except TemporalLeakageError as exc:
            print(str(exc), file=sys.stderr)
            if args.json:
                # Build a minimal JSON from the exception
                import json
                payload = {
                    "status": "LEAKED",
                    "leakage_score": exc.leakage_score,
                    "breached_columns": [m.column_name for m in exc.breached_columns],
                }
                print(json.dumps(payload, indent=2))
            return 1

        if args.json:
            print(report.to_json())
        else:
            print(report)

        if args.output:
            args.output.write_text(report.to_html(), encoding="utf-8")
            print(f"\nHTML report written to: {args.output}")

    return 0


def cli_entry() -> None:
    sys.exit(main())
