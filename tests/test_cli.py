"""
tests/test_cli.py
~~~~~~~~~~~~~~~~~

Tests for the CLI entry point.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from temporal_leaks.cli import main


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ts": np.arange(100), "value": rng.normal(10, 2, size=100)})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_check_clean_file_exits_zero(csv_file: Path) -> None:
    code = main(["check", "--file", str(csv_file), "--timestamp-col", "ts",
                 "--mode", "nullify", "--threshold", "1.1"])
    assert code == 0


def test_cli_check_missing_file_exits_nonzero(tmp_path: Path) -> None:
    code = main(["check", "--file", str(tmp_path / "nope.csv"), "--timestamp-col", "ts"])
    assert code == 2


def test_cli_check_json_flag(csv_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["check", "--file", str(csv_file), "--timestamp-col", "ts",
                 "--threshold", "1.1", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    import json
    payload = json.loads(captured.out)
    assert "leakage_score" in payload


def test_cli_html_output(csv_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    code = main(["check", "--file", str(csv_file), "--timestamp-col", "ts",
                 "--threshold", "1.1", "--output", str(out)])
    assert code == 0
    assert out.exists()
    content = out.read_text()
    assert "temporal-leaks" in content
