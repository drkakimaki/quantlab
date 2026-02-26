from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.reporting.generate_equity_report import report_periods_equity_only


def _toy_bt() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=40, freq="5min")

    # Small random-ish walk, deterministic.
    prices = 100.0 + np.linspace(0.0, 1.0, len(idx))

    # One continuous long position segment.
    position = pd.Series([0.0] + [1.0] * (len(idx) - 1), index=idx)

    # Simple net returns: flat first bar, then constant small positive.
    r = pd.Series([0.0] + [0.0002] * (len(idx) - 1), index=idx)

    equity = (1000.0 * (1.0 + r).cumprod()).astype(float)

    return pd.DataFrame({"position": position, "returns_net": r, "equity": equity}, index=idx)


def test_bt_report_smoke_writes_html(tmp_path: Path) -> None:
    bt = _toy_bt()

    out = report_periods_equity_only(
        periods={"P": bt},
        out_path=tmp_path / "bt.html",
        title="Toy BT",
        initial_capital=1000.0,
    )

    assert out.exists()
    html = out.read_text(encoding="utf-8")

    # Basic sanity: title + key table headers.
    assert "Toy BT" in html
    assert "Sharpe" in html
    assert "Max DD" in html
    assert "PnL" in html


def test_bt_report_empty_period_does_not_crash(tmp_path: Path) -> None:
    out = report_periods_equity_only(
        periods={"EMPTY": pd.DataFrame()},
        out_path=tmp_path / "bt_empty.html",
        title="Empty",
        initial_capital=1000.0,
    )

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Empty" in html
    assert "EMPTY" in html
