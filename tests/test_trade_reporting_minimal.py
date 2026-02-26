from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.engine.trades import build_trade_ledger
from quantlab.reporting.generate_trades_report import report_periods_trades_html


def _toy_bt_two_trades() -> pd.DataFrame:
    # 12 bars of 5min. Two trades: long for 3 bars, flat, short for 2 bars.
    idx = pd.date_range("2025-01-01", periods=12, freq="5min")

    position = [0, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0]

    # Keep return math simple (small returns during in-position bars).
    returns_net = [0.0] * 12
    returns_net[1:4] = [0.001, 0.001, -0.0005]  # trade 1
    returns_net[6:8] = [-0.001, 0.002]  # trade 2

    equity = [1000.0]
    for r in returns_net[1:]:
        equity.append(equity[-1] * (1.0 + r))

    costs = [0.0] * 12
    costs[1] = 0.2
    costs[6] = 0.2

    return pd.DataFrame(
        {
            "position": position,
            "returns_net": returns_net,
            "equity": equity,
            "costs": costs,
        },
        index=idx,
    )


def test_build_trade_ledger_basic() -> None:
    bt = _toy_bt_two_trades()
    tl = build_trade_ledger(bt)

    assert not tl.empty
    assert len(tl) == 2

    # Stable core columns
    for col in [
        "trade_id",
        "entry_time",
        "exit_time",
        "bars",
        "side",
        "entry_equity",
        "exit_equity",
        "pnl_net",
        "trade_return",
        "costs_total",
        "win",
        "open",
    ]:
        assert col in tl.columns

    # One long, one short
    assert set(tl["side"].tolist()) == {"long", "short"}


# (CSV trade breakdown artifacts removed)


def test_report_periods_trades_html_no_trades(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="5min")
    bt = pd.DataFrame({"position": 0.0, "returns_net": 0.0, "equity": 1000.0}, index=idx)

    out = report_periods_trades_html(periods={"P": bt}, out_path=tmp_path / "r.html", title="X")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "No trades" in html


def test_report_periods_trades_html_smoke(tmp_path: Path) -> None:
    bt = _toy_bt_two_trades()
    out = report_periods_trades_html(periods={"P": bt}, out_path=tmp_path / "r2.html", title="Toy")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Toy" in html
    # Basic sanity that it rendered some table markup.
    assert "<table" in html
