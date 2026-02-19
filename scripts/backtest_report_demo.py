"""Demo: basic backtest + report.

This demo uses the unified account-style backtest engine.
"""

from __future__ import annotations

import pandas as pd

from quantlab import ReportConfig, positions_from_signal, prices_to_returns, write_html_report
from quantlab.backtest import backtest_positions_account_margin


def main():
    idx = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    prices = pd.Series(100.0 + (pd.Series(range(len(idx))).astype(float).values * 0.1), index=idx)

    # toy signal: long after day 30
    sig = (prices.index >= prices.index[30]).astype(float)
    pos = positions_from_signal(pd.Series(sig, index=prices.index), lag=1, clip=1.0)

    bt = backtest_positions_account_margin(
        prices=prices,
        positions_size=pos,
        initial_capital=1000.0,
        leverage=None,
        lot_per_size=1.0,
        contract_size_per_lot=1.0,
        fee_per_lot=0.0,
        spread_per_lot=0.0,
        lag=0,  # already lagged by positions_from_signal
        max_size=1.0,
        margin_policy="allow_negative",
    )

    cfg = ReportConfig(title="Demo backtest", subtitle="Unified engine")
    write_html_report(bt, cfg, out_path="reports/demo_backtest.html")
    print("Wrote: reports/demo_backtest.html")


if __name__ == "__main__":
    main()
