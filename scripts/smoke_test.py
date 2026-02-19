"""Smoke test for the unified backtest engine."""

from __future__ import annotations

import pandas as pd

from quantlab import performance_summary, positions_from_signal
from quantlab.backtest import backtest_positions_account_margin


def main():
    idx = pd.date_range("2026-01-01", periods=50, freq="D", tz="UTC")
    prices = pd.Series(100.0 + (pd.Series(range(len(idx))).astype(float).values * 0.2), index=idx)

    # simple always-long after warmup
    sig = pd.Series(0.0, index=idx)
    sig.iloc[10:] = 1.0
    pos = positions_from_signal(sig, lag=1, clip=1.0)

    bt = backtest_positions_account_margin(
        prices=prices,
        positions_size=pos,
        initial_capital=1000.0,
        leverage=None,
        lot_per_size=1.0,
        contract_size_per_lot=1.0,
        fee_bps=1.0,
        slippage_bps=1.0,
        lag=0,  # positions_from_signal already shifted
        max_size=1.0,
        discrete_sizes=(0.0, 1.0),
        margin_policy="allow_negative",
    )

    print(performance_summary(bt))


if __name__ == "__main__":
    main()
