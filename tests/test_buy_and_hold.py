import numpy as np
import pandas as pd

from quantlab.strategies.buy_and_hold import BuyAndHoldStrategy
from quantlab.strategies.base import BacktestConfig


def test_buy_and_hold_makes_money_on_uptrend():
    # Simple deterministic uptrend
    idx = pd.date_range("2026-01-01", periods=50, freq="5min")
    prices = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)

    cfg = BacktestConfig(
        initial_capital=1000.0,
        leverage=1.0,  # unlevered baseline (matches runner semantics)
        lot_per_size=0.01,
        contract_size_per_lot=100.0,
        lag=0,
        fee_per_lot=0.0,
        spread_per_lot=0.0,
    )

    res = BuyAndHoldStrategy().run_backtest(prices, config=cfg)

    # Should be profitable in a monotonic uptrend
    assert res.final_equity > cfg.initial_capital

    # Exactly one trade segment for buy & hold (entry then stays in)
    assert res.trade_count == 1

    # Trade log present
    assert isinstance(res.trades, list)
    assert len(res.trades) == 1
    t0 = res.trades[0]
    assert "trade_return" in t0
    assert t0["trade_return"] > 0
