import numpy as np
import pandas as pd

from quantlab.engine.backtest import backtest_positions_account_margin
from quantlab.engine.trades import extract_trade_log


def test_extract_trade_log_counts_segments():
    idx = pd.date_range("2026-01-01", periods=12, freq="5min")
    prices = pd.Series(np.linspace(100, 101, len(idx)), index=idx)

    # Two trades: [1,1] and [2,2,2]
    pos = pd.Series([0, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0], index=idx, dtype=float)

    bt = backtest_positions_account_margin(prices=prices, positions_size=pos, lag=0, use_numba=False)
    log = extract_trade_log(bt)

    assert len(log) == 2
    assert list(log["trade_id"]) == [1, 2]
    assert bool(log.loc[0, "open"]) is False
    assert bool(log.loc[1, "open"]) is False


def test_extract_trade_log_open_trade():
    idx = pd.date_range("2026-01-01", periods=8, freq="5min")
    prices = pd.Series(np.linspace(100, 101, len(idx)), index=idx)

    # One open trade (never exits)
    pos = pd.Series([0, 1, 1, 1, 1, 1, 1, 1], index=idx, dtype=float)

    bt = backtest_positions_account_margin(prices=prices, positions_size=pos, lag=0, use_numba=False)
    log = extract_trade_log(bt)

    assert len(log) == 1
    assert bool(log.loc[0, "open"]) is True
