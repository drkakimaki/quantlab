import numpy as np
import pandas as pd

from quantlab.engine.backtest import backtest_positions_account_margin


def test_backtest_numba_matches_python():
    # deterministic input
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=2000, freq="5min")
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.05, size=len(idx))), index=idx)
    pos = pd.Series(rng.choice([0.0, 1.0, 2.0], size=len(idx)), index=idx)

    bt_py = backtest_positions_account_margin(
        prices=prices,
        positions_size=pos,
        lag=0,
        fee_per_lot=3.5,
        spread_per_lot=7.0,
        use_numba=False,
    )
    bt_nb = backtest_positions_account_margin(
        prices=prices,
        positions_size=pos,
        lag=0,
        fee_per_lot=3.5,
        spread_per_lot=7.0,
        use_numba=True,
    )

    cols = ["equity", "pnl", "costs", "contract_units"]
    diff = (bt_py[cols] - bt_nb[cols]).abs().to_numpy().max()
    assert diff == 0.0
