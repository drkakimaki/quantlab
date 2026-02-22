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


def test_backtest_lag_shifts_position_one_bar() -> None:
    idx = pd.date_range("2026-01-01", periods=6, freq="5min")
    prices = pd.Series([100, 101, 102, 103, 104, 105], index=idx, dtype=float)

    # Signal says: enter from t1 onward.
    pos = pd.Series([0, 1, 1, 1, 1, 1], index=idx, dtype=float)

    bt_lag0 = backtest_positions_account_margin(
        prices=prices,
        positions_size=pos,
        lag=0,
        fee_per_lot=0.0,
        spread_per_lot=0.0,
        use_numba=False,
    )
    bt_lag1 = backtest_positions_account_margin(
        prices=prices,
        positions_size=pos,
        lag=1,
        fee_per_lot=0.0,
        spread_per_lot=0.0,
        use_numba=False,
    )

    # With lag=1, the position series is shifted forward by one bar.
    assert bt_lag1["position"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert bt_lag0["position"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # And pnl starts accruing one bar later (when position becomes active).
    assert bt_lag0["pnl"].iloc[2] > 0.0  # units held over price change 101->102
    assert bt_lag1["pnl"].iloc[2] == 0.0
    assert bt_lag1["pnl"].iloc[3] > 0.0
