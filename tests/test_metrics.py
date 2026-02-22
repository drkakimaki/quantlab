import pandas as pd

from quantlab.engine.metrics import max_drawdown, sharpe


def test_max_drawdown_simple():
    e = pd.Series([1.0, 1.1, 1.0, 1.2, 0.9, 1.3])
    dd = max_drawdown(e)
    assert dd < 0


def test_sharpe_constant_is_nan():
    # Flat equity => 0 daily returns => std=0 => Sharpe is undefined (nan)
    idx = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    eq = pd.Series([100.0] * 10, index=idx)
    assert pd.isna(sharpe(eq))
