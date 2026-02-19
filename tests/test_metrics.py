import pandas as pd

from quantlab.metrics import max_drawdown, sharpe


def test_max_drawdown_simple():
    e = pd.Series([1.0, 1.1, 1.0, 1.2, 0.9, 1.3])
    dd = max_drawdown(e)
    assert dd < 0


def test_sharpe_constant_is_nan():
    r = pd.Series([0.0] * 10)
    assert pd.isna(sharpe(r))
