import pandas as pd

from quantlab.strategies.gates.filters import MonthFlatGate


def test_month_flat_gate_blocks_june() -> None:
    idx = pd.date_range("2026-05-31", periods=4, freq="1D")
    # 2026-05-31, 2026-06-01, 2026-06-02, 2026-06-03
    pos = pd.Series([1.0, 1.0, 2.0, 1.0], index=idx)

    g = MonthFlatGate(months=[6])
    out = g(pos, prices=pd.Series([100.0] * len(idx), index=idx), context=None)

    assert out.tolist() == [1.0, 0.0, 0.0, 0.0]
