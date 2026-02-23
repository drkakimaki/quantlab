import pandas as pd

from quantlab.strategies.gates.filters import TimeFilterGate


def test_time_filter_gate_uses_context_mask_when_self_mask_missing() -> None:
    idx = pd.date_range("2026-01-01", periods=6, freq="5min")
    pos = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], index=idx)

    # Block bars 2 and 3 (0-based), allow others.
    allow_mask = pd.Series([True, True, False, False, True, True], index=idx)

    g = TimeFilterGate(allow_mask=None)
    out = g(pos, prices=pd.Series([100.0] * len(idx), index=idx), context={"allow_mask": allow_mask})

    assert out.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
