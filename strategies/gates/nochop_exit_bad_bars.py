from __future__ import annotations

import pandas as pd

from .types import SignalGate
from .registry import register_gate


@register_gate("nochop_exit_bad_bars")
class NoChopExitBadBarsGate:
    """Post-processing NoChop exit gate.

    This gate recomputes the same `nochop_ok_base` mask as NoChopGate and then
    force-flats a segment once it sees `exit_bad_bars` consecutive bad bars while
    in-position.

    Intended placement: after other filters like time_filter.
    """

    def __init__(
        self,
        *,
        ema: int = 20,
        lookback: int = 40,
        min_closes: int = 24,
        exit_bad_bars: int = 0,
    ):
        self.ema = int(ema)
        self.lookback = int(lookback)
        self.min_closes = int(min_closes)
        self.exit_bad_bars = int(exit_bad_bars)

    @property
    def name(self) -> str:
        return f"NoChopExitBad(streak={self.exit_bad_bars})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if self.exit_bad_bars <= 0:
            return positions

        if context is None or "bars_15m" not in context:
            return positions

        htf_close = context["bars_15m"]["close"].astype(float).dropna()
        htf_close.index = pd.to_datetime(htf_close.index)

        ema_nc = htf_close.ewm(span=self.ema, adjust=False).mean()
        above = (htf_close > ema_nc).astype(int)
        above_cnt = above.rolling(self.lookback, min_periods=self.lookback).sum()
        nochop_ok = (above_cnt >= self.min_closes).astype(bool)
        nochop_ok_base = nochop_ok.reindex(positions.index).ffill().fillna(False)

        gate_on = positions.fillna(0.0).astype(float) > 0.0
        seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()

        bad_in_seg = (~nochop_ok_base) & gate_on
        grp = (~bad_in_seg).cumsum()
        streak = bad_in_seg.astype(int).groupby(grp).cumsum()
        trigger = streak >= int(self.exit_bad_bars)
        seg_kill = trigger.groupby(seg).cummax()

        return positions.where(~seg_kill, 0.0)
