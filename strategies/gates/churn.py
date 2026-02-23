from __future__ import annotations

import pandas as pd

from .types import SignalGate
from .registry import register_gate


@register_gate("churn")
class ChurnGate:
    """Churn-reduction gate (signal debouncing + re-entry cooldown).

    This is a *post-processing* gate: it only transforms an existing position
    size series (0/1/2). It does not create entries.

    Features
    --------
    1) Entry persistence / debounce (min_on_bars)
       Delay entry until the raw signal has been ON for N consecutive bars.
       (Chops off the first N-1 bars of each segment.)

    2) Re-entry cooldown (cooldown_bars)
       After an exit (ON -> OFF), block new entries for the next C bars.

    Notes
    -----
    - Long-only is assumed by upstream (positions > 0).
    - This gate is designed to attack the "churn is toxic" fingerprint.
    """

    def __init__(
        self,
        *,
        min_on_bars: int = 1,
        cooldown_bars: int = 0,
    ):
        self.min_on_bars = int(min_on_bars)
        self.cooldown_bars = int(cooldown_bars)

    @property
    def name(self) -> str:
        return f"Churn(min_on={self.min_on_bars}, cd={self.cooldown_bars})"

    def _apply_entry_persistence(self, pos: pd.Series) -> pd.Series:
        n = int(self.min_on_bars)
        if n <= 1:
            return pos

        p = pos.fillna(0.0).astype(float)
        on = (p > 0.0).astype(int)

        # stable_on[t] = True iff on[t-n+1 : t] are all True
        stable_on = on.rolling(n, min_periods=n).min().fillna(0).astype(bool)
        return p.where(stable_on, 0.0)

    def _apply_reentry_cooldown(self, pos: pd.Series) -> pd.Series:
        c = int(self.cooldown_bars)
        if c <= 0:
            return pos

        p = pos.fillna(0.0).astype(float)
        on = p > 0.0

        exit_bar = (~on) & (on.shift(1, fill_value=False))
        cool = exit_bar.astype(int).rolling(c, min_periods=1).max().astype(bool)

        entry = on & (~on.shift(1, fill_value=False))
        block = entry & cool

        # If an entry is blocked, kill the whole would-be segment.
        seg = on.ne(on.shift(1, fill_value=False)).cumsum()
        seg_block = block.groupby(seg).transform("max")

        return p.where(~seg_block, 0.0)

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        # prices/context unused; kept for gate signature parity
        pos = positions.copy()
        pos = self._apply_entry_persistence(pos)
        pos = self._apply_reentry_cooldown(pos)
        return pos
