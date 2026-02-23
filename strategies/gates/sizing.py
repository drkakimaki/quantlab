from __future__ import annotations

import pandas as pd

from ...engine.backtest import prices_to_returns

from .types import SignalGate
from .registry import register_gate


@register_gate("ema_strength_sizing")
class EMAStrengthSizingGate:
    """EMA separation *strength* sizing gate.

    This gate does NOT create entries. It only changes position *size* for bars
    where the strategy is already long (positions > 0).

    Semantics
    ---------
    - Compute EMA_fast, EMA_slow, and ATR on HTF (15m) bars.
    - Define `strong_ok` when (EMA_fast > EMA_slow) AND (EMA_fast - EMA_slow) > strong_k * ATR.
    - Decide size at entry (segment-held), using previous-bar info (shift(1)).

    Typical use
    -----------
    Keep a conservative base entry condition (e.g. ema_sep with sep_k), but size up
    when separation is *stronger* (strong_k > sep_k).
    """

    def __init__(
        self,
        *,
        ema_fast: int = 40,
        ema_slow: int = 300,
        atr_n: int = 20,
        strong_k: float = 0.10,
        size_base: float = 1.0,
        size_strong: float = 2.0,
    ):
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)
        self.atr_n = int(atr_n)
        self.strong_k = float(strong_k)
        self.size_base = float(size_base)
        self.size_strong = float(size_strong)

    @property
    def name(self) -> str:
        return f"EMAStrengthSize(k={self.strong_k:g}, {self.size_base:g}/{self.size_strong:g})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if context is None or "bars_15m" not in context:
            return positions

        bars = context["bars_15m"].copy()
        bars.index = pd.to_datetime(bars.index)
        bars = bars.rename(columns={c: c.lower() for c in bars.columns})

        htf_close = bars["close"].astype(float).dropna()

        ema_f = htf_close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = htf_close.ewm(span=self.ema_slow, adjust=False).mean()

        h = bars["high"].astype(float).reindex(htf_close.index)
        l = bars["low"].astype(float).reindex(htf_close.index)
        prev_c = htf_close.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_n, min_periods=self.atr_n).mean()

        strong_ok = (ema_f > ema_s) & ((ema_f - ema_s) > self.strong_k * atr)
        strong_ok_base = strong_ok.reindex(positions.index).ffill().fillna(False).astype(bool)

        p = positions.fillna(0.0).astype(float)
        gate_on = p > 0.0
        entry_bar = gate_on & (~gate_on.shift(1, fill_value=False))
        seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()

        # Decide size using prior-bar info (avoid lookahead)
        strong_entry = strong_ok_base.shift(1).fillna(False).astype(bool)
        size_series = pd.Series(self.size_base, index=p.index)
        size_series = size_series.where(~strong_entry, self.size_strong)

        size_on_entry = size_series.where(entry_bar)
        size_in_seg = size_on_entry.groupby(seg).ffill().fillna(self.size_base)

        # Apply sizing only when in-position
        return p.where(~gate_on, size_in_seg)


@register_gate("seasonality_cap")
class SeasonalitySizeCapGate:
    """Seasonality-based position size cap.

    Post-processing gate that caps position size based on calendar month.

    Example: cap to 1.0 during June (month=6) while leaving other months unchanged.
    """

    def __init__(self, *, month_size_cap: dict[int, float] | None = None):
        self.month_size_cap = month_size_cap or {}

    @property
    def name(self) -> str:
        if not self.month_size_cap:
            return "SeasonalitySizeCap(off)"
        parts = ",".join(f"{m}:{c:g}" for m, c in sorted(self.month_size_cap.items()))
        return f"SeasonalitySizeCap({parts})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if not self.month_size_cap:
            return positions

        pos = positions.copy().fillna(0.0).astype(float)
        idx = pd.DatetimeIndex(pos.index)
        for m, cap in self.month_size_cap.items():
            mask = (idx.month == int(m))
            if mask.any():
                pos.loc[mask] = pos.loc[mask].clip(upper=float(cap))
        return pos
