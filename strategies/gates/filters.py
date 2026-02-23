from __future__ import annotations

import pandas as pd

from ...engine.backtest import prices_to_returns
from ...time_filter import apply_time_filter

from .types import SignalGate
from .registry import register_gate


@register_gate("htf_confirm")
class HTFConfirmGate:
    """HTF confirmation gate.
    
    Gates positions by HTF SMA(fast) > SMA(slow), forward-filled to 5m.
    """

    def __init__(self, fast: int = 30, slow: int = 75):
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"HTF({self.fast}/{self.slow})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if context is None or "bars_15m" not in context:
            return positions

        bars = context["bars_15m"]
        if isinstance(bars, pd.DataFrame):
            htf_close = bars["close"].astype(float)
        else:
            htf_close = bars
        
        htf_close = htf_close.dropna()
        htf_close.index = pd.to_datetime(htf_close.index)

        sma_fast = htf_close.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = htf_close.rolling(self.slow, min_periods=self.slow).mean()
        htf_on = (sma_fast > sma_slow).astype(float)
        
        # Forward-fill to 5m frequency
        htf_on_base = htf_on.reindex(positions.index).ffill().fillna(0.0)
        
        return positions * htf_on_base


@register_gate("ema_sep")
class EMASeparationGate:
    """EMA separation gate using HTF bars.
    
    Requires EMA separation > ATR * k to avoid choppy markets.
    Uses HTF (15m) bars for proper TR-ATR calculation.
    """

    def __init__(
        self,
        ema_fast: int = 40,
        ema_slow: int = 300,
        atr_n: int = 20,
        sep_k: float = 0.05,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_n = atr_n
        self.sep_k = sep_k

    @property
    def name(self) -> str:
        return f"EMASep({self.ema_fast}/{self.ema_slow})"

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
        
        # EMAs on HTF
        ema_f = htf_close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = htf_close.ewm(span=self.ema_slow, adjust=False).mean()
        
        # TR-ATR on HTF bars
        h = bars["high"].astype(float).reindex(htf_close.index)
        l = bars["low"].astype(float).reindex(htf_close.index)
        prev_c = htf_close.shift(1)
        
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(self.atr_n, min_periods=self.atr_n).mean()
        
        # Gate: EMA fast > slow AND separation > k * ATR
        ema_sep_ok = (ema_f > ema_s) & ((ema_f - ema_s) > self.sep_k * atr)
        ema_sep_ok_base = ema_sep_ok.reindex(positions.index).ffill().fillna(False)
        
        return positions.where(ema_sep_ok_base, 0.0)


@register_gate("nochop")
class NoChopGate:
    """No-chop gate using HTF bars.

    Counts closes above EMA in lookback period.
    Supports entry_held mode.
    """

    def __init__(
        self,
        ema: int = 20,
        lookback: int = 40,
        min_closes: int = 24,
        entry_held: bool = False,
    ):
        self.ema = ema
        self.lookback = lookback
        self.min_closes = min_closes
        self.entry_held = entry_held

    @property
    def name(self) -> str:
        return f"NoChop(ema={self.ema}, entry_held={self.entry_held})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if context is None or "bars_15m" not in context:
            return positions

        htf_close = context["bars_15m"]["close"].astype(float).dropna()
        htf_close.index = pd.to_datetime(htf_close.index)
        
        # EMA on HTF
        ema_nc = htf_close.ewm(span=self.ema, adjust=False).mean()
        
        # Count closes above EMA in lookback
        above = (htf_close > ema_nc).astype(int)
        above_cnt = above.rolling(self.lookback, min_periods=self.lookback).sum()
        
        nochop_ok = (above_cnt >= self.min_closes).astype(bool)
        nochop_ok_base = nochop_ok.reindex(positions.index).ffill().fillna(False)
        
        pos = positions.copy()
        
        if not self.entry_held:
            # Simple mode: gate all positions
            pos = pos.where(nochop_ok_base, 0.0)
        else:
            # Entry-held mode: only gate at entry, hold for segment
            gate_on = pos > 0.0
            entry_bar = gate_on & (~gate_on.shift(1, fill_value=False))
            entry_ok = entry_bar & nochop_ok_base
            seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()
            seg_entry_ok = entry_ok.groupby(seg).transform("max")
            pos = (gate_on & seg_entry_ok).astype(float)

        return pos

@register_gate("corr")
class CorrelationGate:
    """Correlation stability gate with flip counting and confirm sizing.
    
    Implements:
    - Rolling correlation with flip counting
    - Segment-held entry gate
    - Confirm sizing (size based on XAG/EUR confirmation at entry)
    """

    def __init__(
        self,
        logic: str = "or",
        xag_window: int = 40,
        xag_min_abs: float = 0.10,
        xag_flip_lookback: int = 50,
        xag_max_flips: int = 0,
        eur_window: int = 75,
        eur_min_abs: float = 0.10,
        eur_flip_lookback: int = 75,
        eur_max_flips: int = 5,
        confirm_size_one: float = 1.0,
        confirm_size_both: float = 2.0,
    ):
        self.logic = logic
        self.xag_window = xag_window
        self.xag_min_abs = xag_min_abs
        self.xag_flip_lookback = xag_flip_lookback
        self.xag_max_flips = xag_max_flips
        self.eur_window = eur_window
        self.eur_min_abs = eur_min_abs
        self.eur_flip_lookback = eur_flip_lookback
        self.eur_max_flips = eur_max_flips
        self.confirm_size_one = confirm_size_one
        self.confirm_size_both = confirm_size_both

    @property
    def name(self) -> str:
        return f"Corr({self.logic}, size={self.confirm_size_one}/{self.confirm_size_both})"

    def _stable_mask(
        self,
        close: pd.Series,
        px: pd.Series,
        window: int,
        min_abs: float,
        flip_lb: int,
        max_flips: int,
        shift1: bool = True,
    ) -> pd.Series:
        """Compute stability mask for a correlation series."""
        cc = pd.Series(close).dropna().copy()
        cc.index = pd.to_datetime(cc.index)
        cc = cc.sort_index().reindex(px.index).ffill()

        r_base = prices_to_returns(px).fillna(0.0)
        r_cc = prices_to_returns(cc).fillna(0.0)
        c = r_base.rolling(window, min_periods=window).corr(r_cc)

        sign = c.apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
        flips = (sign != sign.shift(1)).astype(int)
        flips = flips.where(sign != 0.0, 0)
        flip_cnt = flips.rolling(flip_lb, min_periods=flip_lb).sum()

        ok = (c.abs() >= min_abs) & (flip_cnt <= max_flips)
        return ok.shift(1).fillna(False) if shift1 else ok.fillna(False)

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if context is None:
            return positions

        xag_close = context.get("prices_xag")
        eur_close = context.get("prices_eur")

        if xag_close is None:
            return positions

        px = prices.dropna().astype(float)
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        # Compute stability masks
        stable1 = self._stable_mask(
            xag_close, px,
            window=self.xag_window,
            min_abs=self.xag_min_abs,
            flip_lb=self.xag_flip_lookback,
            max_flips=self.xag_max_flips,
            shift1=True,
        )

        stable2 = None
        stable_ok = stable1
        
        if eur_close is not None:
            stable2 = self._stable_mask(
                eur_close, px,
                window=self.eur_window,
                min_abs=self.eur_min_abs,
                flip_lb=self.eur_flip_lookback,
                max_flips=self.eur_max_flips,
                shift1=True,
            )
            stable_ok = (stable1 | stable2) if self.logic == "or" else (stable1 & stable2)

        # Segment-held entry gate
        gate_on = positions > 0.0
        entry_bar = gate_on & (~gate_on.shift(1, fill_value=False))
        entry_ok = entry_bar & stable_ok
        seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()
        seg_entry_ok = entry_ok.groupby(seg).transform("max")
        pos = (gate_on & seg_entry_ok).astype(float)

        # Confirm sizing
        if stable2 is None:
            both_ok = stable_ok
            one_ok = stable_ok
        else:
            if self.logic == "or":
                both_ok = stable1 & stable2
                one_ok = stable_ok
            else:
                # AND mode: passing implies both confirmations, so sizing uses the "both" tier.
                both_ok = stable_ok
                one_ok = stable_ok

        s = pd.Series(0.0, index=pos.index)
        s = s.where(~one_ok, self.confirm_size_one)
        s = s.where(~both_ok, self.confirm_size_both)

        size_entry = s.shift(1).fillna(0.0)
        size_on_entry = size_entry.where(entry_bar)
        size_in_seg = size_on_entry.groupby(seg).ffill().fillna(0.0)
        
        return pos * size_in_seg


@register_gate("time_filter")
class TimeFilterGate:
    """Time filter gate.

    Canonical semantics: **force_flat** (positions are zeroed when blocked).
    """

    def __init__(
        self,
        allow_mask: pd.Series | None = None,
    ):
        self.allow_mask = allow_mask

    @property
    def name(self) -> str:
        return "TimeFilter(force_flat)"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        if self.allow_mask is None:
            return positions

        # Get allow_mask from context if not set
        mask = self.allow_mask
        if context and "allow_mask" in context:
            mask = context["allow_mask"]

        if mask is None:
            return positions

        return apply_time_filter(
            positions,
            pd.Series(mask, index=positions.index),
        )
