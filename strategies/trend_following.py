"""Trend following strategies.

This module provides:
1. TrendStrategy - Simple MA crossover baseline (e.g., 20/100)
2. TrendStrategyWithGates - Composable trend strategy with signal gates (from YAML config)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import numpy as np
import yaml

from .base import StrategyBase, BacktestResult, BacktestConfig
from ..engine.backtest import backtest_positions_account_margin, prices_to_returns
from ..data.resample import load_dukascopy_ohlc
from ..time_filter import EventWindow, build_allow_mask_from_events


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class SignalGate(Protocol):
    """Protocol for signal modification gates."""

    @property
    def name(self) -> str:
        """Gate name for logging/debugging."""
        ...

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        """Apply gate to positions."""
        ...


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


class NoChopGate:
    """No-chop gate using HTF bars.
    
    Counts closes above EMA in lookback period.
    Supports entry_held mode and exit_bad_bars.
    """

    def __init__(
        self,
        ema: int = 20,
        lookback: int = 40,
        min_closes: int = 24,
        entry_held: bool = False,
        exit_bad_bars: int = 0,
    ):
        self.ema = ema
        self.lookback = lookback
        self.min_closes = min_closes
        self.entry_held = entry_held
        self.exit_bad_bars = exit_bad_bars

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
        
        # Store nochop_ok_base for exit_bad_bars
        pos.attrs['nochop_ok_base'] = nochop_ok_base
        
        return pos
    
    def apply_exit_bad_bars(self, positions: pd.Series) -> pd.Series:
        """Apply exit_bad_bars after all other gates."""
        if self.exit_bad_bars <= 0:
            return positions
        
        nochop_ok_base = positions.attrs.get('nochop_ok_base', pd.Series(True, index=positions.index))
        
        gate_on = positions > 0.0
        seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()
        bad_in_seg = (~nochop_ok_base) & gate_on
        grp = (~bad_in_seg).cumsum()
        streak = bad_in_seg.astype(int).groupby(grp).cumsum()
        trigger = streak >= self.exit_bad_bars
        seg_kill = trigger.groupby(seg).cummax()
        
        return positions.where(~seg_kill, 0.0)


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

        from ..time_filter import apply_time_filter

        return apply_time_filter(
            positions,
            pd.Series(mask, index=positions.index),
        )


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


class MidDurationLossLimiterGate:
    """Exit losers in the toxic mid-duration band.

    Motivation
    ----------
    Trade breakdown shows a persistent negative expectancy for trades that last
    ~13–48 bars. This gate targets that regime by force-flattening a segment once
    it breaches a loss threshold during a specified bar window.

    Semantics
    ---------
    - Only applies when already in-position (positions > 0).
    - For each segment, compute:
        - bars_since_entry (0-based)
        - unrealized return since entry: (price / entry_price - 1)
    - If `min_bars <= bars_since_entry <= max_bars` AND unrealized_return <= stop_ret,
      then kill the remainder of the segment (set positions to 0 from that point on).

    Anti-lookahead
    -------------
    The trigger uses the *last closed bar* information (shifted by 1), consistent
    with other gates.
    """

    def __init__(
        self,
        *,
        min_bars: int = 13,
        max_bars: int = 48,
        stop_ret: float = -0.01,
    ):
        self.min_bars = int(min_bars)
        self.max_bars = int(max_bars)
        self.stop_ret = float(stop_ret)

    @property
    def name(self) -> str:
        return f"MidDurLoss(min={self.min_bars},max={self.max_bars},ret={self.stop_ret:g})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        p = positions.fillna(0.0).astype(float)
        px = prices.reindex(p.index).astype(float).ffill()

        on = p > 0.0
        if not bool(on.any()):
            return p

        entry = on & (~on.shift(1, fill_value=False))
        seg = on.ne(on.shift(1, fill_value=False)).cumsum()

        # entry price held constant over the segment
        entry_px = px.where(entry).groupby(seg).ffill()

        # bars since entry within each segment
        bars_since = on.astype(int).groupby(seg).cumsum() - 1

        # unrealized return since entry (safe when entry_px is nan outside segments)
        uret = (px / entry_px) - 1.0

        # Use last-closed-bar info for trigger
        uret_lag = uret.shift(1)
        bars_lag = bars_since.shift(1)

        in_window = (bars_lag >= self.min_bars) & (bars_lag <= self.max_bars)
        trigger = on & in_window & (uret_lag <= self.stop_ret)

        if not bool(trigger.any()):
            return p

        # Kill remainder of segment once triggered
        kill = trigger.groupby(seg).cummax()
        return p.where(~kill, 0.0)


class TimeStopGate:
    """Time-stop / no-recovery exit.

    If a position has not achieved at least `min_ret` return since entry by a
    specified holding age, we force-flat the remainder of the segment.

    This targets the toxic mid-duration region without adding new entry filters.

    Semantics
    ---------
    - Compute unrealized return since entry: (price/entry_price - 1)
    - When bars_since_entry >= bar_n AND unrealized_return(last_closed) <= min_ret:
      kill the remainder of the segment.

    Anti-lookahead: uses last closed bar info (shift(1)).
    """

    def __init__(self, *, bar_n: int = 24, min_ret: float = 0.0):
        self.bar_n = int(bar_n)
        self.min_ret = float(min_ret)

    @property
    def name(self) -> str:
        return f"TimeStop(n={self.bar_n},min_ret={self.min_ret:g})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        p = positions.fillna(0.0).astype(float)
        px = prices.reindex(p.index).astype(float).ffill()

        on = p > 0.0
        if not bool(on.any()):
            return p

        entry = on & (~on.shift(1, fill_value=False))
        seg = on.ne(on.shift(1, fill_value=False)).cumsum()
        entry_px = px.where(entry).groupby(seg).ffill()
        bars_since = on.astype(int).groupby(seg).cumsum() - 1
        uret = (px / entry_px) - 1.0

        uret_lag = uret.shift(1)
        bars_lag = bars_since.shift(1)

        trigger = on & (bars_lag >= self.bar_n) & (uret_lag <= self.min_ret)
        if not bool(trigger.any()):
            return p

        kill = trigger.groupby(seg).cummax()
        return p.where(~kill, 0.0)


class ProfitMilestoneGate:
    """Kill trades that fail to reach a profit milestone by N bars.

    Idea
    ----
    The long-duration winners tend to show profit early at some point, while toxic
    mid-duration trades often never meaningfully get off the ground. This gate
    exits a segment if it hasn't *ever* reached `milestone_ret` unrealized return
    by `bar_n` bars since entry.

    Semantics
    ---------
    - Compute unrealized return since entry: (price/entry_price - 1)
    - Track running max of unrealized return within each segment.
    - If bars_since_entry >= bar_n AND running_max_unrealized(last_closed) < milestone_ret:
      kill the remainder of the segment.

    Anti-lookahead: uses last closed bar info (shift(1)).
    """

    def __init__(self, *, bar_n: int = 24, milestone_ret: float = 0.002):
        self.bar_n = int(bar_n)
        self.milestone_ret = float(milestone_ret)

    @property
    def name(self) -> str:
        return f"ProfitMilestone(n={self.bar_n},ret={self.milestone_ret:g})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        p = positions.fillna(0.0).astype(float)
        px = prices.reindex(p.index).astype(float).ffill()

        on = p > 0.0
        if not bool(on.any()):
            return p

        entry = on & (~on.shift(1, fill_value=False))
        seg = on.ne(on.shift(1, fill_value=False)).cumsum()
        entry_px = px.where(entry).groupby(seg).ffill()
        bars_since = on.astype(int).groupby(seg).cumsum() - 1
        uret = (px / entry_px) - 1.0

        run_max = uret.groupby(seg).cummax()

        run_max_lag = run_max.shift(1)
        bars_lag = bars_since.shift(1)

        trigger = on & (bars_lag >= self.bar_n) & (run_max_lag < self.milestone_ret)
        if not bool(trigger.any()):
            return p

        kill = trigger.groupby(seg).cummax()
        return p.where(~kill, 0.0)


class RollingMaxExitGate:
    """Exit if recent max unrealized return is too low (stagnation filter).

    This is a post-entry control intended to hit mid-duration "never gets going"
    trades without requiring a one-time milestone.

    Semantics
    ---------
    - Compute unrealized return since entry: uret = price/entry_price - 1
    - For each segment, compute rolling max of uret over the last `window_bars`.
    - If bars_since_entry >= min_bars AND rolling_max_uret(last_closed) < min_peak_ret:
      kill the remainder of the segment.

    Anti-lookahead: uses last closed bar info (shift(1)).
    """

    def __init__(
        self,
        *,
        window_bars: int = 24,
        min_bars: int = 24,
        min_peak_ret: float = 0.0,
    ):
        self.window_bars = int(window_bars)
        self.min_bars = int(min_bars)
        self.min_peak_ret = float(min_peak_ret)

    @property
    def name(self) -> str:
        return f"RollingMaxExit(w={self.window_bars},min={self.min_bars},peak={self.min_peak_ret:g})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        p = positions.fillna(0.0).astype(float)
        px = prices.reindex(p.index).astype(float).ffill()

        on = p > 0.0
        if not bool(on.any()):
            return p

        entry = on & (~on.shift(1, fill_value=False))
        seg = on.ne(on.shift(1, fill_value=False)).cumsum()
        entry_px = px.where(entry).groupby(seg).ffill()
        bars_since = on.astype(int).groupby(seg).cumsum() - 1
        uret = (px / entry_px) - 1.0

        # Rolling max within each segment
        def _roll_max(s: pd.Series) -> pd.Series:
            return s.rolling(self.window_bars, min_periods=1).max()

        roll_max = uret.groupby(seg, group_keys=False).apply(_roll_max)

        roll_max_lag = roll_max.shift(1)
        bars_lag = bars_since.shift(1)

        trigger = on & (bars_lag >= self.min_bars) & (roll_max_lag < self.min_peak_ret)
        if not bool(trigger.any()):
            return p

        kill = trigger.groupby(seg).cummax()
        return p.where(~kill, 0.0)


class RiskGate:
    """Risk management gate.

    Implements shock exit (abs return or sigma-based) and optional cooldown.

    Note: segment TTL was removed from the canonical strategy code (unused in the
    promoted config). If we want TTL back later, re-introduce it as a separate
    gate/module so it can be tested/ablationed cleanly.
    """

    def __init__(
        self,
        shock_exit_abs_ret: float = 0.0,
        shock_exit_sigma_k: float = 0.0,
        shock_exit_sigma_window: int = 96,
        shock_cooldown_bars: int = 0,
    ):
        self.shock_exit_abs_ret = shock_exit_abs_ret
        self.shock_exit_sigma_k = shock_exit_sigma_k
        self.shock_exit_sigma_window = shock_exit_sigma_window
        self.shock_cooldown_bars = shock_cooldown_bars

    @property
    def name(self) -> str:
        return f"Risk(shock={self.shock_exit_abs_ret})"

    def __call__(
        self,
        positions: pd.Series,
        prices: pd.Series,
        context: dict | None = None,
    ) -> pd.Series:
        pos = positions.copy()
        px = prices.dropna().astype(float)
        
        # Shock exit
        if self.shock_exit_abs_ret > 0.0 or (self.shock_exit_sigma_k > 0.0 and self.shock_exit_sigma_window > 1):
            r = prices_to_returns(px).fillna(0.0).astype(float)
            shock = pd.Series(False, index=pos.index)

            if self.shock_exit_abs_ret > 0.0:
                shock = shock | (r.abs() >= self.shock_exit_abs_ret)

            if self.shock_exit_sigma_k > 0.0 and self.shock_exit_sigma_window > 1:
                sig = r.rolling(self.shock_exit_sigma_window, min_periods=self.shock_exit_sigma_window).std()
                shock = shock | (r.abs() >= (self.shock_exit_sigma_k * sig))

            shock = shock.shift(1).fillna(False)

            if bool(shock.any()):
                gate_on = pos > 0.0
                seg = gate_on.ne(gate_on.shift(1, fill_value=False)).cumsum()

                shock_in_seg = shock & gate_on
                seg_kill = shock_in_seg.groupby(seg).cummax()
                pos = pos.where(~seg_kill, 0.0)

                if self.shock_cooldown_bars > 0:
                    cool_mask = shock.astype(int).rolling(self.shock_cooldown_bars, min_periods=1).max().astype(bool)
                    pos = pos.where(~cool_mask, 0.0)

        return pos


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

class TrendStrategy(StrategyBase):
    """Simple trend following strategy using MA crossover.
    
    Baseline strategy without any filters.
    Buy when fast SMA > slow SMA, flat otherwise.
    """

    def __init__(self, fast: int = 20, slow: int = 100):
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast < slow")
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"Trend(SMA {self.fast}/{self.slow})"

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        """Generate positions: 1.0 when fast SMA > slow SMA."""
        px = prices.dropna().astype(float)
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        sma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = px.rolling(self.slow, min_periods=self.slow).mean()
        pos = (sma_fast > sma_slow).astype(float)

        return pos


class TrendStrategyWithGates(StrategyBase):
    """Trend strategy built from composable gates.

    Gates are applied in sequence:
        base_signal -> htf -> ema_sep -> nochop -> corr -> time_filter -> nochop_exit
        -> ema_strength_sizing -> seasonality_cap -> churn -> risk

    Each gate implements filtering/risk logic.
    """

    def __init__(
        self,
        fast: int = 30,
        slow: int = 75,
        *,
        htf_gate: HTFConfirmGate | None = None,
        ema_sep_gate: EMASeparationGate | None = None,
        nochop_gate: NoChopGate | None = None,
        corr_gate: CorrelationGate | None = None,
        time_filter_gate: TimeFilterGate | None = None,
        ema_strength_sizing_gate: EMAStrengthSizingGate | None = None,
        seasonality_gate: SeasonalitySizeCapGate | None = None,
        churn_gate: ChurnGate | None = None,
        mid_loss_gate_early: MidDurationLossLimiterGate | None = None,
        mid_loss_gate: MidDurationLossLimiterGate | None = None,
        time_stop_gate: TimeStopGate | None = None,
        profit_milestone_gate: ProfitMilestoneGate | None = None,
        rolling_max_exit_gate: RollingMaxExitGate | None = None,
        risk_gate: RiskGate | None = None,
    ):
        self.fast = fast
        self.slow = slow
        self.htf_gate = htf_gate
        self.ema_sep_gate = ema_sep_gate
        self.nochop_gate = nochop_gate
        self.corr_gate = corr_gate
        self.time_filter_gate = time_filter_gate
        self.ema_strength_sizing_gate = ema_strength_sizing_gate
        self.seasonality_gate = seasonality_gate
        self.churn_gate = churn_gate
        self.mid_loss_gate_early = mid_loss_gate_early
        self.mid_loss_gate = mid_loss_gate
        self.time_stop_gate = time_stop_gate
        self.profit_milestone_gate = profit_milestone_gate
        self.rolling_max_exit_gate = rolling_max_exit_gate
        self.risk_gate = risk_gate

    @property
    def name(self) -> str:
        gates = [g for g in [
            self.htf_gate,
            self.ema_sep_gate,
            self.nochop_gate,
            self.corr_gate,
            self.time_filter_gate,
            self.ema_strength_sizing_gate,
            self.seasonality_gate,
            self.churn_gate,
            self.mid_loss_gate_early,
            self.mid_loss_gate,
            self.time_stop_gate,
            self.profit_milestone_gate,
            self.rolling_max_exit_gate,
            self.risk_gate,
        ] if g is not None]
        gate_names = [g.name for g in gates]
        return f"Trend({self.fast}/{self.slow}) + [{', '.join(gate_names)}]"

    @property
    def data_requirements(self) -> dict[str, str]:
        reqs = {}
        if self.htf_gate or self.ema_sep_gate or self.nochop_gate:
            reqs["bars_15m"] = "15-minute OHLC"
        if self.corr_gate:
            reqs["prices_xag"] = "XAGUSD prices"
            reqs["prices_eur"] = "EURUSD prices (optional)"
        if self.time_filter_gate and self.time_filter_gate.allow_mask is None:
            reqs["allow_mask"] = "Time filter allow mask"
        if self.ema_strength_sizing_gate:
            reqs["bars_15m"] = "15-minute OHLC (for EMA strength sizing)"
        return reqs

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        """Generate positions: base signal -> gates in sequence."""
        px = prices.dropna().astype(float)
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        # Base signal: SMA crossover
        sma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = px.rolling(self.slow, min_periods=self.slow).mean()
        pos = (sma_fast > sma_slow).astype(float)

        # Apply gates in sequence
        if self.htf_gate:
            pos = self.htf_gate(pos, px, context)

        if self.ema_sep_gate:
            pos = self.ema_sep_gate(pos, px, context)

        if self.nochop_gate:
            pos = self.nochop_gate(pos, px, context)

        if self.corr_gate:
            pos = self.corr_gate(pos, px, context)

        if self.time_filter_gate:
            pos = self.time_filter_gate(pos, px, context)

        # NoChop exit_bad_bars (after time_filter)
        if self.nochop_gate and self.nochop_gate.exit_bad_bars > 0:
            pos = self.nochop_gate.apply_exit_bad_bars(pos)

        # EMA strength sizing (after time_filter + exit_bad_bars, before seasonality cap)
        if self.ema_strength_sizing_gate:
            pos = self.ema_strength_sizing_gate(pos, px, context)

        # Seasonality size cap (after sizing, before churn/risk)
        if self.seasonality_gate:
            pos = self.seasonality_gate(pos, px, context)

        # Churn gate (after seasonality cap, before mid-loss/risk)
        if self.churn_gate:
            pos = self.churn_gate(pos, px, context)

        # Mid-duration loss limiter (two-stage optional): early window then mid window
        if self.mid_loss_gate_early:
            pos = self.mid_loss_gate_early(pos, px, context)

        if self.mid_loss_gate:
            pos = self.mid_loss_gate(pos, px, context)

        # Time-stop (no-recovery) exit (after loss limiter, before milestone/risk)
        if self.time_stop_gate:
            pos = self.time_stop_gate(pos, px, context)

        # Profit milestone gate (after time-stop, before rolling-max/risk)
        if self.profit_milestone_gate:
            pos = self.profit_milestone_gate(pos, px, context)

        # Rolling max exit gate (after milestone, before risk)
        if self.rolling_max_exit_gate:
            pos = self.rolling_max_exit_gate(pos, px, context)

        if self.risk_gate:
            pos = self.risk_gate(pos, px, context)

        return pos

    @classmethod
    def from_config(cls, config: dict, allow_mask: pd.Series | None = None) -> "TrendStrategyWithGates":
        """Build strategy from config dict (like current.yaml)."""
        fast = config.get("trend", {}).get("fast", 30)
        slow = config.get("trend", {}).get("slow", 75)

        # HTF gate
        htf_gate = None
        if config.get("htf_confirm"):
            htf_gate = HTFConfirmGate(fast=fast, slow=slow)

        # EMA separation gate
        ema_sep_gate = None
        if config.get("ema_sep"):
            ema_sep_cfg = config["ema_sep"]
            ema_sep_gate = EMASeparationGate(
                ema_fast=ema_sep_cfg.get("ema_fast", 40),
                ema_slow=ema_sep_cfg.get("ema_slow", 300),
                atr_n=ema_sep_cfg.get("atr_n", 20),
                sep_k=ema_sep_cfg.get("sep_k", 0.05),
            )

        # NoChop gate
        nochop_gate = None
        if config.get("nochop"):
            nochop_cfg = config["nochop"]
            nochop_gate = NoChopGate(
                ema=nochop_cfg.get("ema", 20),
                lookback=nochop_cfg.get("lookback", 40),
                min_closes=nochop_cfg.get("min_closes", 24),
                entry_held=nochop_cfg.get("entry_held", False),
                exit_bad_bars=nochop_cfg.get("exit_bad_bars", 0),
            )

        # Correlation gate
        corr_gate = None
        if config.get("corr"):
            corr_cfg = config["corr"]
            xag_cfg = corr_cfg.get("xag", {})
            eur_cfg = corr_cfg.get("eur", {})
            sizing_cfg = config.get("sizing", {})
            
            corr_gate = CorrelationGate(
                logic=corr_cfg.get("logic", "or"),
                xag_window=xag_cfg.get("window", 40),
                xag_min_abs=xag_cfg.get("min_abs", 0.10),
                xag_flip_lookback=xag_cfg.get("flip_lookback", 50),
                xag_max_flips=xag_cfg.get("max_flips", 0),
                eur_window=eur_cfg.get("window", 75),
                eur_min_abs=eur_cfg.get("min_abs", 0.10),
                eur_flip_lookback=eur_cfg.get("flip_lookback", 75),
                eur_max_flips=eur_cfg.get("max_flips", 5),
                confirm_size_one=sizing_cfg.get("confirm_size_one", 1.0),
                confirm_size_both=sizing_cfg.get("confirm_size_both", 2.0),
            )

        # Time filter gate (canonical: force_flat)
        time_filter_gate = None
        if config.get("time_filter") or allow_mask is not None:
            time_filter_gate = TimeFilterGate(
                allow_mask=allow_mask,
            )

        # EMA strength sizing gate (optional)
        ema_strength_sizing_gate = None
        ema_strength_cfg = config.get("ema_strength_sizing", {}) or {}
        if ema_strength_cfg:
            # Pull EMA/ATR params from ema_sep when available, else allow override.
            base_ema = config.get("ema_sep", {}) or {}
            sizing_cfg = config.get("sizing", {}) or {}
            ema_strength_sizing_gate = EMAStrengthSizingGate(
                ema_fast=int(ema_strength_cfg.get("ema_fast", base_ema.get("ema_fast", 40))),
                ema_slow=int(ema_strength_cfg.get("ema_slow", base_ema.get("ema_slow", 300))),
                atr_n=int(ema_strength_cfg.get("atr_n", base_ema.get("atr_n", 20))),
                strong_k=float(ema_strength_cfg.get("strong_k", 0.10)),
                size_base=float(sizing_cfg.get("confirm_size_one", 1.0)),
                size_strong=float(sizing_cfg.get("confirm_size_both", 2.0)),
            )

        # Seasonality gate (optional)
        seasonality_gate = None
        seasonality_cfg = config.get("seasonality", {}) or {}
        month_cap = seasonality_cfg.get("month_size_cap")
        if isinstance(month_cap, dict) and month_cap:
            # normalize keys/values
            caps: dict[int, float] = {int(k): float(v) for k, v in month_cap.items()}
            seasonality_gate = SeasonalitySizeCapGate(month_size_cap=caps)

        # Churn gate
        churn_gate = None
        churn_cfg = config.get("churn", {})
        if churn_cfg:
            churn_gate = ChurnGate(
                min_on_bars=int(churn_cfg.get("min_on_bars", 1) or 1),
                cooldown_bars=int(churn_cfg.get("cooldown_bars", 0) or 0),
            )

        # Mid-duration loss limiter gate (optional; can be two-stage)
        mid_loss_gate_early = None
        early_cfg = config.get("mid_loss_limiter_early", {}) or {}
        if early_cfg:
            mid_loss_gate_early = MidDurationLossLimiterGate(
                min_bars=int(early_cfg.get("min_bars", 7)),
                max_bars=int(early_cfg.get("max_bars", 12)),
                stop_ret=float(early_cfg.get("stop_ret", -0.006)),
            )

        mid_loss_gate = None
        mid_loss_cfg = config.get("mid_loss_limiter", {}) or {}
        if mid_loss_cfg:
            mid_loss_gate = MidDurationLossLimiterGate(
                min_bars=int(mid_loss_cfg.get("min_bars", 13)),
                max_bars=int(mid_loss_cfg.get("max_bars", 48)),
                stop_ret=float(mid_loss_cfg.get("stop_ret", -0.01)),
            )

        # Time-stop gate (optional)
        time_stop_gate = None
        ts_cfg = config.get("time_stop", {}) or {}
        if ts_cfg:
            time_stop_gate = TimeStopGate(
                bar_n=int(ts_cfg.get("bar_n", 24)),
                min_ret=float(ts_cfg.get("min_ret", 0.0)),
            )

        # Profit milestone gate (optional)
        profit_milestone_gate = None
        pm_cfg = config.get("profit_milestone", {}) or {}
        if pm_cfg:
            profit_milestone_gate = ProfitMilestoneGate(
                bar_n=int(pm_cfg.get("bar_n", 24)),
                milestone_ret=float(pm_cfg.get("milestone_ret", 0.002)),
            )

        # Rolling-max exit gate (optional)
        rolling_max_exit_gate = None
        rm_cfg = config.get("rolling_max_exit", {}) or {}
        if rm_cfg:
            rolling_max_exit_gate = RollingMaxExitGate(
                window_bars=int(rm_cfg.get("window_bars", 24)),
                min_bars=int(rm_cfg.get("min_bars", 24)),
                min_peak_ret=float(rm_cfg.get("min_peak_ret", 0.0)),
            )

        # Risk gate
        risk_gate = None
        risk_cfg = config.get("risk", {})
        if risk_cfg:
            risk_gate = RiskGate(
                shock_exit_abs_ret=risk_cfg.get("shock_exit_abs_ret", 0.0),
                shock_exit_sigma_k=risk_cfg.get("shock_exit_sigma_k", 0.0),
                shock_exit_sigma_window=risk_cfg.get("shock_exit_sigma_window", 96),
                shock_cooldown_bars=risk_cfg.get("shock_cooldown_bars", 0),
            )

        return cls(
            fast=fast,
            slow=slow,
            htf_gate=htf_gate,
            ema_sep_gate=ema_sep_gate,
            nochop_gate=nochop_gate,
            corr_gate=corr_gate,
            time_filter_gate=time_filter_gate,
            ema_strength_sizing_gate=ema_strength_sizing_gate,
            seasonality_gate=seasonality_gate,
            churn_gate=churn_gate,
            mid_loss_gate_early=mid_loss_gate_early,
            mid_loss_gate=mid_loss_gate,
            time_stop_gate=time_stop_gate,
            profit_milestone_gate=profit_milestone_gate,
            rolling_max_exit_gate=rolling_max_exit_gate,
            risk_gate=risk_gate,
        )


# Re-export gates for convenience
__all__ = [
    # Strategy classes
    "TrendStrategy",
    "TrendStrategyWithGates",
    # Gates
    "SignalGate",
    "HTFConfirmGate",
    "EMASeparationGate",
    "NoChopGate",
    "CorrelationGate",
    "TimeFilterGate",
    "EMAStrengthSizingGate",
    "SeasonalitySizeCapGate",
    "ChurnGate",
    "MidDurationLossLimiterGate",
    "TimeStopGate",
    "ProfitMilestoneGate",
    "RollingMaxExitGate",
    "RiskGate",
]
