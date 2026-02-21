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

    Applies allow_mask to block positions during specific times.
    """

    def __init__(
        self,
        allow_mask: pd.Series | None = None,
        mode: str = "force_flat",
        entry_shift: int = 1,
    ):
        self.allow_mask = allow_mask
        self.mode = mode
        self.entry_shift = entry_shift

    @property
    def name(self) -> str:
        return f"TimeFilter({self.mode})"

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
            mode=self.mode,
            entry_shift=self.entry_shift,
        )


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
        base_signal -> htf -> ema_sep -> nochop -> corr -> time_filter -> nochop_exit -> risk
    
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
        churn_gate: ChurnGate | None = None,
        risk_gate: RiskGate | None = None,
    ):
        self.fast = fast
        self.slow = slow
        self.htf_gate = htf_gate
        self.ema_sep_gate = ema_sep_gate
        self.nochop_gate = nochop_gate
        self.corr_gate = corr_gate
        self.time_filter_gate = time_filter_gate
        self.churn_gate = churn_gate
        self.risk_gate = risk_gate

    @property
    def name(self) -> str:
        gates = [g for g in [
            self.htf_gate,
            self.ema_sep_gate,
            self.nochop_gate,
            self.corr_gate,
            self.time_filter_gate,
            self.churn_gate,
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

        # Churn gate (after time_filter + optional exit_bad_bars, before risk)
        if self.churn_gate:
            pos = self.churn_gate(pos, px, context)

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

        # Time filter gate
        time_filter_gate = None
        if config.get("time_filter") or allow_mask is not None:
            tf_cfg = config.get("time_filter", {})
            time_filter_gate = TimeFilterGate(
                allow_mask=allow_mask,
                mode=tf_cfg.get("mode", "force_flat"),
                entry_shift=tf_cfg.get("entry_shift", 1),
            )

        # Churn gate
        churn_gate = None
        churn_cfg = config.get("churn", {})
        if churn_cfg:
            churn_gate = ChurnGate(
                min_on_bars=int(churn_cfg.get("min_on_bars", 1) or 1),
                cooldown_bars=int(churn_cfg.get("cooldown_bars", 0) or 0),
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
            churn_gate=churn_gate,
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
    "ChurnGate",
    "RiskGate",
]
