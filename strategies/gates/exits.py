from __future__ import annotations

import pandas as pd

from ...engine.backtest import prices_to_returns

from .types import SignalGate
from .registry import register_gate


@register_gate("mid_loss_limiter")
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


@register_gate("no_recovery_exit")
class NoRecoveryExitGate:
    """No-recovery exit.

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
        return f"NoRecovery(n={self.bar_n},min_ret={self.min_ret:g})"

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


@register_gate("profit_milestone")
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


@register_gate("rolling_max_exit")
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


@register_gate("shock_exit")
class ShockExitGate:
    """Shock-exit risk gate.

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
        return f"ShockExit(abs={self.shock_exit_abs_ret})"

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

