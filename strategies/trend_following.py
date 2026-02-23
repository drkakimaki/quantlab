"""Trend following strategies.

This module provides:
1. TrendStrategy - Simple MA crossover baseline (e.g., 20/100)
2. TrendStrategyWithGates - Composable trend strategy with signal gates (from YAML config)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

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

from .gates import (
    SignalGate,
    HTFConfirmGate,
    EMASeparationGate,
    NoChopGate,
    CorrelationGate,
    TimeFilterGate,
    EMAStrengthSizingGate,
    SeasonalitySizeCapGate,
    ChurnGate,
    MidDurationLossLimiterGate,
    TimeStopGate,
    ProfitMilestoneGate,
    RollingMaxExitGate,
    ShockExitGate,
)

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
        risk_gate: ShockExitGate | None = None,
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
            pos = self.nochop_gate.apply_exit_bad_bars(pos, px, context)

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
            risk_gate = ShockExitGate(
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
    "ShockExitGate",
]
