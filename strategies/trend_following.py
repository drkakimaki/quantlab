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
    NoRecoveryExitGate,
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
    """Trend strategy built from a composable gate pipeline.

    Design
    ------
    - Base signal: SMA crossover (fast/slow) on the main price series.
    - Gates: a list of `SignalGate` objects applied sequentially.

    Configuration
    -------------
    Two config shapes are supported:

    1) New: `pipeline: [...]` list (order is explicit)
    2) Legacy: flat blocks (htf_confirm/ema_sep/nochop/...) which are translated
       into the canonical order.
    """

    def __init__(
        self,
        fast: int = 30,
        slow: int = 75,
        *,
        gates: list[SignalGate] | None = None,
        requirements: dict[str, str] | None = None,
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.gates = list(gates or [])
        self._requirements = dict(requirements or {})

    @property
    def name(self) -> str:
        gate_names = [g.name for g in self.gates]
        return f"Trend({self.fast}/{self.slow}) + [{', '.join(gate_names)}]"

    @property
    def data_requirements(self) -> dict[str, str]:
        return dict(self._requirements)

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        px = prices.dropna().astype(float)
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        sma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = px.rolling(self.slow, min_periods=self.slow).mean()
        pos = (sma_fast > sma_slow).astype(float)

        for gate in self.gates:
            pos = gate(pos, px, context)

        return pos

    @staticmethod
    def _infer_requirements(gates: list[SignalGate]) -> dict[str, str]:
        reqs: dict[str, str] = {}

        # Heuristic mapping by class name to avoid importing gate classes here.
        for g in gates:
            n = g.__class__.__name__
            if n in {"HTFConfirmGate", "EMASeparationGate", "NoChopGate", "EMAStrengthSizingGate"}:
                reqs["bars_15m"] = "15-minute OHLC"
            if n in {"CorrelationGate"}:
                reqs["prices_xag"] = "XAGUSD prices"
                reqs["prices_eur"] = "EURUSD prices (optional)"
        # TimeFilterGate uses allow_mask either from constructor or context.
        for g in gates:
            if g.__class__.__name__ == "TimeFilterGate":
                # If the gate was constructed without a mask, it expects allow_mask in context.
                if getattr(g, "allow_mask", None) is None:
                    reqs["allow_mask"] = "Time filter allow mask"

        return reqs

    @classmethod
    def from_config(cls, config: dict, allow_mask: pd.Series | None = None) -> "TrendStrategyWithGates":
        from .gates import make_gate

        cfg = dict(config or {})

        fast = cfg.get("trend", {}).get("fast", 30)
        slow = cfg.get("trend", {}).get("slow", 75)

        pipeline = cfg.get("pipeline")
        specs: list[dict]

        if isinstance(pipeline, list) and pipeline:
            specs = [dict(x) for x in pipeline]
        else:
            # Legacy config translation (canonical order).
            specs = []

            if cfg.get("htf_confirm"):
                specs.append({"gate": "htf_confirm", "params": {"fast": fast, "slow": slow}})

            if cfg.get("ema_sep"):
                specs.append({"gate": "ema_sep", "params": dict(cfg.get("ema_sep") or {})})

            nc = cfg.get("nochop") or {}
            if nc:
                # Entry gate
                params = dict(nc)
                # Removed: exit_bad_bars semantics (was a separate post gate).
                params.pop("exit_bad_bars", None)
                specs.append({"gate": "nochop", "params": params})

            if cfg.get("corr"):
                corr_cfg = dict(cfg.get("corr") or {})
                xag = dict(corr_cfg.get("xag") or {})
                eur = dict(corr_cfg.get("eur") or {})
                sizing_cfg = dict(cfg.get("sizing") or {})
                params = {
                    "logic": corr_cfg.get("logic", "or"),
                    "xag_window": xag.get("window", 40),
                    "xag_min_abs": xag.get("min_abs", 0.10),
                    "xag_flip_lookback": xag.get("flip_lookback", 50),
                    "xag_max_flips": xag.get("max_flips", 0),
                    "eur_window": eur.get("window", 75),
                    "eur_min_abs": eur.get("min_abs", 0.10),
                    "eur_flip_lookback": eur.get("flip_lookback", 75),
                    "eur_max_flips": eur.get("max_flips", 5),
                    "confirm_size_one": sizing_cfg.get("confirm_size_one", 1.0),
                    "confirm_size_both": sizing_cfg.get("confirm_size_both", 2.0),
                }
                specs.append({"gate": "corr", "params": params})

            if cfg.get("time_filter") or allow_mask is not None:
                specs.append({"gate": "time_filter", "params": {}})

            es_cfg = cfg.get("ema_strength_sizing", {}) or {}
            if es_cfg:
                base_ema = cfg.get("ema_sep", {}) or {}
                sizing_cfg = cfg.get("sizing", {}) or {}
                params = {
                    "ema_fast": int(es_cfg.get("ema_fast", base_ema.get("ema_fast", 40))),
                    "ema_slow": int(es_cfg.get("ema_slow", base_ema.get("ema_slow", 300))),
                    "atr_n": int(es_cfg.get("atr_n", base_ema.get("atr_n", 20))),
                    "strong_k": float(es_cfg.get("strong_k", 0.10)),
                    "size_base": float(sizing_cfg.get("confirm_size_one", 1.0)),
                    "size_strong": float(sizing_cfg.get("confirm_size_both", 2.0)),
                }
                specs.append({"gate": "ema_strength_sizing", "params": params})

            season_cfg = cfg.get("seasonality", {}) or {}
            if isinstance(season_cfg.get("month_size_cap"), dict) and season_cfg.get("month_size_cap"):
                specs.append({"gate": "seasonality_cap", "params": {"month_size_cap": season_cfg.get("month_size_cap")}})

            churn_cfg = cfg.get("churn", {}) or {}
            if churn_cfg:
                specs.append({"gate": "churn", "params": dict(churn_cfg)})

            early_cfg = cfg.get("mid_loss_limiter_early", {}) or {}
            if early_cfg:
                specs.append({"gate": "mid_loss_limiter", "params": {
                    "min_bars": int(early_cfg.get("min_bars", 7)),
                    "max_bars": int(early_cfg.get("max_bars", 12)),
                    "stop_ret": float(early_cfg.get("stop_ret", -0.006)),
                }})

            mid_loss_cfg = cfg.get("mid_loss_limiter", {}) or {}
            if mid_loss_cfg:
                specs.append({"gate": "mid_loss_limiter", "params": dict(mid_loss_cfg)})

            ts_cfg = cfg.get("time_stop", {}) or {}
            if ts_cfg:
                specs.append({"gate": "no_recovery_exit", "params": dict(ts_cfg)})

            pm_cfg = cfg.get("profit_milestone", {}) or {}
            if pm_cfg:
                specs.append({"gate": "profit_milestone", "params": dict(pm_cfg)})

            rm_cfg = cfg.get("rolling_max_exit", {}) or {}
            if rm_cfg:
                specs.append({"gate": "rolling_max_exit", "params": dict(rm_cfg)})

            risk_cfg = cfg.get("risk", {}) or {}
            if risk_cfg:
                specs.append({"gate": "shock_exit", "params": dict(risk_cfg)})

        # Instantiate gates
        gates: list[SignalGate] = []
        for spec in specs:
            spec = dict(spec)
            gate_name = spec.get("gate") or spec.get("name")
            if not gate_name:
                raise ValueError(f"Invalid gate spec: {spec!r}")

            params = spec.get("params")
            if params is None:
                params = {k: v for k, v in spec.items() if k not in {"gate", "name", "params"}}
            params = dict(params or {})

            # Inject allow_mask for time_filter when provided.
            if str(gate_name).strip() == "time_filter" and allow_mask is not None and "allow_mask" not in params:
                params["allow_mask"] = allow_mask

            gates.append(make_gate({"gate": gate_name, "params": params}))

        reqs = cls._infer_requirements(gates)
        return cls(fast=fast, slow=slow, gates=gates, requirements=reqs)
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
    "NoRecoveryExitGate",
    "ProfitMilestoneGate",
    "RollingMaxExitGate",
    "ShockExitGate",
]
