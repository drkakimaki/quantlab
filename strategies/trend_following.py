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
from ..data.loaders import load_dukascopy_ohlc
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

        if not (isinstance(pipeline, list) and pipeline):
            raise ValueError(
                "Legacy config blocks are no longer supported. "
                "Provide a canonical `pipeline:` list of {gate, params} entries."
            )

        specs = [dict(x) for x in pipeline]

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
