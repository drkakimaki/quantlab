"""Trend following strategies.

This module provides:
1. TrendStrategy - Simple MA crossover baseline (e.g., 20/100)
2. TrendStrategyWithGates - Composable trend strategy with signal gates (from YAML config)
"""

from __future__ import annotations

import pandas as pd

from .base import StrategyBase

# Import gate classes for typing and requirement inference (by class name).
from .gates import SignalGate


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

class TrendStrategy(StrategyBase):
    """Simple trend following strategy using MA crossover.

    Baseline strategy without any filters.

    - ma_kind="sma": buy when fast SMA > slow SMA
    - ma_kind="ema": buy when fast EMA > slow EMA
    """

    def __init__(self, fast: int = 20, slow: int = 100, *, ma_kind: str = "sma"):
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast < slow")
        mk = str(ma_kind).strip().lower()
        if mk not in {"sma", "ema"}:
            raise ValueError("ma_kind must be one of: sma, ema")
        self.fast = int(fast)
        self.slow = int(slow)
        self.ma_kind = mk

    @property
    def name(self) -> str:
        k = "SMA" if self.ma_kind == "sma" else "EMA"
        return f"Trend({k} {self.fast}/{self.slow})"

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        """Generate positions: 1.0 when fast MA > slow MA."""
        px = prices.dropna().astype(float)
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        if self.ma_kind == "ema":
            ma_fast = px.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
            ma_slow = px.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        else:
            ma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
            ma_slow = px.rolling(self.slow, min_periods=self.slow).mean()

        pos = (ma_fast > ma_slow).astype(float)
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
        ma_kind: str = "sma",
        gates: list[SignalGate] | None = None,
        requirements: dict[str, str] | None = None,
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.ma_kind = str(ma_kind).strip().lower()
        if self.ma_kind not in {"sma", "ema"}:
            raise ValueError("ma_kind must be one of: sma, ema")
        self.gates = list(gates or [])
        self._requirements = dict(requirements or {})

    @property
    def name(self) -> str:
        gate_names = [g.name for g in self.gates]
        k = "SMA" if self.ma_kind == "sma" else "EMA"
        return f"Trend({k} {self.fast}/{self.slow}) + [{', '.join(gate_names)}]"

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

        if self.ma_kind == "ema":
            ma_fast = px.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
            ma_slow = px.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        else:
            ma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
            ma_slow = px.rolling(self.slow, min_periods=self.slow).mean()

        pos = (ma_fast > ma_slow).astype(float)

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
        ma_kind = cfg.get("trend", {}).get("ma_kind", "sma")

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
        return cls(fast=fast, slow=slow, ma_kind=ma_kind, gates=gates, requirements=reqs)
