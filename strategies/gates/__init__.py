"""Composable gates for TrendStrategyWithGates.

This package holds small, focused gate implementations.
"""

from .types import SignalGate
from .registry import GateSpec, make_gate, register_factory, register_gate, registered_gates
from .filters import HTFConfirmGate, EMASeparationGate, NoChopGate, CorrelationGate, TimeFilterGate
from .sizing import EMAStrengthSizingGate, SeasonalitySizeCapGate
from .churn import ChurnGate
from .exits import (
    MidDurationLossLimiterGate,
    NoRecoveryExitGate,
    ProfitMilestoneGate,
    RollingMaxExitGate,
    ShockExitGate,
)

__all__ = [
    "SignalGate",
    # registry
    "GateSpec",
    "register_gate",
    "register_factory",
    "make_gate",
    "registered_gates",
    # filters
    "HTFConfirmGate",
    "EMASeparationGate",
    "NoChopGate",
    "CorrelationGate",
    "TimeFilterGate",
    # sizing
    "EMAStrengthSizingGate",
    "SeasonalitySizeCapGate",
    # churn
    "ChurnGate",
    # exits
    "MidDurationLossLimiterGate",
    "NoRecoveryExitGate",
    "ProfitMilestoneGate",
    "RollingMaxExitGate",
    "ShockExitGate",
]
