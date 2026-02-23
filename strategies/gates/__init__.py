"""Composable gates for TrendStrategyWithGates.

This package holds small, focused gate implementations.
"""

from .types import SignalGate
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
