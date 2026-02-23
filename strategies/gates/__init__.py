"""Composable gates for TrendStrategyWithGates.

This package holds small, focused gate implementations.
"""

from .types import SignalGate
from .filters import HTFConfirmGate, EMASeparationGate, NoChopGate, CorrelationGate, TimeFilterGate
from .sizing import EMAStrengthSizingGate, SeasonalitySizeCapGate
from .churn import ChurnGate
from .exits import (
    MidDurationLossLimiterGate,
    TimeStopGate,
    ProfitMilestoneGate,
    RollingMaxExitGate,
    RiskGate,
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
    "TimeStopGate",
    "ProfitMilestoneGate",
    "RollingMaxExitGate",
    "RiskGate",
]
