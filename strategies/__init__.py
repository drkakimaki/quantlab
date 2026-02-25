"""Trading strategies (signal/position generators and simple baselines).

This module provides:
1. Base classes: Strategy, StrategyBase, BacktestResult, BacktestConfig
2. Strategy classes: BuyAndHoldStrategy, MeanReversionStrategy, TrendStrategy, TrendStrategyWithGates
3. Composable gates: see `quantlab.strategies.gates`
"""

from .base import (
    Strategy,
    StrategyBase,
    BacktestResult,
    BacktestConfig,
)
from .buy_and_hold import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import (
    TrendStrategy,
    TrendStrategyWithGates,
)

# Gates live under strategies.gates (not re-exported from trend_following.py).
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

__all__ = [
    # Base classes
    "Strategy",
    "StrategyBase",
    "BacktestResult",
    "BacktestConfig",
    # Strategy classes
    "BuyAndHoldStrategy",
    "MeanReversionStrategy",
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
