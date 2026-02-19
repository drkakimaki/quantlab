"""Trading strategies (signal/position generators and simple baselines).

This module provides:
1. Base classes: Strategy, StrategyBase, BacktestResult, BacktestConfig, TimeFilterConfig
2. Strategy classes: BuyAndHoldStrategy, MeanReversionStrategy, TrendStrategy, TrendStrategyWithGates
3. Composable gates: SignalGate, HTFConfirmGate, EMASeparationGate, NoChopGate, CorrelationGate
"""

from .base import (
    Strategy,
    StrategyBase,
    BacktestResult,
    BacktestConfig,
    TimeFilterConfig,
)
from .buy_and_hold import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import (
    TrendStrategy,
    TrendStrategyWithGates,
    SignalGate,
    HTFConfirmGate,
    EMASeparationGate,
    NoChopGate,
    CorrelationGate,
    TimeFilterGate,
    RiskGate,
)

__all__ = [
    # Base classes
    "Strategy",
    "StrategyBase",
    "BacktestResult",
    "BacktestConfig",
    "TimeFilterConfig",
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
    "RiskGate",
]
