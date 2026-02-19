"""Strategy definitions and path configuration."""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]  # workspace root
QUANTLAB = WORKSPACE / "quantlab"
REPORTS_BASELINES = QUANTLAB / "reports" / "baselines"
REPORTS_TREND = QUANTLAB / "reports" / "trend_based"
CONFIG_PATH = QUANTLAB / "configs" / "trend_based" / "current.yaml"

DATA_ROOTS = {
    "5m_ohlc": QUANTLAB / "data" / "dukascopy_5m_ohlc",
    "15m_ohlc": QUANTLAB / "data" / "dukascopy_15m_ohlc",
    "5m": QUANTLAB / "data" / "dukascopy_5m",
}


@dataclass
class StrategyInfo:
    """Strategy definition."""
    id: str
    name: str
    strategy_type: str  # 'buy_and_hold', 'mean_reversion', 'trend', 'best_trend'
    output: str
    output_dir: Path
    params: dict[str, Any] | None = None
    
    # For best_trend, we load from config
    config_path: Path | None = None


def get_strategies() -> dict[str, StrategyInfo]:
    """Get all available strategies."""
    return {
        "buy_and_hold": StrategyInfo(
            id="buy_and_hold",
            name="Buy & Hold",
            strategy_type="buy_and_hold",
            output="buy_and_hold.html",
            output_dir=REPORTS_BASELINES,
        ),
        "trend_baseline": StrategyInfo(
            id="trend_baseline",
            name="Trend (MA 20/100)",
            strategy_type="trend",
            output="trend.html",
            output_dir=REPORTS_BASELINES,
            params={"fast": 20, "slow": 100},
        ),
        "mean_reversion": StrategyInfo(
            id="mean_reversion",
            name="Mean Reversion (Z-score)",
            strategy_type="mean_reversion",
            output="mean_reversion.html",
            output_dir=REPORTS_BASELINES,
        ),
        "best_trend": StrategyInfo(
            id="best_trend",
            name="Best Trend (all filters)",
            strategy_type="best_trend",
            output="best_trend.html",
            output_dir=REPORTS_TREND,
            config_path=CONFIG_PATH,
        ),
    }