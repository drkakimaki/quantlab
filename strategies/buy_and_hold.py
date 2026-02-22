"""Buy and hold strategy using the new Strategy interface."""

from __future__ import annotations

import pandas as pd

from .base import StrategyBase, BacktestResult, BacktestConfig


class BuyAndHoldStrategy(StrategyBase):
    """Buy at start, sell at end. Simplest baseline."""

    def __init__(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ):
        self.start = start
        self.end = end

    @property
    def name(self) -> str:
        return "Buy & Hold"

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        """Generate position: 1.0 (baseline) from start to end."""
        px = prices.dropna().copy()
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        start_ts = pd.to_datetime(self.start) if self.start else px.index[0]
        end_ts = pd.to_datetime(self.end) if self.end else px.index[-1]

        # Handle timezone
        idx_tz = getattr(px.index, "tz", None)
        if idx_tz is not None:
            if getattr(start_ts, "tzinfo", None) is None:
                start_ts = start_ts.tz_localize(idx_tz)
            else:
                start_ts = start_ts.tz_convert(idx_tz)
            if getattr(end_ts, "tzinfo", None) is None:
                end_ts = end_ts.tz_localize(idx_tz)
            else:
                end_ts = end_ts.tz_convert(idx_tz)

        pos = pd.Series(0.0, index=px.index)
        mask = (pos.index >= start_ts) & (pos.index <= end_ts)
        pos.loc[mask] = 1.0

        return pos
