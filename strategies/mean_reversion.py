"""Mean reversion strategy using the new Strategy interface."""

from __future__ import annotations

import pandas as pd

from .base import StrategyBase, BacktestResult, BacktestConfig


class MeanReversionStrategy(StrategyBase):
    """Mean reversion strategy using rolling z-score.

    Logic (long-only default):
    - Compute z = (price - rolling_mean) / rolling_std
    - Enter long when z <= -entry_z (oversold)
    - Exit to flat when z >= -exit_z (back to normal)

    If long_only=False, also shorts symmetrically:
    - Enter short when z >= +entry_z (overbought)
    - Exit short when z <= +exit_z
    """

    # Mean reversion uses single lot sizing
    default_max_size = 1.0

    def __init__(
        self,
        *,
        lookback: int = 50,
        entry_z: float = 1.0,
        exit_z: float = 0.0,
        long_only: bool = True,
    ):
        if lookback <= 1:
            raise ValueError("lookback must be > 1")
        if entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if exit_z < 0:
            raise ValueError("exit_z must be >= 0")
        if exit_z > entry_z:
            raise ValueError("exit_z must be <= entry_z")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.long_only = long_only

    @property
    def name(self) -> str:
        direction = "long" if self.long_only else "long/short"
        return f"Mean Reversion(z={self.entry_z}/{self.exit_z}, {direction})"

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        """Generate positions using z-score mean reversion."""
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a Series")

        px = prices.dropna().copy()
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        if px.empty:
            raise ValueError("empty price series")

        # Compute z-score
        mu = px.rolling(self.lookback, min_periods=self.lookback).mean()
        sd = px.rolling(self.lookback, min_periods=self.lookback).std(ddof=0)
        z = (px - mu) / sd

        # State machine for position tracking
        pos = pd.Series(0.0, index=px.index)
        state = 0.0

        for t, zi in z.items():
            if pd.isna(zi):
                pos.loc[t] = state
                continue

            if self.long_only:
                # Long only: enter when oversold, exit when normalized
                if state == 0.0 and zi <= -self.entry_z:
                    state = 1.0
                elif state == 1.0 and zi >= -self.exit_z:
                    state = 0.0
            else:
                # Long/short: symmetric entries
                if state == 0.0:
                    if zi <= -self.entry_z:
                        state = 1.0
                    elif zi >= self.entry_z:
                        state = -1.0
                elif state == 1.0:
                    if zi >= -self.exit_z:
                        state = 0.0
                elif state == -1.0:
                    if zi <= self.exit_z:
                        state = 0.0

            pos.loc[t] = state

        return pos
