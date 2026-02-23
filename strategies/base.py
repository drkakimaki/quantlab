"""Base classes and protocols for trading strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Any

import pandas as pd
import numpy as np

from ..engine.backtest import backtest_positions_account_margin

@dataclass
class BacktestResult:
    """Result of a backtest run."""

    df: pd.DataFrame  # Full backtest dataframe (equity, pnl, positions, etc.)
    trades: list[dict] = field(default_factory=list)  # List of trade dicts (entry/exit pairs)

    # Metrics
    final_equity: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    
    # Execution log (one row per position change)
    executions: pd.DataFrame | None = None

    def to_summary(self) -> dict:
        """Return a summary dict for reports."""
        return {
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
        }

@dataclass
class BacktestConfig:
    """Configuration for backtest execution.

    Costs are modeled as absolute per-lot costs:
    - fee_per_lot + spread_per_lot

    For IC Markets Raw Spread (XAUUSD):
        fee_per_lot: 3.50      # $7 RT commission = $3.50/side
        spread_per_lot: 7.00   # $0.07/oz × 100 oz = $7/side
    """

    initial_capital: float = 1000.0
    leverage: float | None = 20.0
    lot_per_size: float = 0.01
    contract_size_per_lot: float = 100.0
    lag: int = 1
    max_size: float = 2.0
    margin_policy: str = "skip_entry"
    record_executions: bool = False

    # Absolute costs per lot per side (recommended)
    fee_per_lot: float = 0.0       # $ per lot commission (per side)
    spread_per_lot: float = 0.0    # $ per lot spread cost (per side, since fills at mid)

    # (legacy bps costs removed)

@runtime_checkable
class Strategy(Protocol):
    """Protocol for trading strategies.

    A strategy generates position signals from price data and optionally
    runs a full backtest. Implementations can be:
    - Simple functions wrapped in a class
    - Complex composable strategies with gates/filters

    Two-level design:
    1. generate_positions() - pure signal generation, no engine params
    2. run_backtest() - default implementation using engine
    """

    @property
    def name(self) -> str:
        """Strategy name for reports and logging."""
        ...

    @property
    def data_requirements(self) -> dict[str, str]:
        """Declare what data this strategy needs beyond the main price series.

        Returns:
            Dict mapping requirement name to description.
            Example: {"bars_15m": "15-minute OHLC for HTF confirmation"}
        """
        ...

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        """Generate position sizes from prices.

        This is the core signal generation logic. It should be pure:
        - No backtest parameters (leverage, costs, etc.)
        - No execution logic
        - Just: prices -> position signal

        Args:
            prices: Main price series (typically 5m close).
            context: Optional additional data needed by the strategy:
                - "bars_15m": DataFrame with 15m OHLC
                - "prices_xag": Series for XAGUSD correlation
                - "prices_eur": Series for EURUSD correlation
                - etc.

        Returns:
            Position sizes aligned with prices index.
            Typically 0/1/2 for long-only, or -1/0/1 for long/short.
        """
        ...

    def run_backtest(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Run a full backtest.

        Default implementation uses backtest_positions_account_margin.
        Subclasses can override for custom behavior.

        Args:
            prices: Main price series.
            context: Additional data for signal generation.
            config: Backtest engine configuration.

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        ...


class StrategyBase:
    """Base class for strategies with default backtest implementation.

    Provides a default run_backtest() that:
    - Calls generate_positions()
    - Runs backtest_positions_account_margin()
    - Computes metrics
    - Extracts trades

    Subclasses only need to implement:
    - name property
    - generate_positions()
    - data_requirements (if needed)
    - default_config (optional, for strategy-specific defaults)
    """

    # Override in subclasses for strategy-specific defaults
    default_max_size: float = 2.0

    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement 'name'")

    @property
    def data_requirements(self) -> dict[str, str]:
        return {}

    def generate_positions(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
    ) -> pd.Series:
        raise NotImplementedError("Subclasses must implement 'generate_positions'")

    def _extract_executions(
        self,
        df: pd.DataFrame,
        prices: pd.Series,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        """Extract execution log (one row per position change).
        
        Fields:
            time: Execution timestamp
            prev_size: Position size before execution
            new_size: Position size after execution
            d_size: Position size change (= new - prev)
            prev_contract_units: Contract units before execution
            new_contract_units: Contract units after execution
            d_contract_units: Contract unit change
            fill_price: Execution price (bar close)
            d_lots: Lot change
            notional: abs(d_contract_units) * fill_price
            fee_per_lot: Fee per lot
            spread_per_lot: Spread per lot
            (legacy bps cost fields removed)
            costs: Total cost
            reason: Optional reason (later from gates/risk modules)
        
        Args:
            df: Backtest dataframe with 'position' column
            prices: Price series for fill prices
            config: BacktestConfig with cost parameters
        
        Returns:
            DataFrame with execution records.
        """
        pos = df["position"]
        lots = df["lots"]
        contract_units = df.get("contract_units")
        
        # Find position changes
        d_pos = pos.diff()
        change_mask = d_pos.abs() > 0
        
        if not change_mask.any():
            return pd.DataFrame(columns=[
                "time",
                "prev_size", "new_size", "d_size",
                "prev_contract_units", "new_contract_units", "d_contract_units",
                "fill_price", "d_lots", "notional",
                "fee_per_lot", "spread_per_lot",
                "costs", "reason",
            ])
        
        cost_per_lot = config.fee_per_lot + config.spread_per_lot
        
        records = []
        for t in pos.index[change_mask]:
            idx = pos.index.get_loc(t)
            if idx == 0:
                continue
            
            prev_size = float(pos.iloc[idx - 1])
            new_size = float(pos.iloc[idx])
            d_size = new_size - prev_size

            if contract_units is not None:
                prev_cu = float(contract_units.iloc[idx - 1])
                new_cu = float(contract_units.iloc[idx])
                d_cu = new_cu - prev_cu
            else:
                # Backward fallback (shouldn't happen in current engine)
                prev_cu = float(prev_size)
                new_cu = float(new_size)
                d_cu = float(d_size)
            
            prev_lots = float(lots.iloc[idx - 1])
            new_lots = float(lots.iloc[idx])
            d_lots = new_lots - prev_lots
            
            fill_price = float(prices.loc[t])
            notional = abs(d_cu) * fill_price
            
            costs = abs(d_lots) * cost_per_lot
            
            records.append({
                "time": t,
                "prev_size": prev_size,
                "new_size": new_size,
                "d_size": d_size,
                "prev_contract_units": prev_cu,
                "new_contract_units": new_cu,
                "d_contract_units": d_cu,
                "fill_price": fill_price,
                "d_lots": d_lots,
                "notional": notional,
                "fee_per_lot": config.fee_per_lot,
                "spread_per_lot": config.spread_per_lot,
                "costs": costs,
                "reason": None,  # Later from gates/risk modules
            })
        
        return pd.DataFrame(records)

    # NOTE: trade extraction helper removed. Use `quantlab.engine.trades.extract_trade_log`.

    def run_backtest(
        self,
        prices: pd.Series,
        *,
        context: dict | None = None,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Default backtest implementation using the unified engine."""
        if config is None:
            config = BacktestConfig(
                max_size=self.default_max_size,
            )

        # Generate positions
        positions = self.generate_positions(prices, context=context)
        # Run backtest engine
        df = backtest_positions_account_margin(
            prices=prices,
            positions_size=positions,
            initial_capital=config.initial_capital,
            leverage=config.leverage,
            lot_per_size=config.lot_per_size,
            contract_size_per_lot=config.contract_size_per_lot,
            fee_per_lot=config.fee_per_lot,
            spread_per_lot=config.spread_per_lot,
            lag=config.lag,
            max_size=config.max_size,
            margin_policy=config.margin_policy,
        )

        # Compute metrics (canonical helpers live in quantlab.engine.metrics)
        from ..engine.metrics import max_drawdown as _max_dd, sharpe as _sharpe, n_trades_from_position

        equity = df["equity"].astype(float)
        returns = df["returns_net"].astype(float)

        final_equity = float(equity.iloc[-1])
        total_return = (final_equity / config.initial_capital - 1.0) * 100

        # Canonical trade count: contiguous position segments
        trade_count = int(n_trades_from_position(df, pos_col="position"))

        # Canonical Sharpe: computed on daily returns derived from equity.
        sharpe = float(_sharpe(equity))

        # Max drawdown as percent
        max_drawdown = float(_max_dd(equity)) * 100

        # Trade log (canonical)
        from ..engine.trades import extract_trade_log

        trades = extract_trade_log(df).to_dict(orient="records")
        
        # Extract executions if requested
        executions = None
        if config.record_executions:
            executions = self._extract_executions(df, prices, config)

        return BacktestResult(
            df=df,
            trades=trades,
            final_equity=final_equity,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            trade_count=trade_count,
            executions=executions,
        )
