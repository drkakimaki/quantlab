"""Backtest execution using Strategy classes."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import WORKSPACE, QUANTLAB, DATA_ROOTS, StrategyInfo, get_strategies
from ..strategies import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    TrendStrategy,
    TrendStrategyWithGates,
    BacktestConfig,
)
from ..data.resample import load_dukascopy_ohlc
from ..time_filter import EventWindow, build_allow_mask_from_events
from .. import report_periods_equity_only
from .periods import build_periods


def load_period_data(
    symbol: str,
    start: dt.date,
    end: dt.date,
    *,
    need_htf: bool = False,
    need_corr: bool = False,
    corr_symbol: str = "XAGUSD",
    corr2_symbol: str = "EURUSD",
) -> dict[str, Any]:
    """Load all data needed for a period.
    
    Returns:
        dict with keys: prices, bars_5m, bars_15m (optional), prices_xag, prices_eur
    """
    root_5m = DATA_ROOTS["5m_ohlc"]
    root_15m = DATA_ROOTS["15m_ohlc"]
    
    bars_5m = load_dukascopy_ohlc(symbol=symbol, start=start, end=end, root=root_5m)
    prices = bars_5m["close"].astype(float)
    
    data = {
        "prices": prices,
        "bars_5m": bars_5m,
    }
    
    if need_htf:
        bars_15m = load_dukascopy_ohlc(symbol=symbol, start=start, end=end, root=root_15m)
        data["bars_15m"] = bars_15m
    
    if need_corr:
        xag = load_dukascopy_ohlc(symbol=corr_symbol, start=start, end=end, root=root_5m)
        data["prices_xag"] = xag["close"].astype(float)
        
        eur = load_dukascopy_ohlc(symbol=corr2_symbol, start=start, end=end, root=root_5m)
        data["prices_eur"] = eur["close"].astype(float)
    
    return data


def load_fomc_mask(
    index: pd.DatetimeIndex,
    start: dt.date,
    end: dt.date,
    fomc_days_path: Path,
    fomc_cfg: dict | None = None,
) -> pd.Series | None:
    """Build FOMC allow mask for the period."""
    if not fomc_days_path.exists():
        return None
    
    fomc_cfg = fomc_cfg or {}
    utc_hhmm = fomc_cfg.get("utc_hhmm", "19:00")
    pre_hours = fomc_cfg.get("pre_hours", 2.0)
    post_hours = fomc_cfg.get("post_hours", 0.5)
    
    df = pd.read_csv(fomc_days_path)
    if "date" not in df.columns:
        return None
    
    days = [dt.date.fromisoformat(str(x)) for x in df["date"].tolist()]
    days = [d for d in days if start <= d <= end]
    
    if not days:
        return None
    
    hh, mm = (int(x) for x in utc_hhmm.split(":"))
    events = [
        EventWindow(
            ts=pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC)),
            pre=dt.timedelta(hours=pre_hours),
            post=dt.timedelta(hours=post_hours),
        )
        for d in days
    ]
    
    return build_allow_mask_from_events(index, events=events)


def run_backtest(strategy_id: str) -> tuple[bool, str, Path]:
    """Run a backtest using Strategy classes.
    
    Args:
        strategy_id: Strategy key from get_strategies()
    
    Returns:
        (success, output, report_path)
    """
    strategies = get_strategies()
    if strategy_id not in strategies:
        return False, f"Unknown strategy: {strategy_id}", Path(".")
    
    info = strategies[strategy_id]
    
    try:
        # Define periods (config-driven)
        cfg_path = WORKSPACE / "quantlab/configs/trend_based/current.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

        periods = build_periods(cfg)

        config = BacktestConfig()
        
        # Run based on strategy type
        if info.strategy_type == "buy_and_hold":
            results = _run_buy_and_hold(periods, config)
        
        elif info.strategy_type == "trend":
            params = info.params or {}
            results = _run_trend_baseline(periods, config, **params)
        
        elif info.strategy_type == "mean_reversion":
            results = _run_mean_reversion(periods, config)
        
        elif info.strategy_type == "best_trend":
            results = _run_best_trend(periods, config, info.config_path)
        
        else:
            return False, f"Unknown strategy type: {info.strategy_type}", Path(".")
        
        # Generate report
        report_path = info.output_dir / info.output
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        _generate_report(results, info, report_path)
        
        output = f"Generated: {report_path}"
        return True, output, report_path
    
    except Exception as e:
        import traceback
        return False, f"Error: {e}\n{traceback.format_exc()}", Path(".")


def _run_buy_and_hold(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
) -> dict[str, tuple[pd.DataFrame, float, int]]:
    """Run buy and hold strategy."""
    results = {}
    
    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        strategy = BuyAndHoldStrategy()
        result = strategy.run_backtest(data["prices"], config=config)
        
        executions = int((result.df["position"].diff().abs() > 0).sum())
        results[name] = (result.df, result.final_equity - 1000.0, executions)
    
    return results


def _run_trend_baseline(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
    fast: int = 20,
    slow: int = 100,
) -> dict[str, tuple[pd.DataFrame, float, int]]:
    """Run simple trend strategy (no filters)."""
    results = {}
    
    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        strategy = TrendStrategy(fast=fast, slow=slow)
        result = strategy.run_backtest(data["prices"], config=config)
        
        executions = int((result.df["position"].diff().abs() > 0).sum())
        results[name] = (result.df, result.final_equity - 1000.0, executions)
    
    return results


def _run_mean_reversion(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
) -> dict[str, tuple[pd.DataFrame, float, int]]:
    """Run mean reversion strategy."""
    results = {}
    
    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        strategy = MeanReversionStrategy(lookback=50, entry_z=1.0, exit_z=0.0)
        result = strategy.run_backtest(data["prices"], config=config)
        
        executions = int((result.df["position"].diff().abs() > 0).sum())
        results[name] = (result.df, result.final_equity - 1000.0, executions)
    
    return results


def _run_best_trend(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
    config_path: Path | None,
) -> dict[str, tuple[pd.DataFrame, float, int]]:
    """Run best_trend strategy using composable gates."""
    # Load config
    if config_path is None:
        config_path = WORKSPACE / "quantlab/configs/trend_based/current.yaml"
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    symbol = cfg.get("symbol", "XAUUSD")
    corr_symbol = cfg.get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg.get("corr2_symbol", "EURUSD")
    
    # Load FOMC days
    fomc_path = WORKSPACE / cfg.get("time_filter", {}).get("fomc", {}).get(
        "days_csv", "quantlab/data/econ_calendar/fomc_decision_days.csv"
    )
    fomc_cfg = cfg.get("time_filter", {}).get("fomc", {})
    
    results = {}
    
    for name, start, end in periods:
        # Load all data
        data = load_period_data(
            symbol, start, end,
            need_htf=True,
            need_corr=True,
            corr_symbol=corr_symbol,
            corr2_symbol=corr2_symbol,
        )
        
        # Build FOMC mask
        allow_mask = load_fomc_mask(
            data["prices"].index, start, end, fomc_path, fomc_cfg
        )
        
        # Build strategy from config
        strategy = TrendStrategyWithGates.from_config(cfg, allow_mask=allow_mask)
        
        # Build context
        context = {
            "bars_15m": data.get("bars_15m"),
            "prices_xag": data.get("prices_xag"),
            "prices_eur": data.get("prices_eur"),
        }
        
        # Run backtest
        result = strategy.run_backtest(data["prices"], context=context, config=config)
        
        executions = int((result.df["position"].diff().abs() > 0).sum())
        results[name] = (result.df, result.final_equity - 1000.0, executions)
    
    return results


def _generate_report(
    results: dict[str, tuple[pd.DataFrame, float, int]],
    info: StrategyInfo,
    report_path: Path,
) -> None:
    """Generate HTML report."""
    periods = {name: df for name, (df, _, _) in results.items()}
    
    n_trades = {}
    for name, (_, pnl, executions) in results.items():
        if info.strategy_type == "buy_and_hold":
            n_trades[name] = 1
        else:
            n_trades[name] = executions // 2
    
    initial_capital = {name: 1000.0 for name in periods}
    
    title = info.name
    
    report_periods_equity_only(
        periods=periods,
        out_path=report_path,
        title=title,
        freq="5MIN",
        initial_capital=initial_capital,
        n_trades=n_trades,
    )


def get_report_path(strategy_id: str) -> Path | None:
    """Get the expected report path for a strategy."""
    strategies = get_strategies()
    if strategy_id not in strategies:
        return None
    
    info = strategies[strategy_id]
    return info.output_dir / info.output


def report_exists(strategy_id: str) -> bool:
    """Check if a report already exists."""
    path = get_report_path(strategy_id)
    return path is not None and path.exists()