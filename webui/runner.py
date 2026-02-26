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
from ..data.loaders import load_dukascopy_ohlc
from ..time_filter import (
    EventWindow,
    build_allow_mask_from_events,
    build_allow_mask_from_econ_calendar,
    build_allow_mask_from_months,
)
from .. import report_periods_equity_only, report_robustness
from ..reporting.generate_trades_report import report_periods_trades_html
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


def load_time_filter_mask(
    index: pd.DatetimeIndex,
    start: dt.date,
    end: dt.date,
    *,
    cfg: dict,
    workspace: Path,
) -> pd.Series | None:
    """Load allow mask for the configured time_filter kind.

    Supported kinds:
    - fomc
    - econ_calendar
    - months (months-only force-flat; no event windows)
    """
    tf = cfg.get("time_filter", {}) or {}
    kind = (tf.get("kind") or "fomc").strip().lower()

    # Optional month blocking overlay (force-flat semantics).
    months_cfg = tf.get("months", {}) or {}
    block_months = months_cfg.get("block") or []
    m_mask = build_allow_mask_from_months(index, block=block_months)

    if kind == "months":
        return pd.Series(m_mask, index=index).astype(bool)

    if kind == "fomc":
        fomc_cfg = tf.get("fomc", {}) or {}
        days_csv = fomc_cfg.get("days_csv", "quantlab/data/econ_calendar/fomc_decision_days.csv")
        mask = load_fomc_mask(index, start, end, workspace / str(days_csv), fomc_cfg)

        if mask is None:
            return m_mask
        return (pd.Series(mask, index=index).astype(bool) & pd.Series(m_mask, index=index).astype(bool)).astype(bool)

    if kind == "econ_calendar":
        ec = tf.get("econ_calendar", {}) or {}
        csv_rel = ec.get("csv", "quantlab/data/econ_calendar/usd_important_events.csv")
        rules = ec.get("rules", {}) or {}

        csv_path = str((workspace / str(csv_rel)).resolve())

        mask = build_allow_mask_from_econ_calendar(
            index,
            start=start,
            end=end,
            csv_path=csv_path,
            rules=rules,
        )

        # Optional month blocking overlay.
        months_cfg = tf.get("months", {}) or {}
        block_months = months_cfg.get("block") or []
        m_mask = build_allow_mask_from_months(index, block=block_months)

        if mask is None:
            return m_mask
        return (pd.Series(mask, index=index).astype(bool) & pd.Series(m_mask, index=index).astype(bool)).astype(bool)

    # Unknown kind -> no mask
    return None


def run_backtest(
    strategy_id: str,
    *,
    breakdown: str | None = None,  # three_block | yearly
    record_executions: bool = False,
) -> tuple[bool, str, Path]:
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

        # Fail fast on typos / unknown keys.
        from ..configs.schema import validate_config_dict

        cfg = validate_config_dict(cfg)

        # Optional breakdown override from UI
        if breakdown in {"three_block", "yearly"}:
            cfg = dict(cfg)
            cfg_periods = dict(cfg.get("periods", {}) or {})
            cfg_periods["mode"] = breakdown
            cfg["periods"] = cfg_periods

        periods = build_periods(cfg)

        costs = cfg.get("costs", {}) or {}
        config = BacktestConfig(
            fee_per_lot=float(costs.get("fee_per_lot", 0.0) or 0.0),
            spread_per_lot=float(costs.get("spread_per_lot", 0.0) or 0.0),
            record_executions=bool(record_executions),
        )
        
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
        
        # Generate report (variant naming)
        mode = (cfg.get("periods", {}) or {}).get("mode")
        # We repurpose yearly breakdown as a robustness-style report surface.
        variant = "robustness" if mode == "yearly" else None

        report_path = get_report_path(strategy_id, variant=variant, kind="equity") or (info.output_dir / info.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        baseline_results = None
        if info.strategy_type == "best_trend":
            # Buy & Hold baseline over the same periods (for excess return vs baseline).
            baseline_results = _run_buy_and_hold(periods, config)

        final_title = _generate_report(results, info, report_path, cfg=cfg, baseline_results=baseline_results)

        # Also generate trade breakdown report (convenience for debugging / diagnosis)
        trades_path = get_report_path(strategy_id, variant=variant, kind="trades")
        if trades_path is not None:
            try:
                periods_df = {name: df for name, (df, _, _) in results.items()}
                # Trade breakdown report should keep a clean title (avoid dumping the
                # hyperparam-enriched string into the page title/header).
                base_title = (final_title.split(" + ", 1)[0].strip() if final_title else info.name)
                score_exclude = list(((cfg.get("periods", {}) or {}).get("score_exclude") or []) or [])
                report_periods_trades_html(
                    periods=periods_df,
                    out_path=trades_path,
                    title=base_title + " — trade breakdown",
                    score_exclude=score_exclude,
                )
            except Exception:
                # Don't fail the whole UI run if trade report fails.
                pass

        output = f"Generated: {report_path}" + (f"\nGenerated: {trades_path}" if trades_path is not None else "")
        return True, output, report_path
    
    except Exception as e:
        import traceback
        return False, f"Error: {e}\n{traceback.format_exc()}", Path(".")


def _run_buy_and_hold(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
) -> dict[str, tuple[pd.DataFrame, float, int]]:
    """Run buy and hold strategy.

    Baseline semantics: buy-and-hold with ~100% notional at entry (unlevered).

    We achieve this by:
    - setting leverage=1.0 (so required margin ~= notional)
    - setting lot_per_size per-period so position_size=1 maps to notional ~= initial_capital
    """
    results = {}

    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        prices = data["prices"].astype(float)

        # Per-period sizing to target notional ~= initial_capital at entry.
        px0 = float(prices.iloc[0]) if len(prices) else 1.0
        contract = float(config.contract_size_per_lot)
        cap0 = float(config.initial_capital)
        lot_per_size = cap0 / max(1e-12, (px0 * contract))

        cfg = BacktestConfig(
            initial_capital=float(config.initial_capital),
            leverage=1.0,
            lot_per_size=float(lot_per_size),
            contract_size_per_lot=float(config.contract_size_per_lot),
            lag=int(config.lag),
            max_size=float(config.max_size),
            margin_policy=str(config.margin_policy),
            record_executions=bool(config.record_executions),
            fee_per_lot=float(config.fee_per_lot),
            spread_per_lot=float(config.spread_per_lot),
        )

        strategy = BuyAndHoldStrategy()
        result = strategy.run_backtest(prices, config=cfg)

        executions = int((result.df["position"].diff().abs() > 0).sum())
        results[name] = (result.df, result.final_equity - float(cfg.initial_capital), executions)

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
    
    # time_filter is loaded per-period via load_time_filter_mask (supports fomc or econ_calendar)
    
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
        
        # Build allow mask (time_filter)
        allow_mask = load_time_filter_mask(
            data["prices"].index,
            start,
            end,
            cfg=cfg,
            workspace=WORKSPACE,
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
    *,
    cfg: dict | None = None,
    baseline_results: dict[str, tuple[pd.DataFrame, float, int]] | None = None,
) -> str:
    """Generate HTML report.

    Returns the final title string used in the report.
    """
    periods = {name: df for name, (df, _, _) in results.items()}
    
    n_trades = {}
    for name, (_, pnl, executions) in results.items():
        if info.strategy_type == "buy_and_hold":
            n_trades[name] = 1
        else:
            n_trades[name] = executions // 2
    
    initial_capital = {name: 1000.0 for name in periods}
    
    title = info.name

    # (removed) legacy title enrichment: configs are pipeline-based; keep titles stable.
    score_exclude = list(((cfg.get("periods", {}) or {}).get("score_exclude") or []) or []) if isinstance(cfg, dict) else []

    baseline_periods = {name: df for name, (df, _, _) in (baseline_results or {}).items()} if baseline_results else None

    mode = (cfg.get("periods", {}) or {}).get("mode") if isinstance(cfg, dict) else None
    report_fn = report_robustness if (mode == "yearly") else report_periods_equity_only

    report_fn(
        periods=periods,
        baseline_periods=baseline_periods,
        out_path=report_path,
        title=title,
        initial_capital=initial_capital,
        n_trades=n_trades,
        score_exclude=score_exclude,
    )

    return title


def get_report_path(strategy_id: str, *, variant: str | None = None, kind: str = "equity") -> Path | None:
    """Get the expected report path for a strategy.

    kind:
      - "equity": default performance report
      - "trades": trade breakdown report

    variant:
      - None / "default": default report name
      - "yearly": yearly breakdown report (suffix "_y")
    """
    strategies = get_strategies()
    if strategy_id not in strategies:
        return None

    info = strategies[strategy_id]
    base = info.output_dir / info.output

    # kind suffix
    if kind.strip().lower() in {"trades", "trade", "trade_breakdown"}:
        base = base.with_name(base.stem + "_trades" + base.suffix)

    if variant in {"robustness", "r"}:
        return base.with_name(base.stem + "_robustness" + base.suffix)

    if variant in {"yearly", "y"}:
        # Backward-compat alias (shouldn't be used by new UI code)
        return base.with_name(base.stem + "_robustness" + base.suffix)

    return base


def report_exists(strategy_id: str, *, variant: str | None = None, kind: str = "equity") -> bool:
    """Check if a report already exists."""
    path = get_report_path(strategy_id, variant=variant, kind=kind)
    return path is not None and path.exists()


def report_mtime(strategy_id: str, *, variant: str | None = None, kind: str = "equity") -> float | None:
    """Return report modification time (unix seconds) or None."""
    path = get_report_path(strategy_id, variant=variant, kind=kind)
    if path is None or not path.exists():
        return None
    try:
        return path.stat().st_mtime
    except Exception:
        return None