from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

from quantlab import report_periods_equity_only
from quantlab.strategies import trend_following_ma_crossover


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _load_daily_paths(root: Path, symbol: str, start: dt.date, end: dt.date) -> list[str]:
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one
    return paths


def load_ohlc_daily(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
    paths = _load_daily_paths(root, symbol, start, end)
    if not paths:
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = pl.scan_parquet(paths).select(["ts", "open", "high", "low", "close"]).sort("ts").collect(engine="streaming")
    out = df.to_pandas().set_index("ts").sort_index()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simple filter contribution analysis for best trend strategy."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trend_based/current.yaml"),
        help="Canonical config YAML.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based"))
    ap.add_argument("--out-name", type=str, default="filter_contributions_simple.html")
    
    # Use a shorter period for faster testing
    ap.add_argument("--start", type=parse_date, default=dt.date(2023, 1, 1))
    ap.add_argument("--end", type=parse_date, default=dt.date(2024, 12, 31))
    
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    def cfg_get(path: str, default=None):
        cur = cfg
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # Load data
    symbol = cfg_get("symbol", "XAUUSD")
    corr_symbol = cfg_get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg_get("corr2_symbol", "EURUSD")
    
    print(f"Loading {symbol} data from {args.start} to {args.end}...")
    
    px_5m = load_ohlc_daily(
        symbol=symbol,
        start=args.start,
        end=args.end,
        root=Path(cfg_get("roots.root_5m_ohlc", "data/dukascopy_5m_ohlc")),
    )
    
    px_15m = load_ohlc_daily(
        symbol=symbol,
        start=args.start,
        end=args.end,
        root=Path(cfg_get("roots.root_15m_ohlc", "data/dukascopy_15m_ohlc")),
    )
    
    # Load correlation data
    cc_xag = None
    cc_eur = None
    
    try:
        cc_xag = load_ohlc_daily(
            symbol=corr_symbol,
            start=args.start,
            end=args.end,
            root=Path(cfg_get("roots.root_5m_ohlc", "data/dukascopy_5m_ohlc")),
        )["close"]
        print(f"Loaded {corr_symbol} data: {len(cc_xag)} bars")
    except Exception as e:
        print(f"Warning: Could not load {corr_symbol}: {e}")
    
    try:
        cc_eur = load_ohlc_daily(
            symbol=corr2_symbol,
            start=args.start,
            end=args.end,
            root=Path(cfg_get("roots.root_5m_ohlc", "data/dukascopy_5m_ohlc")),
        )["close"]
        print(f"Loaded {corr2_symbol} data: {len(cc_eur)} bars")
    except Exception as e:
        print(f"Warning: Could not load {corr2_symbol}: {e}")
    
    print(f"Loaded {len(px_5m)} {symbol} 5m bars, {len(px_15m)} 15m bars")

    # Define key configurations to test
    filter_configs = [
        {
            "name": "1. Baseline (SMA 30/75 + HTF)",
            "description": "Just trend signal with HTF confirmation",
            "params": {
                "fast": cfg_get("trend.fast", 30),
                "slow": cfg_get("trend.slow", 75),
                "fee_bps": cfg_get("costs.fee_bps", 0.0),
                "slippage_bps": cfg_get("costs.slippage_bps", 0.0),
                "htf_confirm": {"bars_15m": px_15m, "rule": cfg_get("htf_confirm.rule", "15min")},
                "ema_sep": None,
                "nochop": None,
                "corr": None,
                "time_filter": None,
                "risk": None,
                "sizing": cfg_get("sizing", {"confirm_size_one": 1.0, "confirm_size_both": 2.0}),
            }
        },
        {
            "name": "2. + EMA-separation",
            "description": "Adds EMA 40/300 separation filter on HTF",
            "params": {
                "fast": cfg_get("trend.fast", 30),
                "slow": cfg_get("trend.slow", 75),
                "fee_bps": cfg_get("costs.fee_bps", 0.0),
                "slippage_bps": cfg_get("costs.slippage_bps", 0.0),
                "htf_confirm": {"bars_15m": px_15m, "rule": cfg_get("htf_confirm.rule", "15min")},
                "ema_sep": cfg_get("ema_sep", {}),
                "nochop": None,
                "corr": None,
                "time_filter": None,
                "risk": None,
                "sizing": cfg_get("sizing", {"confirm_size_one": 1.0, "confirm_size_both": 2.0}),
            }
        },
        {
            "name": "3. + NoChop filter",
            "description": "Adds NoChop filter (EMA20, lookback 40)",
            "params": {
                "fast": cfg_get("trend.fast", 30),
                "slow": cfg_get("trend.slow", 75),
                "fee_bps": cfg_get("costs.fee_bps", 0.0),
                "slippage_bps": cfg_get("costs.slippage_bps", 0.0),
                "htf_confirm": {"bars_15m": px_15m, "rule": cfg_get("htf_confirm.rule", "15min")},
                "ema_sep": cfg_get("ema_sep", {}),
                "nochop": cfg_get("nochop", {}),
                "corr": None,
                "time_filter": None,
                "risk": None,
                "sizing": cfg_get("sizing", {"confirm_size_one": 1.0, "confirm_size_both": 2.0}),
            }
        },
        {
            "name": "4. + Correlation (XAG+EUR)",
            "description": "Adds correlation stability filter",
            "params": {
                "fast": cfg_get("trend.fast", 30),
                "slow": cfg_get("trend.slow", 75),
                "fee_bps": cfg_get("costs.fee_bps", 0.0),
                "slippage_bps": cfg_get("costs.slippage_bps", 0.0),
                "htf_confirm": {"bars_15m": px_15m, "rule": cfg_get("htf_confirm.rule", "15min")},
                "ema_sep": cfg_get("ema_sep", {}),
                "nochop": cfg_get("nochop", {}),
                "corr": _build_corr_module(cfg_get("corr", {}), cc_xag, cc_eur, corr2_symbol),
                "time_filter": None,
                "risk": None,
                "sizing": cfg_get("sizing", {"confirm_size_one": 1.0, "confirm_size_both": 2.0}),
            }
        },
        {
            "name": "5. All filters (current best)",
            "description": "All filters + FOMC + ShockExit",
            "params": {
                "fast": cfg_get("trend.fast", 30),
                "slow": cfg_get("trend.slow", 75),
                "fee_bps": cfg_get("costs.fee_bps", 0.0),
                "slippage_bps": cfg_get("costs.slippage_bps", 0.0),
                "htf_confirm": {"bars_15m": px_15m, "rule": cfg_get("htf_confirm.rule", "15min")},
                "ema_sep": cfg_get("ema_sep", {}),
                "nochop": cfg_get("nochop", {}),
                "corr": _build_corr_module(cfg_get("corr", {}), cc_xag, cc_eur, corr2_symbol),
                "time_filter": None,  # Skip FOMC for simplicity
                "risk": cfg_get("risk", {}),
                "sizing": cfg_get("sizing", {"confirm_size_one": 1.0, "confirm_size_both": 2.0}),
            }
        },
    ]

    # Run backtests
    results = {}
    print(f"\nRunning {len(filter_configs)} backtest configurations...")
    
    for i, config in enumerate(filter_configs):
        print(f"  {i+1}. {config['name']}")
        try:
            bt_result = trend_following_ma_crossover(
                px_5m["close"],
                **config["params"]
            )
            
            if isinstance(bt_result, tuple) and len(bt_result) > 0:
                bt_df = bt_result[0]
                results[config["name"]] = bt_df
                
                # Count trades
                if "position" in bt_df.columns:
                    pos_changes = (bt_df["position"].fillna(0.0).diff().abs() > 0).sum()
                    trades = int(pos_changes // 2)
                    print(f"     ✓ {trades} trades, {len(bt_df)} bars")
                else:
                    print(f"     ✓ {len(bt_df)} bars")
            else:
                results[config["name"]] = None
                print(f"     ✗ Failed")
                
        except Exception as e:
            results[config["name"]] = None
            print(f"     ✗ Error: {str(e)[:100]}")

    # Generate report
    out_path = args.out_dir / args.out_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Count trades
    n_trades = {}
    for name, bt in results.items():
        if bt is not None and "position" in bt.columns:
            pos_changes = (bt["position"].fillna(0.0).diff().abs() > 0).sum()
            n_trades[name] = int(pos_changes // 2)
        else:
            n_trades[name] = 0
    
    title = f"Filter Contributions: {symbol} {args.start} to {args.end}"
    
    report_path = report_periods_equity_only(
        periods=results,
        out_path=out_path,
        title=title,
        freq="5min",
        initial_capital={k: 1000.0 for k in results},
        returns_col="returns_net",
        equity_col="equity",
        n_trades=n_trades,
    )
    
    print(f"\n✅ Report written to: {report_path}")
    
    # Print summary
    print(f"\n📊 Filter Contribution Summary:")
    print("=" * 80)
    for config in filter_configs:
        name = config["name"]
        if name in results and results[name] is not None:
            bt = results[name]
            if "equity" in bt.columns:
                eq = bt["equity"]
                pnl = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
                print(f"{name:40} → {pnl:7.2f}% PnL")
            else:
                print(f"{name:40} → No equity data")
        else:
            print(f"{name:40} → Failed")


def _build_corr_module(corr_cfg, cc_xag, cc_eur, corr2_symbol):
    """Build correlation module with data series."""
    if not corr_cfg or (cc_xag is None and cc_eur is None):
        return None
    
    module = dict(corr_cfg)
    
    if cc_xag is not None and "xag" in module:
        xag_cfg = dict(module["xag"])
        xag_cfg["close"] = cc_xag
        module["xag"] = xag_cfg
    
    if corr2_symbol and cc_eur is not None and "eur" in module:
        eur_cfg = dict(module["eur"])
        eur_cfg["close"] = cc_eur
        module["eur"] = eur_cfg
    
    return module


if __name__ == "__main__":
    main()