from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from quantlab import report_periods_equity_only
from quantlab.strategies import trend_following_ma_crossover
from quantlab.time_filter import EventWindow, build_allow_mask_from_events


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
        description="Best trend report with yearly breakdowns instead of period setup."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trend_based/current.yaml"),
        help="Canonical config YAML (defaults to configs/trend_based/current.yaml).",
    )

    # Defaults below may be overridden by the config file.
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--corr-symbol", type=str, default=None)
    ap.add_argument("--corr2-symbol", type=str, default=None)

    ap.add_argument("--root-5m-ohlc", type=Path, default=None)
    ap.add_argument("--root-15m-ohlc", type=Path, default=None)

    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based"))
    ap.add_argument("--out-name", type=str, default="best_trend_yearly.html")

    # Yearly breakdown instead of periods
    ap.add_argument("--start-year", type=int, default=2021, help="First year to include")
    ap.add_argument("--end-year", type=int, default=2026, help="Last year to include")

    ap.add_argument("--fee-bps", type=float, default=None)
    ap.add_argument("--slippage-bps", type=float, default=None)

    # Core trend / HTF rule (optional overrides)
    ap.add_argument("--fast", type=int, default=None, help="Base/HTF SMA fast window")
    ap.add_argument("--slow", type=int, default=None, help="Base/HTF SMA slow window")
    ap.add_argument("--htf-rule", type=str, default=None, help="HTF resample rule (canonical: 15min)")

    # Module toggles (default ON, match best config)
    ap.add_argument("--no-ema-sep", action="store_true", help="Disable HTF EMA separation filter")
    ap.add_argument("--no-nochop", action="store_true", help="Disable HTF no-chop filter")
    ap.add_argument("--no-corr", action="store_true", help="Disable corr stability filter (then sizing uses confirm_size_one always)")

    # Time filter: FOMC (default ON in best config)
    ap.add_argument("--no-fomc", action="store_true", help="Disable FOMC time filter")
    ap.add_argument("--fomc-days", type=Path, default=Path("data/econ_calendar/fomc_decision_days.csv"))
    ap.add_argument("--fomc-utc-hhmm", type=str, default="19:00", help="Approx decision time in UTC")
    ap.add_argument("--fomc-pre-hours", type=float, default=2.0)
    ap.add_argument("--fomc-post-hours", type=float, default=0.5)

    args = ap.parse_args()

    # Load canonical config (YAML). CLI values override config values.
    if args.config is None or not args.config.exists():
        raise FileNotFoundError(f"Config YAML not found: {args.config}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    def cfg_get(path: str, default=None):
        cur = cfg
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # Symbols / roots
    args.symbol = args.symbol or cfg_get("symbol", "XAUUSD")
    args.corr_symbol = args.corr_symbol or cfg_get("corr_symbol", "XAGUSD")
    args.corr2_symbol = args.corr2_symbol or cfg_get("corr2_symbol", "EURUSD")

    args.root_5m_ohlc = args.root_5m_ohlc or Path(cfg_get("roots.root_5m_ohlc", "data/dukascopy_5m_ohlc"))
    args.root_15m_ohlc = args.root_15m_ohlc or Path(cfg_get("roots.root_15m_ohlc", "data/dukascopy_15m_ohlc"))

    # Core trend / HTF params
    args.fast = int(args.fast) if args.fast is not None else int(cfg_get("trend.fast", 30))
    args.slow = int(args.slow) if args.slow is not None else int(cfg_get("trend.slow", 75))
    args.htf_rule = args.htf_rule or str(cfg_get("htf_confirm.rule", "15min"))

    # Costs
    args.fee_bps = float(args.fee_bps) if args.fee_bps is not None else float(cfg_get("costs.fee_bps", 0.0))
    args.slippage_bps = (
        float(args.slippage_bps) if args.slippage_bps is not None else float(cfg_get("costs.slippage_bps", 0.0))
    )

    # Module defaults from config
    ema_sep_cfg = cfg_get("ema_sep", None)
    nochop_cfg = cfg_get("nochop", None)
    corr_cfg = cfg_get("corr", None)
    sizing_cfg = cfg_get("sizing", None)
    risk_cfg = cfg_get("risk", None)
    time_filter_cfg = cfg_get("time_filter", None)

    # Create yearly periods
    yearly_periods = {}
    for year in range(args.start_year, args.end_year + 1):
        yearly_periods[str(year)] = (dt.date(year, 1, 1), dt.date(year, 12, 31))

    # Load data for each year
    yearly_data = {}
    for year_name, (year_start, year_end) in yearly_periods.items():
        try:
            # Load 5m data
            px_5m = load_ohlc_daily(
                symbol=args.symbol,
                start=year_start,
                end=year_end,
                root=args.root_5m_ohlc,
            )
            
            # Load 15m data for HTF
            px_15m = load_ohlc_daily(
                symbol=args.symbol,
                start=year_start,
                end=year_end,
                root=args.root_15m_ohlc,
            )
            
            # Load correlation symbols if needed
            cc_xag = None
            cc_eur = None
            if not args.no_corr:
                try:
                    cc_xag = load_ohlc_daily(
                        symbol=args.corr_symbol,
                        start=year_start,
                        end=year_end,
                        root=args.root_5m_ohlc,
                    )["close"]
                except Exception:
                    print(f"Warning: Could not load {args.corr_symbol} data for {year_name}")
                
                if args.corr2_symbol:
                    try:
                        cc_eur = load_ohlc_daily(
                            symbol=args.corr2_symbol,
                            start=year_start,
                            end=year_end,
                            root=args.root_5m_ohlc,
                        )["close"]
                    except Exception:
                        print(f"Warning: Could not load {args.corr2_symbol} data for {year_name}")
            
            yearly_data[year_name] = {
                "px_5m": px_5m,
                "px_15m": px_15m,
                "cc_xag": cc_xag,
                "cc_eur": cc_eur,
            }
            print(f"Loaded data for {year_name}")
            
        except FileNotFoundError as e:
            print(f"Skipping {year_name}: {e}")
            continue

    if not yearly_data:
        raise ValueError("No yearly data loaded. Check if data files exist.")

    # Build module configs
    ema_sep_mod = None if args.no_ema_sep else (ema_sep_cfg if isinstance(ema_sep_cfg, dict) else {})
    nochop_mod = None if args.no_nochop else (nochop_cfg if isinstance(nochop_cfg, dict) else {})
    
    corr_mod_template = None
    if not args.no_corr:
        base_corr = corr_cfg if isinstance(corr_cfg, dict) else {}
        corr_mod_template = dict(base_corr)
    
    sizing_mod = sizing_cfg if isinstance(sizing_cfg, dict) else {"confirm_size_one": 1.0, "confirm_size_both": 2.0}
    
    risk_mod = {
        "shock_exit_abs_ret": float(cfg_get("risk.shock_exit_abs_ret", 0.006)),
        "shock_exit_sigma_k": float(cfg_get("risk.shock_exit_sigma_k", 0.0)),
        "shock_exit_sigma_window": int(cfg_get("risk.shock_exit_sigma_window", 96)),
        "shock_cooldown_bars": int(cfg_get("risk.shock_cooldown_bars", 0)),
        "segment_ttl_bars": int(cfg_get("risk.segment_ttl_bars", 0)),
    }

    # Run backtest for each year
    yearly_results = {}
    for year_name, data in yearly_data.items():
        print(f"Running backtest for {year_name}...")
        
        px_5m = data["px_5m"]
        px_15m = data["px_15m"]
        cc_xag = data["cc_xag"]
        cc_eur = data["cc_eur"]
        
        # Create FOMC mask if enabled
        allow_mask = None
        if not args.no_fomc and args.fomc_days.exists():
            try:
                import pandas as pd
                fomc_days = pd.read_csv(args.fomc_days, parse_dates=["date"])
                fomc_days = fomc_days["date"].dt.tz_localize("UTC")
                
                year_start = yearly_periods[year_name][0]
                year_end = yearly_periods[year_name][1]
                
                # Convert dates to UTC timestamps for comparison
                year_start_ts = pd.Timestamp(year_start).tz_localize("UTC")
                year_end_ts = pd.Timestamp(year_end).tz_localize("UTC")
                
                year_fomc = fomc_days[(fomc_days >= year_start_ts) & 
                                      (fomc_days <= year_end_ts)]
                
                hh, mm = (int(x) for x in args.fomc_utc_hhmm.split(":"))
                events = [
                    EventWindow(
                        ts=pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC)),
                        pre=dt.timedelta(hours=float(args.fomc_pre_hours)),
                        post=dt.timedelta(hours=float(args.fomc_post_hours)),
                    )
                    for d in year_fomc
                ]
                
                from quantlab.time_filter import build_allow_mask_from_events
                allow_mask = build_allow_mask_from_events(px_5m.index, events=events)
            except Exception as e:
                print(f"Warning: Could not create FOMC mask for {year_name}: {e}")
        
        # Build correlation module
        corr_mod = None
        if corr_mod_template is not None:
            c = dict(corr_mod_template)
            if cc_xag is not None:
                xag = dict(c.get("xag", {}) or {})
                xag["close"] = cc_xag
                c["xag"] = xag
            if args.corr2_symbol and cc_eur is not None:
                eur = dict(c.get("eur", {}) or {})
                eur["close"] = cc_eur
                c["eur"] = eur
            corr_mod = c
        
        # Build time filter module
        tf_mod = None
        if allow_mask is not None:
            tf_mode = "force_flat"
            tf_entry_shift = 1
            if isinstance(time_filter_cfg, dict):
                tf_mode = str(time_filter_cfg.get("mode") or tf_mode)
                tf_entry_shift = int(time_filter_cfg.get("entry_shift") or tf_entry_shift)
            tf_mod = {"allow_mask": allow_mask, "mode": tf_mode, "entry_shift": tf_entry_shift}
        
        # Build HTF module
        htf_mod = None
        if isinstance(cfg_get("htf_confirm", None), dict):
            htf_mod = {"bars_15m": px_15m, "rule": args.htf_rule}
        
        # Run strategy
        try:
            bt, _, exec_count = trend_following_ma_crossover(
                px_5m["close"],
                fast=args.fast,
                slow=args.slow,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                htf_confirm=htf_mod,
                time_filter=tf_mod,
                ema_sep=ema_sep_mod,
                nochop=nochop_mod,
                corr=corr_mod,
                sizing=sizing_mod,
                risk=risk_mod,
            )
            yearly_results[year_name] = bt
            print(f"  Completed {year_name}: {len(bt)} bars, {exec_count} executions")
        except Exception as e:
            print(f"  Error in {year_name}: {e}")
            import traceback
            traceback.print_exc()
            yearly_results[year_name] = None

    # Generate report
    out_path = args.out_dir / args.out_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build title with hyperparameters
    title_parts = [
        f"Best trend (yearly) {args.symbol}",
        f"SMA {args.fast}/{args.slow}",
    ]
    
    if not args.no_ema_sep:
        title_parts.append("EMA-sep")
    if not args.no_nochop:
        title_parts.append("NoChop")
    if not args.no_corr:
        title_parts.append(f"Corr({args.corr_symbol}+{args.corr2_symbol})")
    if not args.no_fomc:
        title_parts.append("FOMC")
    
    title = " + ".join(title_parts)
    
    # Count trades for each year
    n_trades = {}
    for year_name, bt in yearly_results.items():
        if bt is not None and "position" in bt.columns:
            # Count position changes (entry+exit pairs)
            pos_changes = (bt["position"].fillna(0.0).diff().abs() > 0).sum()
            n_trades[year_name] = int(pos_changes // 2)
        else:
            n_trades[year_name] = 0
    
    # Generate report - use 1000.0 as initial capital (matches original report)
    initial_capital = {k: 1000.0 for k in yearly_results}
    
    report_path = report_periods_equity_only(
        periods=yearly_results,
        out_path=out_path,
        title=title,
        freq="5min",
        initial_capital=initial_capital,
        returns_col="returns_net",
        equity_col="equity",
        n_trades=n_trades,
    )
    
    print(f"\nReport written to: {report_path}")
    print(f"Years included: {list(yearly_results.keys())}")


if __name__ == "__main__":
    main()