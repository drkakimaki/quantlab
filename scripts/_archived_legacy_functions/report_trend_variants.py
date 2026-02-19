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
        description="Best trend report runner (clean version; only supports current best config knobs)."
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
    ap.add_argument("--out-name", type=str, default="best_trend.html")

    ap.add_argument("--p0-start", type=parse_date, default=None)
    ap.add_argument("--p1-end", type=parse_date, default=None)
    ap.add_argument("--p2-start", type=parse_date, default=None)
    ap.add_argument("--p2-end", type=parse_date, default=None)
    ap.add_argument("--p3-start", type=parse_date, default=None)
    ap.add_argument("--p3-end", type=parse_date, default=None)

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

    # Optional NoChop semantics / exit (experiment knobs)
    ap.add_argument(
        "--nochop-entry-held",
        action="store_true",
        help="Use NoChop as an entry-only (segment-held) gate instead of continuous gating.",
    )
    ap.add_argument(
        "--nochop-exit-bad-bars",
        type=int,
        default=0,
        help="If >0: force-flat for rest of segment if NoChop is bad for N consecutive 5m bars while in-position.",
    )

    # Optional shock kill-switch (experiment knobs)
    ap.add_argument(
        "--shock-exit-abs-ret",
        type=float,
        default=None,
        help="If >0: shock exit when abs(5m return) >= threshold (canonical set via config YAML).",
    )
    ap.add_argument(
        "--shock-exit-sigma-k",
        type=float,
        default=None,
        help="If >0: shock exit when abs(5m return) >= k * rolling_std(returns) (canonical set via config YAML).",
    )
    ap.add_argument(
        "--shock-exit-sigma-window",
        type=int,
        default=None,
        help="Rolling window (bars) for sigma shock exit (canonical set via config YAML).",
    )
    ap.add_argument(
        "--shock-cooldown-bars",
        type=int,
        default=None,
        help="If >0: after a shock triggers, force flat for N base bars (cooldown) (canonical set via config YAML).",
    )

    # Optional segment TTL (time-stop)
    ap.add_argument(
        "--segment-ttl-bars",
        type=int,
        default=None,
        help="If >0: force-flat for rest of segment after holding >= N base bars since entry (time stop) (canonical set via config YAML).",
    )

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
    # HTF confirm is a module; rule lives under htf_confirm.rule
    args.htf_rule = args.htf_rule or str(cfg_get("htf_confirm.rule", "15min"))

    # Periods
    def cfg_date(key: str, fallback: dt.date | None = None) -> dt.date | None:
        v = cfg_get(f"periods.{key}", None)
        if v is None:
            return fallback
        return dt.date.fromisoformat(str(v))

    args.p0_start = args.p0_start or cfg_date("p0_start", dt.date(2021, 1, 1))
    args.p1_end = args.p1_end or cfg_date("p1_end", dt.date(2022, 12, 31))
    args.p2_start = args.p2_start or cfg_date("p2_start", dt.date(2023, 1, 1))
    args.p2_end = args.p2_end or cfg_date("p2_end", dt.date(2025, 12, 31))
    args.p3_start = args.p3_start or cfg_date("p3_start", dt.date(2026, 1, 1))
    args.p3_end = args.p3_end or cfg_date("p3_end", None)

    p3_end = args.p3_end or dt.date.today()

    # Costs
    args.fee_bps = float(args.fee_bps) if args.fee_bps is not None else float(cfg_get("costs.fee_bps", 0.0))
    args.slippage_bps = (
        float(args.slippage_bps) if args.slippage_bps is not None else float(cfg_get("costs.slippage_bps", 0.0))
    )

    # Module defaults from config (superset YAML): a module is ON if its block is present (non-null).
    ema_sep_cfg = cfg_get("ema_sep", None)
    nochop_cfg = cfg_get("nochop", None)
    corr_cfg = cfg_get("corr", None)
    sizing_cfg = cfg_get("sizing", None)
    risk_cfg = cfg_get("risk", None)
    time_filter_cfg = cfg_get("time_filter", None)

    if ema_sep_cfg is None:
        args.no_ema_sep = True
    if nochop_cfg is None:
        args.no_nochop = True
    if corr_cfg is None:
        args.no_corr = True

    # Time filter (currently: FOMC)
    # Backwards-compat: also accept legacy top-level "fomc" block.
    tf_kind = None
    if isinstance(time_filter_cfg, dict):
        tf_kind = str(time_filter_cfg.get("kind") or "").strip().lower() or None

    if tf_kind is None and isinstance(cfg_get("fomc", None), dict):
        # Legacy style
        tf_kind = "fomc"
        time_filter_cfg = {"kind": "fomc", "mode": "force_flat", "entry_shift": 1, "fomc": cfg_get("fomc", {})}

    if tf_kind != "fomc":
        args.no_fomc = True

    # FOMC params (from time_filter.fomc.* when present)
    if not args.no_fomc:
        fomc_block = (time_filter_cfg or {}).get("fomc", {}) if isinstance(time_filter_cfg, dict) else {}
        args.fomc_days = args.fomc_days or Path(fomc_block.get("days_csv", "data/econ_calendar/fomc_decision_days.csv"))
        args.fomc_utc_hhmm = args.fomc_utc_hhmm or str(fomc_block.get("utc_hhmm", "19:00"))
        args.fomc_pre_hours = float(args.fomc_pre_hours) if args.fomc_pre_hours is not None else float(fomc_block.get("pre_hours", 2.0))
        args.fomc_post_hours = float(args.fomc_post_hours) if args.fomc_post_hours is not None else float(fomc_block.get("post_hours", 0.5))

    # NoChop semantics (from nochop block; CLI flags can override)
    if args.nochop_entry_held is False and isinstance(nochop_cfg, dict) and bool(nochop_cfg.get("entry_held", False)):
        args.nochop_entry_held = True
    if args.nochop_exit_bad_bars == 0 and isinstance(nochop_cfg, dict):
        args.nochop_exit_bad_bars = int(nochop_cfg.get("exit_bad_bars", 0) or 0)

    # Risk / exits
    args.shock_exit_abs_ret = (
        float(args.shock_exit_abs_ret) if args.shock_exit_abs_ret is not None else float(cfg_get("risk.shock_exit_abs_ret", 0.0))
    )
    args.shock_exit_sigma_k = (
        float(args.shock_exit_sigma_k) if args.shock_exit_sigma_k is not None else float(cfg_get("risk.shock_exit_sigma_k", 0.0))
    )
    args.shock_exit_sigma_window = (
        int(args.shock_exit_sigma_window) if args.shock_exit_sigma_window is not None else int(cfg_get("risk.shock_exit_sigma_window", 96))
    )
    args.shock_cooldown_bars = (
        int(args.shock_cooldown_bars) if args.shock_cooldown_bars is not None else int(cfg_get("risk.shock_cooldown_bars", 0))
    )
    args.segment_ttl_bars = (
        int(args.segment_ttl_bars) if args.segment_ttl_bars is not None else int(cfg_get("risk.segment_ttl_bars", 0))
    )

    def load_period(symbol: str, start: dt.date, end: dt.date) -> tuple[pd.Series, pd.DataFrame]:
        bars5 = load_ohlc_daily(symbol=symbol, start=start, end=end, root=args.root_5m_ohlc)
        bars15 = load_ohlc_daily(symbol=symbol, start=start, end=end, root=args.root_15m_ohlc)
        return bars5["close"].astype(float).copy(), bars15

    # Periods
    px_2122, htf_2122 = load_period(args.symbol, args.p0_start, args.p1_end)
    px_2325, htf_2325 = load_period(args.symbol, args.p2_start, args.p2_end)
    px_2026, htf_2026 = load_period(args.symbol, args.p3_start, p3_end)

    corr_xag_2122 = load_ohlc_daily(symbol=args.corr_symbol, start=args.p0_start, end=args.p1_end, root=args.root_5m_ohlc)[
        "close"
    ].astype(float)
    corr_xag_2325 = load_ohlc_daily(symbol=args.corr_symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)[
        "close"
    ].astype(float)
    corr_xag_2026 = load_ohlc_daily(symbol=args.corr_symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)[
        "close"
    ].astype(float)

    corr_eur_2122 = load_ohlc_daily(symbol=args.corr2_symbol, start=args.p0_start, end=args.p1_end, root=args.root_5m_ohlc)[
        "close"
    ].astype(float)
    corr_eur_2325 = load_ohlc_daily(symbol=args.corr2_symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)[
        "close"
    ].astype(float)
    corr_eur_2026 = load_ohlc_daily(symbol=args.corr2_symbol, start=args.p3_start, end=p3_end, root=args.root_5m_ohlc)[
        "close"
    ].astype(float)

    def _fomc_allow_mask(index: pd.DatetimeIndex, *, start: dt.date, end: dt.date) -> pd.Series | None:
        if args.no_fomc:
            return None
        if not args.fomc_days.exists():
            raise FileNotFoundError(f"FOMC days file not found: {args.fomc_days}")

        df = pd.read_csv(args.fomc_days)
        if "date" not in df.columns:
            raise ValueError("FOMC days CSV must have a 'date' column")
        days = [dt.date.fromisoformat(str(x)) for x in df["date"].tolist()]
        days = [d for d in days if start <= d <= end]

        hh, mm = (int(x) for x in args.fomc_utc_hhmm.split(":"))
        events = [
            EventWindow(
                ts=pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC)),
                pre=dt.timedelta(hours=float(args.fomc_pre_hours)),
                post=dt.timedelta(hours=float(args.fomc_post_hours)),
            )
            for d in days
        ]
        return build_allow_mask_from_events(index, events=events)

    fomc_2122 = _fomc_allow_mask(px_2122.index, start=args.p0_start, end=args.p1_end)
    fomc_2325 = _fomc_allow_mask(px_2325.index, start=args.p2_start, end=args.p2_end)
    fomc_2026 = _fomc_allow_mask(px_2026.index, start=args.p3_start, end=p3_end)

    # Pull module params from YAML (single source of truth) for report title readability.
    # Even if a module is disabled, we compute defaults here to keep the title builder simple.
    ema_fast = int(cfg_get("ema_sep.ema_fast", 40))
    ema_slow = int(cfg_get("ema_sep.ema_slow", 300))
    atr_n = int(cfg_get("ema_sep.atr_n", 20))
    sep_k = float(cfg_get("ema_sep.sep_k", 0.05))

    nochop_ema = int(cfg_get("nochop.ema", 20))
    nochop_lookback = int(cfg_get("nochop.lookback", 40))
    nochop_min_closes = int(cfg_get("nochop.min_closes", 24))

    corr_logic = str(cfg_get("corr.logic", "or")).strip().lower()
    corr_window_xag = int(cfg_get("corr.xag.window", 40))
    corr_min_abs_xag = float(cfg_get("corr.xag.min_abs", 0.10))
    corr_flip_lookback_xag = int(cfg_get("corr.xag.flip_lookback", 50))
    corr_max_flips_xag = int(cfg_get("corr.xag.max_flips", 0))

    corr_window_eur = int(cfg_get("corr.eur.window", 75))
    corr_min_abs_eur = float(cfg_get("corr.eur.min_abs", 0.10))
    corr_flip_lookback_eur = int(cfg_get("corr.eur.flip_lookback", 75))
    corr_max_flips_eur = int(cfg_get("corr.eur.max_flips", 5))

    confirm_size_one = float(cfg_get("sizing.confirm_size_one", 1.0))
    confirm_size_both = float(cfg_get("sizing.confirm_size_both", 2.0))

    # Build module config dicts for the strategy (None disables a module).
    ema_sep_mod = None if args.no_ema_sep else (ema_sep_cfg if isinstance(ema_sep_cfg, dict) else {})

    nochop_mod = None
    if not args.no_nochop:
        base_nc = nochop_cfg if isinstance(nochop_cfg, dict) else {}
        # Apply CLI overrides
        base_nc = dict(base_nc)
        base_nc["entry_held"] = bool(args.nochop_entry_held)
        base_nc["exit_bad_bars"] = int(args.nochop_exit_bad_bars or 0)
        nochop_mod = base_nc

    corr_mod_template = None
    if not args.no_corr:
        base_corr = corr_cfg if isinstance(corr_cfg, dict) else {}
        corr_mod_template = dict(base_corr)

    sizing_mod = sizing_cfg if isinstance(sizing_cfg, dict) else {"confirm_size_one": confirm_size_one, "confirm_size_both": confirm_size_both}

    risk_mod = {
        "shock_exit_abs_ret": float(args.shock_exit_abs_ret),
        "shock_exit_sigma_k": float(args.shock_exit_sigma_k),
        "shock_exit_sigma_window": int(args.shock_exit_sigma_window),
        "shock_cooldown_bars": int(args.shock_cooldown_bars),
        "segment_ttl_bars": int(args.segment_ttl_bars),
    }

    tf_mode = "force_flat"
    tf_entry_shift = 1
    if isinstance(time_filter_cfg, dict):
        tf_mode = str(time_filter_cfg.get("mode") or tf_mode)
        tf_entry_shift = int(time_filter_cfg.get("entry_shift") or tf_entry_shift)

    def run_one(px: pd.Series, htf: pd.DataFrame, *, cc_xag: pd.Series, cc_eur: pd.Series, allow_mask: pd.Series | None):
        corr_mod = None
        if corr_mod_template is not None:
            c = dict(corr_mod_template)
            xag = dict(c.get("xag", {}) or {})
            xag["close"] = cc_xag
            c["xag"] = xag
            if args.corr2_symbol and cc_eur is not None:
                eur = dict(c.get("eur", {}) or {})
                eur["close"] = cc_eur
                c["eur"] = eur
            corr_mod = c

        tf_mod = None
        if allow_mask is not None:
            tf_mod = {"allow_mask": allow_mask, "mode": tf_mode, "entry_shift": tf_entry_shift}

        htf_mod = None
        if isinstance(cfg_get("htf_confirm", None), dict):
            htf_mod = {"bars_15m": htf, "rule": args.htf_rule}

        return trend_following_ma_crossover(
            px,
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

    bt_2122, _, exec_2122 = run_one(px_2122, htf_2122, cc_xag=corr_xag_2122, cc_eur=corr_eur_2122, allow_mask=fomc_2122)
    bt_2325, _, exec_2325 = run_one(px_2325, htf_2325, cc_xag=corr_xag_2325, cc_eur=corr_eur_2325, allow_mask=fomc_2325)
    bt_2026, _, exec_2026 = run_one(px_2026, htf_2026, cc_xag=corr_xag_2026, cc_eur=corr_eur_2026, allow_mask=fomc_2026)

    periods = {"2021-2022": bt_2122, "2023-2025": bt_2325, "2026": bt_2026}

    initial_capital = {k: 1000.0 for k in periods}

    def round_trips(executions: int) -> int:
        return int(executions // 2)

    n_trades = {
        "2021-2022": round_trips(exec_2122),
        "2023-2025": round_trips(exec_2325),
        "2026": round_trips(exec_2026),
    }

    title = (
        f"{args.symbol} trend (5m MA {args.fast}/{args.slow}) [base=5m OHLC close]"
        f" + HTF({args.htf_rule}) [bars=15m OHLC]"
    )
    if not args.no_ema_sep:
        title += f" + EMAsep(HTF EMA{ema_fast}/{ema_slow}, TR-ATR{atr_n}, k={sep_k:g})"
    if not args.no_nochop:
        title += f" + NoChop(HTF EMA{nochop_ema}, lookback={nochop_lookback}, min={nochop_min_closes})"

    if bool(args.nochop_entry_held):
        title += " + NoChop(entry-held)"
    if int(args.nochop_exit_bad_bars or 0) > 0:
        title += f" + NoChopExit(bad_bars={int(args.nochop_exit_bad_bars)})"
    if not args.no_corr:
        logic_txt = "OR" if corr_logic == "or" else "AND"
        title += (
            f" + CorrStability({args.corr_symbol}, win={corr_window_xag}, abs>={corr_min_abs_xag:g}, flips<={corr_max_flips_xag}/{corr_flip_lookback_xag})"
            f" {logic_txt} {args.corr2_symbol}(win={corr_window_eur}, abs>={corr_min_abs_eur:g}, flips<={corr_max_flips_eur}/{corr_flip_lookback_eur})"
        )
        title += f" + Size(confirm one={confirm_size_one:g}, both={confirm_size_both:g})"
    else:
        title += " + Size(fixed=1.0; corr_filter=off)"

    if not args.no_fomc:
        title += f" + FOMC(force-flat @ {args.fomc_utc_hhmm}Z pre={args.fomc_pre_hours:g}h post={args.fomc_post_hours:g}h)"

    if float(args.shock_exit_abs_ret or 0.0) > 0.0:
        title += f" + ShockExit(abs_ret>={float(args.shock_exit_abs_ret):g})"
    if float(args.shock_exit_sigma_k or 0.0) > 0.0:
        title += f" + ShockExit({float(args.shock_exit_sigma_k):g}σ@{int(args.shock_exit_sigma_window)}b)"
    if int(args.shock_cooldown_bars or 0) > 0:
        title += f" + Cooldown({int(args.shock_cooldown_bars)}b)"

    if int(args.segment_ttl_bars or 0) > 0:
        title += f" + TTL({int(args.segment_ttl_bars)}b)"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name

    report_periods_equity_only(
        periods=periods,
        out_path=out_path,
        title=title,
        freq="5MIN",
        initial_capital=initial_capital,
        n_trades=n_trades,
    )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
