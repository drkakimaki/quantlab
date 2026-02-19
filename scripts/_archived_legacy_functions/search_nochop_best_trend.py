from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.metrics import performance_summary
from quantlab.report_periods import report_periods_equity_only
from quantlab.strategies import trend_following_ma_crossover
from quantlab.time_filter import EventWindow, build_allow_mask_from_events


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


def _trade_returns_from_position(bt: pd.DataFrame) -> pd.Series:
    if bt is None or bt.empty:
        return pd.Series(dtype=float)
    if "position" not in bt.columns or "returns_net" not in bt.columns:
        return pd.Series(dtype=float)

    pos = bt["position"].fillna(0.0).astype(float)
    r = bt["returns_net"].fillna(0.0).astype(float)

    in_pos = pos != 0.0
    if not bool(in_pos.any()):
        return pd.Series(dtype=float)

    entry = in_pos & (~in_pos.shift(1, fill_value=False))
    trade_id = entry.cumsum()

    df = pd.DataFrame({"trade_id": trade_id, "in_pos": in_pos, "r": r})
    df = df[df["in_pos"]].copy()
    if df.empty:
        return pd.Series(dtype=float)

    log_r = (1.0 + df["r"]).clip(lower=1e-12)
    df["log1p_r"] = np.log(log_r)
    trade_log = df.groupby("trade_id")["log1p_r"].sum()
    return np.expm1(trade_log).astype(float)


def _win_rate(bt: pd.DataFrame) -> float:
    tr = _trade_returns_from_position(bt)
    if tr.empty:
        return float("nan")
    return float(100.0 * (tr > 0.0).mean())


def _profit_factor(bt: pd.DataFrame) -> float:
    tr = _trade_returns_from_position(bt)
    if tr.empty:
        return float("nan")
    gp = float(tr[tr > 0.0].sum())
    gl = float((-tr[tr < 0.0]).sum())
    if gl <= 0.0:
        return float("nan")
    return float(gp / gl)


def _n_trades(bt: pd.DataFrame) -> int:
    if bt is None or bt.empty or "position" not in bt.columns:
        return 0
    execs = int((bt["position"].diff().abs() > 0).sum())
    return int(execs // 2)


def _config_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def load_fomc_days(path: Path) -> list[dt.date]:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("fomc CSV must contain 'date' column")
    days = [dt.date.fromisoformat(x) for x in df["date"].astype(str).tolist()]
    return sorted(set(days))


def build_fomc_allow_mask(index: pd.DatetimeIndex, days: list[dt.date], *, pre_h: float, post_h: float, utc_hhmm: str = "19:00") -> pd.Series:
    hh, mm = [int(x) for x in utc_hhmm.split(":")]
    events: list[EventWindow] = []
    for d in days:
        ts = pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC))
        events.append(EventWindow(ts=ts, pre=dt.timedelta(hours=float(pre_h)), post=dt.timedelta(hours=float(post_h))))
    return build_allow_mask_from_events(index, events=events)


def min_closes_options(lookback: int) -> list[int]:
    fracs = [0.5, 0.6, 0.7]
    abs_vals = [6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
    vals = set()
    for f in fracs:
        vals.add(int(round(lookback * f)))
    for a in abs_vals:
        vals.add(int(a))
    out = sorted(x for x in vals if 1 <= x <= lookback)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="NoChop hyperparam search (best_trend, corr fixed, EMA-sep fixed, FOMC fixed).")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--xag", type=str, default="XAGUSD")
    ap.add_argument("--eur", type=str, default="EURUSD")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based/nochop_search"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--fomc-days", type=Path, default=Path("data/econ_calendar/fomc_decision_days.csv"))
    ap.add_argument("--p3-end", type=dt.date.fromisoformat, default=dt.date(2026, 2, 13))
    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=250)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--sep-k", type=float, default=0.15)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    periods = {
        "2021-2022": (dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), args.p3_end),
    }

    fomc_days_all = load_fomc_days(args.fomc_days)

    fixed = dict(
        fast=30,
        slow=75,
        htf_rule="15min",
        fee_bps=0.0,
        slippage_bps=0.0,
        ema_sep={"ema_fast": int(args.ema_fast), "ema_slow": int(args.ema_slow), "atr_n": int(args.atr_n), "sep_k": float(args.sep_k)},
        # NoChop values are what we search over; provide placeholders here.
        nochop={"ema": 20, "lookback": 40, "min_closes": 24, "entry_held": False, "exit_bad_bars": 0},
        corr={
            "logic": "or",
            "xag": {"window": 40, "min_abs": 0.10, "flip_lookback": 50, "max_flips": 0},
            "eur": {"window": 75, "min_abs": 0.10, "flip_lookback": 75, "max_flips": 5},
        },
        sizing={"confirm_size_one": 1.0, "confirm_size_both": 2.0},
        risk={"shock_exit_abs_ret": 0.0, "shock_exit_sigma_k": 0.0, "shock_exit_sigma_window": 96, "shock_cooldown_bars": 0, "segment_ttl_bars": 0},
        # FOMC fixed
        time_filter={"mode": "block_entry_hold_segment", "entry_shift": 1},
    )

    space = {
        "nochop_ema": [10, 15, 20],
        "nochop_lookback": [15, 20, 30, 40],
    }

    # Preload data + allow mask
    data: dict[str, dict] = {}
    for pname, (start, end) in periods.items():
        bars5 = load_ohlc_daily(symbol=args.symbol, start=start, end=end, root=args.root_5m_ohlc)
        bars15 = load_ohlc_daily(symbol=args.symbol, start=start, end=end, root=args.root_15m_ohlc)
        px = bars5["close"].astype(float)
        xag = load_ohlc_daily(symbol=args.xag, start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)
        eur = load_ohlc_daily(symbol=args.eur, start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)

        days = [d for d in fomc_days_all if start <= d <= end]
        allow = build_fomc_allow_mask(px.index, days, pre_h=2.0, post_h=2.0, utc_hhmm="19:00")

        data[pname] = {"px": px, "htf": bars15, "xag": xag, "eur": eur, "allow": allow}

    # Build grid (lookback-dependent min_closes)
    grid: list[dict] = []
    for ne in space["nochop_ema"]:
        for lb in space["nochop_lookback"]:
            for mc in min_closes_options(int(lb)):
                grid.append({"nochop_ema": int(ne), "nochop_lookback": int(lb), "nochop_min_closes": int(mc)})

    grid_meta = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "n_configs": int(len(grid)),
        "space": {**space, "nochop_min_closes": "derived per lookback (fractions 0.5/0.6/0.7 + abs vals, clamped to [1,lookback])"},
        "periods": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in periods.items()},
        "fixed_params": fixed,
        "fomc_days": str(args.fomc_days),
        "fomc_window": {"utc_hhmm": "19:00", "pre_h": 2.0, "post_h": 2.0, "mode": "block_entry_hold_segment", "time_entry_shift": 1},
    }
    (out_dir / "nochop_grid.json").write_text(json.dumps(grid_meta, indent=2, sort_keys=True))

    long_rows: list[dict] = []
    agg_rows: list[dict] = []

    for i, cfg in enumerate(grid, start=1):
        cfg_id = _config_id(cfg)
        period_metrics: dict[str, dict] = {}

        for pname, d in data.items():
            corr_mod = dict(fixed["corr"])
            corr_mod["xag"] = dict(corr_mod.get("xag", {}))
            corr_mod["xag"]["close"] = d["xag"]
            corr_mod["eur"] = dict(corr_mod.get("eur", {}))
            corr_mod["eur"]["close"] = d["eur"]

            nochop_mod = dict(fixed["nochop"])
            nochop_mod["ema"] = int(cfg["nochop_ema"])
            nochop_mod["lookback"] = int(cfg["nochop_lookback"])
            nochop_mod["min_closes"] = int(cfg["nochop_min_closes"])

            tf = dict(fixed["time_filter"])
            tf["allow_mask"] = d["allow"]

            bt, _, _ = trend_following_ma_crossover(
                d["px"],
                fast=fixed["fast"],
                slow=fixed["slow"],
                fee_bps=fixed["fee_bps"],
                slippage_bps=fixed["slippage_bps"],
                htf_confirm={"bars_15m": d["htf"], "rule": fixed["htf_rule"]},
                ema_sep=fixed["ema_sep"],
                nochop=nochop_mod,
                corr=corr_mod,
                sizing=fixed["sizing"],
                risk=fixed["risk"],
                time_filter=tf,
            )

            summ = performance_summary(bt["returns_net"], bt["equity"], freq="5MIN") if (bt is not None and not bt.empty) else None
            pnl_usd = float(bt["equity"].iloc[-1] - 1000.0) if (bt is not None and not bt.empty) else float("nan")
            maxdd = float(bt["equity"].div(bt["equity"].cummax()).sub(1.0).min()) if (bt is not None and not bt.empty) else float("nan")
            sh = float(summ.sharpe) if summ else float("nan")

            row = {
                "config_id": cfg_id,
                "config_idx": i,
                "period": pname,
                "pnl_usd": pnl_usd,
                "max_drawdown": maxdd,
                "sharpe": sh,
                "win_rate": _win_rate(bt),
                "profit_factor": _profit_factor(bt),
                "trades": _n_trades(bt),
                **cfg,
            }
            long_rows.append(row)
            period_metrics[pname] = row

        sharpe_vals = [period_metrics[p]["sharpe"] for p in periods]
        dd_vals = [period_metrics[p]["max_drawdown"] for p in periods]
        pnl_vals = [period_metrics[p]["pnl_usd"] for p in periods]

        agg_rows.append(
            {
                "config_id": cfg_id,
                "mean_sharpe": float(np.nanmean(sharpe_vals)),
                "min_sharpe": float(np.nanmin(sharpe_vals)),
                "worst_max_drawdown": float(np.nanmin(dd_vals)),
                "sum_pnl_usd": float(np.nansum(pnl_vals)),
                "sharpe_2021_2022": float(period_metrics["2021-2022"]["sharpe"]),
                "sharpe_2023_2025": float(period_metrics["2023-2025"]["sharpe"]),
                "sharpe_2026": float(period_metrics["2026"]["sharpe"]),
                "maxdd_2026": float(period_metrics["2026"]["max_drawdown"]),
                "trades_total": int(sum(period_metrics[p]["trades"] for p in periods)),
                **cfg,
            }
        )

        if i % 25 == 0:
            print(f"[{i}/{len(grid)}] done")

    df_long = pd.DataFrame(long_rows)
    df_agg = pd.DataFrame(agg_rows)

    df_long.to_csv(out_dir / "nochop_results_long.csv", index=False)
    df_agg.to_csv(out_dir / "nochop_results_agg.csv", index=False)

    df_rank = df_agg.sort_values(["mean_sharpe", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    top = df_rank.head(int(args.top_k)).copy()

    report_paths = []
    for _, r in top.iterrows():
        cfg = {k: r[k] for k in ["nochop_ema", "nochop_lookback", "nochop_min_closes"]}
        cfg_id = str(r["config_id"])

        bts = {}
        ntr = {}
        for pname, d in data.items():
            corr_mod = dict(fixed["corr"])
            corr_mod["xag"] = dict(corr_mod.get("xag", {}))
            corr_mod["xag"]["close"] = d["xag"]
            corr_mod["eur"] = dict(corr_mod.get("eur", {}))
            corr_mod["eur"]["close"] = d["eur"]

            nochop_mod = dict(fixed["nochop"])
            nochop_mod["ema"] = int(cfg["nochop_ema"])
            nochop_mod["lookback"] = int(cfg["nochop_lookback"])
            nochop_mod["min_closes"] = int(cfg["nochop_min_closes"])

            tf = dict(fixed["time_filter"])
            tf["allow_mask"] = d["allow"]

            bt, _, _ = trend_following_ma_crossover(
                d["px"],
                fast=fixed["fast"],
                slow=fixed["slow"],
                fee_bps=fixed["fee_bps"],
                slippage_bps=fixed["slippage_bps"],
                htf_confirm={"bars_15m": d["htf"], "rule": fixed["htf_rule"]},
                ema_sep=fixed["ema_sep"],
                nochop=nochop_mod,
                corr=corr_mod,
                sizing=fixed["sizing"],
                risk=fixed["risk"],
                time_filter=tf,
            )
            bts[pname] = bt
            ntr[pname] = _n_trades(bt)

        title = (
            f"{args.symbol} best_trend NoChop search | cfg={cfg_id} | "
            f"nochop_ema={cfg['nochop_ema']} lookback={cfg['nochop_lookback']} min_closes={cfg['nochop_min_closes']} | "
            f"EMAsep({fixed['ema_fast']},{fixed['ema_slow']},atr{fixed['atr_n']},k{fixed['sep_k']}) | Corr fixed | FOMC 19:00Z±2h no-entry"
        )

        out_path = out_dir / f"top_{cfg_id}.html"
        report_periods_equity_only(
            periods=bts,
            out_path=out_path,
            title=title,
            freq="5MIN",
            initial_capital={k: 1000.0 for k in bts},
            n_trades=ntr,
        )
        report_paths.append(str(out_path))

    def _fmt_table(df: pd.DataFrame, cols: list[str], n: int = 15) -> str:
        return df[cols].head(n).to_markdown(index=False)

    cols_show = [
        "config_id",
        "mean_sharpe",
        "worst_max_drawdown",
        "sum_pnl_usd",
        "sharpe_2023_2025",
        "sharpe_2026",
        "nochop_ema",
        "nochop_lookback",
        "nochop_min_closes",
    ]

    by_mean = df_rank
    by_worst_dd = df_agg.sort_values(["worst_max_drawdown", "mean_sharpe"], ascending=[False, False]).reset_index(drop=True)
    by_2325 = df_agg.sort_values(["sharpe_2023_2025", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    by_2026 = df_agg.sort_values(["sharpe_2026", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)

    md: list[str] = []
    md.append("# NoChop filter hyperparam search (best_trend)\n")
    md.append(f"Generated: {grid_meta['created_utc']}\n")
    md.append(f"Configs evaluated: {len(grid)}\n")
    md.append("Fixed modules:\n")
    md.append(
        f"- EMA-sep: ema_fast={fixed['ema_fast']}, ema_slow={fixed['ema_slow']}, atr_n={fixed['atr_n']}, sep_k={fixed['sep_k']}\n"
    )
    md.append("- Corr: XAG(win40 abs0.10 flb50 mf0), EUR(win75 abs0.10 flb75 mf5), logic=or\n")
    md.append("- FOMC: 19:00Z ±2h, mode=block_entry_hold_segment, time_entry_shift=1 (production semantics)\n")

    md.append("\n## Top by mean Sharpe (across all 3 periods)\n")
    md.append(_fmt_table(by_mean, cols_show, n=15))
    md.append("\n## Top by worst MaxDD (least-negative drawdown across periods)\n")
    md.append(_fmt_table(by_worst_dd, cols_show, n=15))
    md.append("\n## Top by Sharpe (2023-2025)\n")
    md.append(_fmt_table(by_2325, cols_show, n=15))
    md.append("\n## Top by Sharpe (2026)\n")
    md.append(_fmt_table(by_2026, cols_show, n=15))

    md.append("\n## HTML reports generated (top-k by mean Sharpe)\n")
    for p in report_paths:
        md.append(f"- {p}")

    (out_dir / "nochop_summary.md").write_text("\n".join(md) + "\n")

    print(f"Wrote: {out_dir / 'nochop_results_long.csv'}")
    print(f"Wrote: {out_dir / 'nochop_results_agg.csv'}")
    print(f"Wrote: {out_dir / 'nochop_summary.md'}")


if __name__ == "__main__":
    main()
