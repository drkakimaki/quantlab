from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
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


def build_allow_mask(index: pd.DatetimeIndex, days: list[dt.date], *, utc_hhmm: str, pre_h: float, post_h: float) -> pd.Series:
    hh, mm = [int(x) for x in utc_hhmm.split(":")]
    events: list[EventWindow] = []
    for d in days:
        ts = pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC))
        events.append(EventWindow(ts=ts, pre=dt.timedelta(hours=float(pre_h)), post=dt.timedelta(hours=float(post_h))))
    return build_allow_mask_from_events(index, events=events)


def main() -> None:
    ap = argparse.ArgumentParser(description="FOMC window hyperparam search (best_trend; production semantics via time_allow_mask).")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--xag", type=str, default="XAGUSD")
    ap.add_argument("--eur", type=str, default="EURUSD")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based/fomc_window_search"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--fomc-days", type=Path, default=Path("data/econ_calendar/fomc_decision_days.csv"))
    ap.add_argument("--p3-end", type=dt.date.fromisoformat, default=dt.date(2026, 2, 13))
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
        ema_sep={"ema_fast": 40, "ema_slow": 300, "atr_n": 20, "sep_k": 0.05},
        nochop={"ema": 20, "lookback": 40, "min_closes": 24, "entry_held": False, "exit_bad_bars": 0},
        corr={
            "logic": "or",
            "xag": {"window": 40, "min_abs": 0.10, "flip_lookback": 50, "max_flips": 0},
            "eur": {"window": 75, "min_abs": 0.10, "flip_lookback": 75, "max_flips": 5},
        },
        sizing={"confirm_size_one": 1.0, "confirm_size_both": 2.0},
        risk={"shock_exit_abs_ret": 0.0, "shock_exit_sigma_k": 0.0, "shock_exit_sigma_window": 96, "shock_cooldown_bars": 0, "segment_ttl_bars": 0},
    )

    hours = [0.5, 1.0, 1.5, 2.0, 3.0]
    modes = {
        "no-entry": "block_entry_hold_segment",
        "force-flat": "force_flat",
    }

    space = {
        "pre_h": hours,
        "post_h": hours,
        "mode": list(modes.keys()),
    }

    # Preload bars per period (prices/htf/corr); allow masks differ per cfg so computed in loop
    data: dict[str, dict] = {}
    for pname, (start, end) in periods.items():
        bars5 = load_ohlc_daily(symbol=args.symbol, start=start, end=end, root=args.root_5m_ohlc)
        bars15 = load_ohlc_daily(symbol=args.symbol, start=start, end=end, root=args.root_15m_ohlc)
        px = bars5["close"].astype(float)
        xag = load_ohlc_daily(symbol=args.xag, start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)
        eur = load_ohlc_daily(symbol=args.eur, start=start, end=end, root=args.root_5m_ohlc)["close"].astype(float)
        days = [d for d in fomc_days_all if start <= d <= end]
        data[pname] = {"px": px, "htf": bars15, "xag": xag, "eur": eur, "days": days}

    grid = [dict(zip(space.keys(), vals)) for vals in itertools.product(*[space[k] for k in space])]

    grid_meta = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "n_configs": int(len(grid)),
        "space": space,
        "periods": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in periods.items()},
        "fixed_params": fixed,
        "fomc_days": str(args.fomc_days),
        "event_time_utc": "19:00",
        "modes": modes,
        "time_entry_shift": 1,
        "note": "Production semantics: pass time_allow_mask into strategy; no post-trade filtering.",
    }
    (out_dir / "fomc_window_grid.json").write_text(json.dumps(grid_meta, indent=2, sort_keys=True))

    long_rows: list[dict] = []
    agg_rows: list[dict] = []

    for i, cfg in enumerate(grid, start=1):
        cfg_id = _config_id(cfg)
        period_metrics: dict[str, dict] = {}

        for pname, d in data.items():
            allow = build_allow_mask(d["px"].index, d["days"], utc_hhmm="19:00", pre_h=float(cfg["pre_h"]), post_h=float(cfg["post_h"]))

            corr_mod = dict(fixed["corr"])
            corr_mod["xag"] = dict(corr_mod.get("xag", {}))
            corr_mod["xag"]["close"] = d["xag"]
            corr_mod["eur"] = dict(corr_mod.get("eur", {}))
            corr_mod["eur"]["close"] = d["eur"]

            bt, _, _ = trend_following_ma_crossover(
                d["px"],
                fast=fixed["fast"],
                slow=fixed["slow"],
                fee_bps=fixed["fee_bps"],
                slippage_bps=fixed["slippage_bps"],
                htf_confirm={"bars_15m": d["htf"], "rule": fixed["htf_rule"]},
                ema_sep=fixed["ema_sep"],
                nochop=fixed["nochop"],
                corr=corr_mod,
                sizing=fixed["sizing"],
                risk=fixed["risk"],
                time_filter={"allow_mask": allow, "mode": modes[str(cfg["mode"])], "entry_shift": 1},
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
                "trades": _n_trades(bt),
                **cfg,
                "time_filter_mode": modes[str(cfg["mode"])],
                "n_events": len(d["days"]),
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
                "time_filter_mode": modes[str(cfg["mode"])],
            }
        )

        if i % 10 == 0:
            print(f"[{i}/{len(grid)}] done")

    df_long = pd.DataFrame(long_rows)
    df_agg = pd.DataFrame(agg_rows)

    df_long.to_csv(out_dir / "fomc_window_results_long.csv", index=False)
    df_agg.to_csv(out_dir / "fomc_window_results_agg.csv", index=False)

    df_rank = df_agg.sort_values(["mean_sharpe", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    top = df_rank.head(int(args.top_k)).copy()

    report_paths = []
    for _, r in top.iterrows():
        cfg = {k: r[k] for k in ["pre_h", "post_h", "mode"]}
        cfg_id = str(r["config_id"])

        bts = {}
        ntr = {}
        for pname, d in data.items():
            allow = build_allow_mask(d["px"].index, d["days"], utc_hhmm="19:00", pre_h=float(cfg["pre_h"]), post_h=float(cfg["post_h"]))
            corr_mod = dict(fixed["corr"])
            corr_mod["xag"] = dict(corr_mod.get("xag", {}))
            corr_mod["xag"]["close"] = d["xag"]
            corr_mod["eur"] = dict(corr_mod.get("eur", {}))
            corr_mod["eur"]["close"] = d["eur"]

            bt, _, _ = trend_following_ma_crossover(
                d["px"],
                fast=fixed["fast"],
                slow=fixed["slow"],
                fee_bps=fixed["fee_bps"],
                slippage_bps=fixed["slippage_bps"],
                htf_confirm={"bars_15m": d["htf"], "rule": fixed["htf_rule"]},
                ema_sep=fixed["ema_sep"],
                nochop=fixed["nochop"],
                corr=corr_mod,
                sizing=fixed["sizing"],
                risk=fixed["risk"],
                time_filter={"allow_mask": allow, "mode": modes[str(cfg["mode"])], "entry_shift": 1},
            )
            bts[pname] = bt
            ntr[pname] = _n_trades(bt)

        title = (
            f"{args.symbol} best_trend FOMC window search | cfg={cfg_id} | "
            f"19:00Z pre={cfg['pre_h']}h post={cfg['post_h']}h mode={cfg['mode']} (entry_shift=1)"
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
        "pre_h",
        "post_h",
        "mode",
        "time_filter_mode",
    ]

    by_mean = df_rank
    by_worst_dd = df_agg.sort_values(["worst_max_drawdown", "mean_sharpe"], ascending=[False, False]).reset_index(drop=True)
    by_2325 = df_agg.sort_values(["sharpe_2023_2025", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    by_2026 = df_agg.sort_values(["sharpe_2026", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)

    md: list[str] = []
    md.append("# FOMC window hyperparam search (best_trend)\n")
    md.append(f"Generated: {grid_meta['created_utc']}\n")
    md.append(f"Configs evaluated: {len(grid)} (full grid)\n")
    md.append("Semantics: time_allow_mask passed into strategy (pre-engine), time_entry_shift=1; no post-filtering.\n")
    md.append(f"FOMC days source: `{args.fomc_days}`\n")

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

    (out_dir / "fomc_window_summary.md").write_text("\n".join(md) + "\n")

    print(f"Wrote: {out_dir / 'fomc_window_results_long.csv'}")
    print(f"Wrote: {out_dir / 'fomc_window_results_agg.csv'}")
    print(f"Wrote: {out_dir / 'fomc_window_summary.md'}")


if __name__ == "__main__":
    main()
