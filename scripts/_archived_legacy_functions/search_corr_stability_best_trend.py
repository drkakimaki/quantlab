from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.metrics import performance_summary
from quantlab.report_periods import report_periods_equity_only
from quantlab.strategies import trend_following_ma_crossover


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
    """Compounded return per trade, where trade = contiguous segment position != 0."""
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


def sample_configs(*, space: dict[str, list], n: int, seed: int) -> list[dict]:
    """Reproducible sampling without replacement from the full Cartesian product.

    We sample indices into the product to avoid materializing everything.
    """

    keys = list(space)
    sizes = [len(space[k]) for k in keys]
    total = int(np.prod(sizes))
    if n > total:
        n = total

    rng = np.random.default_rng(seed)
    # sample integer indices in [0,total)
    idx = rng.choice(total, size=n, replace=False)

    # convert mixed radix
    cfgs: list[dict] = []
    for x in idx:
        rem = int(x)
        cfg: dict = {}
        for k, base in zip(reversed(keys), reversed(sizes), strict=True):
            j = rem % base
            rem //= base
            cfg[k] = space[k][j]
        cfgs.append({k: cfg[k] for k in keys})
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Hyperparam search for best_trend corr-stability module.")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--xag", type=str, default="XAGUSD")
    ap.add_argument("--eur", type=str, default="EURUSD")
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--root-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))

    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--p1-start", type=dt.date.fromisoformat, default=dt.date(2021, 1, 1))
    ap.add_argument("--p1-end", type=dt.date.fromisoformat, default=dt.date(2022, 12, 31))
    ap.add_argument("--p2-start", type=dt.date.fromisoformat, default=dt.date(2023, 1, 1))
    ap.add_argument("--p2-end", type=dt.date.fromisoformat, default=dt.date(2025, 12, 31))
    ap.add_argument("--p3-start", type=dt.date.fromisoformat, default=dt.date(2026, 1, 1))
    ap.add_argument("--p3-end", type=dt.date.fromisoformat, default=dt.date(2026, 2, 13))

    ap.add_argument("--out-dir", type=Path, default=Path("reports/trend_based/corr_search"))
    ap.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data once per period
    def load_period(symbol: str, start: dt.date, end: dt.date) -> tuple[pd.Series, pd.DataFrame]:
        bars5 = load_ohlc_daily(symbol=symbol, start=start, end=end, root=args.root_5m_ohlc)
        bars15 = load_ohlc_daily(symbol=symbol, start=start, end=end, root=args.root_15m_ohlc)
        return bars5["close"].astype(float).copy(), bars15

    # explicit
    print("Loading main symbol OHLC...")
    px_2122, htf_2122 = load_period(args.symbol, args.p1_start, args.p1_end)
    print("  loaded 2021-2022")
    px_2325, htf_2325 = load_period(args.symbol, args.p2_start, args.p2_end)
    print("  loaded 2023-2025")
    px_2026, htf_2026 = load_period(args.symbol, args.p3_start, args.p3_end)
    print("  loaded 2026")

    print("Loading corr symbols (5m close)...")
    xag_2122 = load_ohlc_daily(symbol=args.xag, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)["close"].astype(float)
    xag_2325 = load_ohlc_daily(symbol=args.xag, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)["close"].astype(float)
    xag_2026 = load_ohlc_daily(symbol=args.xag, start=args.p3_start, end=args.p3_end, root=args.root_5m_ohlc)["close"].astype(float)
    print("  loaded XAG")

    eur_2122 = load_ohlc_daily(symbol=args.eur, start=args.p1_start, end=args.p1_end, root=args.root_5m_ohlc)["close"].astype(float)
    eur_2325 = load_ohlc_daily(symbol=args.eur, start=args.p2_start, end=args.p2_end, root=args.root_5m_ohlc)["close"].astype(float)
    eur_2026 = load_ohlc_daily(symbol=args.eur, start=args.p3_start, end=args.p3_end, root=args.root_5m_ohlc)["close"].astype(float)
    print("  loaded EUR")

    data = {
        "2021-2022": dict(px=px_2122, htf=htf_2122, xag=xag_2122, eur=eur_2122),
        "2023-2025": dict(px=px_2325, htf=htf_2325, xag=xag_2325, eur=eur_2325),
        "2026": dict(px=px_2026, htf=htf_2026, xag=xag_2026, eur=eur_2026),
    }

    # Fixed best_trend params (do NOT touch)
    fixed = dict(
        fast=30,
        slow=75,
        htf_rule="15min",
        ema_sep_filter=True,
        ema_fast=50,
        ema_slow=250,
        atr_n=14,
        sep_k=0.15,
        nochop_filter=True,
        nochop_ema=15,
        nochop_lookback=20,
        nochop_min_closes=12,
        corr_filter=True,
        confirm_size_one=1.0,
        confirm_size_both=2.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    # Search space
    space = {
        "corr_logic": ["or", "and"],
        "corr_window_xag": [30, 40, 50, 60, 75, 100],
        "corr_min_abs_xag": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "corr_flip_lookback_xag": [25, 50, 75, 100, 150],
        "corr_max_flips_xag": [0, 1, 2, 3, 4, 5, 6],
        "corr_window_eur": [30, 40, 50, 60, 75, 100],
        "corr_min_abs_eur": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "corr_flip_lookback_eur": [25, 50, 75, 100, 150],
        "corr_max_flips_eur": [0, 1, 2, 3, 4, 5, 6],
    }

    # Add baseline config explicitly (current defaults)
    baseline_cfg = dict(
        corr_logic="or",
        corr_window_xag=50,
        corr_min_abs_xag=0.25,
        corr_flip_lookback_xag=100,
        corr_max_flips_xag=1,
        corr_window_eur=75,
        corr_min_abs_eur=0.20,
        corr_flip_lookback_eur=50,
        corr_max_flips_eur=4,
    )

    sampled = sample_configs(space=space, n=args.n_samples, seed=args.seed)
    # ensure baseline present
    sampled_map = {json.dumps(c, sort_keys=True): c for c in sampled}
    sampled_map[json.dumps(baseline_cfg, sort_keys=True)] = baseline_cfg
    configs = list(sampled_map.values())

    grid_meta = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat().replace('+00:00','Z'),
        "seed": int(args.seed),
        "n_samples_requested": int(args.n_samples),
        "n_configs": int(len(configs)),
        "space": space,
        "baseline_cfg": baseline_cfg,
        "periods": {
            "2021-2022": [args.p1_start.isoformat(), args.p1_end.isoformat()],
            "2023-2025": [args.p2_start.isoformat(), args.p2_end.isoformat()],
            "2026": [args.p3_start.isoformat(), args.p3_end.isoformat()],
        },
        "fixed_params": fixed,
    }
    (out_dir / "corr_search_grid.json").write_text(json.dumps(grid_meta, indent=2, sort_keys=True))

    long_rows: list[dict] = []
    agg_rows: list[dict] = []

    # Run configs
    for i, cfg in enumerate(configs, start=1):
        cfg_id = _config_id(cfg)

        period_metrics = {}
        bts_for_report: dict[str, pd.DataFrame] = {}
        for pname, d in data.items():
            corr_mod = {
                "logic": str(cfg["corr_logic"]),
                "xag": {
                    "close": d["xag"],
                    "window": int(cfg["corr_window_xag"]),
                    "min_abs": float(cfg["corr_min_abs_xag"]),
                    "flip_lookback": int(cfg["corr_flip_lookback_xag"]),
                    "max_flips": int(cfg["corr_max_flips_xag"]),
                },
                "eur": {
                    "close": d["eur"],
                    "window": int(cfg["corr_window_eur"]),
                    "min_abs": float(cfg["corr_min_abs_eur"]),
                    "flip_lookback": int(cfg["corr_flip_lookback_eur"]),
                    "max_flips": int(cfg["corr_max_flips_eur"]),
                },
            }

            bt, _, _ = trend_following_ma_crossover(
                d["px"],
                fast=int(fixed["fast"]),
                slow=int(fixed["slow"]),
                fee_bps=float(fixed["fee_bps"]),
                slippage_bps=float(fixed["slippage_bps"]),
                htf_confirm={"bars_15m": d["htf"], "rule": str(fixed["htf_rule"])},
                ema_sep={"ema_fast": int(fixed["ema_fast"]), "ema_slow": int(fixed["ema_slow"]), "atr_n": int(fixed["atr_n"]), "sep_k": float(fixed["sep_k"])},
                nochop={
                    "ema": int(fixed["nochop_ema"]),
                    "lookback": int(fixed["nochop_lookback"]),
                    "min_closes": int(fixed["nochop_min_closes"]),
                    "entry_held": False,
                    "exit_bad_bars": 0,
                },
                corr=corr_mod,
                sizing={"confirm_size_one": float(fixed["confirm_size_one"]), "confirm_size_both": float(fixed["confirm_size_both"])},
            )

            bts_for_report[pname] = bt

            summ = performance_summary(bt["returns_net"], bt["equity"], freq="5MIN") if (bt is not None and not bt.empty) else None

            pnl_usd = float(bt["equity"].iloc[-1] - 1000.0) if (bt is not None and not bt.empty) else float("nan")
            maxdd = float(bt["equity"].div(bt["equity"].cummax()).sub(1.0).min()) if (bt is not None and not bt.empty) else float("nan")
            sh = float(summ.sharpe) if summ else float("nan")
            wr = _win_rate(bt)
            pf = _profit_factor(bt)
            nt = _n_trades(bt)

            row = {
                "config_id": cfg_id,
                "config_idx": i,
                "period": pname,
                "pnl_usd": pnl_usd,
                "max_drawdown": maxdd,
                "sharpe": sh,
                "win_rate": wr,
                "profit_factor": pf,
                "trades": nt,
                **cfg,
            }
            long_rows.append(row)
            period_metrics[pname] = row

        # Aggregate across periods
        sharpe_vals = [period_metrics[p]["sharpe"] for p in data]
        dd_vals = [period_metrics[p]["max_drawdown"] for p in data]
        pnl_vals = [period_metrics[p]["pnl_usd"] for p in data]

        agg = {
            "config_id": cfg_id,
            "mean_sharpe": float(np.nanmean(sharpe_vals)),
            "min_sharpe": float(np.nanmin(sharpe_vals)),
            "worst_max_drawdown": float(np.nanmin(dd_vals)),
            "sum_pnl_usd": float(np.nansum(pnl_vals)),
            "sharpe_2021_2022": float(period_metrics["2021-2022"]["sharpe"]),
            "sharpe_2023_2025": float(period_metrics["2023-2025"]["sharpe"]),
            "sharpe_2026": float(period_metrics["2026"]["sharpe"]),
            "maxdd_2026": float(period_metrics["2026"]["max_drawdown"]),
            "trades_total": int(sum(period_metrics[p]["trades"] for p in data)),
            **cfg,
        }
        agg_rows.append(agg)

        if i % 25 == 0:
            print(f"[{i}/{len(configs)}] done")

    df_long = pd.DataFrame(long_rows)
    df_agg = pd.DataFrame(agg_rows)

    df_long.to_csv(out_dir / "corr_search_results_long.csv", index=False)
    df_agg.to_csv(out_dir / "corr_search_results_agg.csv", index=False)

    # Pick top-k by mean_sharpe (tie-break: worst_max_drawdown higher is better)
    df_rank = df_agg.sort_values(["mean_sharpe", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    top = df_rank.head(int(args.top_k)).copy()

    # Generate HTML reports for top configs
    report_paths = []
    for _, r in top.iterrows():
        cfg = {k: r[k] for k in space.keys()}
        cfg_id = str(r["config_id"])

        bts = {}
        for pname, d in data.items():
            corr_mod = {
                "logic": str(cfg["corr_logic"]),
                "xag": {
                    "close": d["xag"],
                    "window": int(cfg["corr_window_xag"]),
                    "min_abs": float(cfg["corr_min_abs_xag"]),
                    "flip_lookback": int(cfg["corr_flip_lookback_xag"]),
                    "max_flips": int(cfg["corr_max_flips_xag"]),
                },
                "eur": {
                    "close": d["eur"],
                    "window": int(cfg["corr_window_eur"]),
                    "min_abs": float(cfg["corr_min_abs_eur"]),
                    "flip_lookback": int(cfg["corr_flip_lookback_eur"]),
                    "max_flips": int(cfg["corr_max_flips_eur"]),
                },
            }

            bt, _, _ = trend_following_ma_crossover(
                d["px"],
                fast=int(fixed["fast"]),
                slow=int(fixed["slow"]),
                fee_bps=float(fixed["fee_bps"]),
                slippage_bps=float(fixed["slippage_bps"]),
                htf_confirm={"bars_15m": d["htf"], "rule": str(fixed["htf_rule"])},
                ema_sep={"ema_fast": int(fixed["ema_fast"]), "ema_slow": int(fixed["ema_slow"]), "atr_n": int(fixed["atr_n"]), "sep_k": float(fixed["sep_k"])},
                nochop={
                    "ema": int(fixed["nochop_ema"]),
                    "lookback": int(fixed["nochop_lookback"]),
                    "min_closes": int(fixed["nochop_min_closes"]),
                    "entry_held": False,
                    "exit_bad_bars": 0,
                },
                corr=corr_mod,
                sizing={"confirm_size_one": float(fixed["confirm_size_one"]), "confirm_size_both": float(fixed["confirm_size_both"])},
            )
            bts[pname] = bt

        title = (
            f"{args.symbol} best_trend corr search | cfg={cfg_id} | "
            f"logic={cfg['corr_logic']} | "
            f"XAG(win={cfg['corr_window_xag']},abs>={cfg['corr_min_abs_xag']},flb={cfg['corr_flip_lookback_xag']},mf={cfg['corr_max_flips_xag']}) | "
            f"EUR(win={cfg['corr_window_eur']},abs>={cfg['corr_min_abs_eur']},flb={cfg['corr_flip_lookback_eur']},mf={cfg['corr_max_flips_eur']})"
        )

        out_path = out_dir / f"top_{cfg_id}.html"
        report_periods_equity_only(
            periods=bts,
            out_path=out_path,
            title=title,
            freq="5MIN",
            initial_capital={k: 1000.0 for k in bts},
        )
        report_paths.append(str(out_path))

    # Summary markdown
    def _fmt_table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
        view = df[cols].head(n)
        return view.to_markdown(index=False)

    cols_show = [
        "config_id",
        "mean_sharpe",
        "worst_max_drawdown",
        "sum_pnl_usd",
        "sharpe_2023_2025",
        "sharpe_2026",
        "corr_logic",
        "corr_window_xag",
        "corr_min_abs_xag",
        "corr_flip_lookback_xag",
        "corr_max_flips_xag",
        "corr_window_eur",
        "corr_min_abs_eur",
        "corr_flip_lookback_eur",
        "corr_max_flips_eur",
    ]

    by_mean_sharpe = df_agg.sort_values(["mean_sharpe", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    by_worst_dd = df_agg.sort_values(["worst_max_drawdown", "mean_sharpe"], ascending=[False, False]).reset_index(drop=True)
    by_2325 = df_agg.sort_values(["sharpe_2023_2025", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    by_2026 = df_agg.sort_values(["sharpe_2026", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)

    md = []
    md.append(f"# Corr-stability hyperparam search (best_trend)\n")
    md.append(f"Generated: {grid_meta['created_utc']}\n")
    md.append(f"Configs evaluated: {len(configs)} (seed={args.seed}, n_samples={args.n_samples} + baseline forced)\n")
    md.append("## Top by mean Sharpe (across all 3 periods)\n")
    md.append(_fmt_table(by_mean_sharpe, cols_show, n=15))
    md.append("\n## Top by worst MaxDD (least-negative drawdown across periods)\n")
    md.append(_fmt_table(by_worst_dd, cols_show, n=15))
    md.append("\n## Top by Sharpe (2023-2025)\n")
    md.append(_fmt_table(by_2325, cols_show, n=15))
    md.append("\n## Top by Sharpe (2026-01-01..2026-02-13)\n")
    md.append(_fmt_table(by_2026, cols_show, n=15))
    md.append("\n## HTML reports generated (top-k by mean Sharpe)\n")
    for p in report_paths:
        md.append(f"- {p}")

    (out_dir / "corr_search_summary.md").write_text("\n".join(md) + "\n")

    print(f"Wrote: {out_dir / 'corr_search_results_long.csv'}")
    print(f"Wrote: {out_dir / 'corr_search_results_agg.csv'}")
    print(f"Wrote: {out_dir / 'corr_search_summary.md'}")


if __name__ == "__main__":
    main()
