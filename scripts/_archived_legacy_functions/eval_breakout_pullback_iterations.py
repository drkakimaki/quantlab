from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.metrics import sharpe
from quantlab.report_periods import report_periods_equity_only
from quantlab.strategies import breakout_pullback_iterative


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def load_dukascopy_5m_mid_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.Series:
    paths: list[str] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise SystemExit(f"No 5m parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = pl.scan_parquet(paths).select(["ts", "mid"]).sort("ts").collect(engine="streaming")
    s = df.to_pandas().set_index("ts")["mid"].sort_index()
    s.name = symbol
    return s


@dataclass(frozen=True)
class IterCfg:
    name: str
    params: dict


def max_drawdown_pct(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min() * 100.0)


def run_one(px: pd.Series, cfg: IterCfg) -> dict:
    bt, _pnl, executions = breakout_pullback_iterative(px, **cfg.params)
    if bt is None or len(bt) == 0:
        return {"ret%": np.nan, "maxdd%": np.nan, "sharpe": np.nan, "trades": 0}

    eq = bt["equity"].astype(float)
    r = bt["returns_net"].astype(float)

    ret_pct = (float(eq.iloc[-1]) - 1.0) * 100.0
    mdd = max_drawdown_pct(eq)
    s = float(sharpe(r, freq="MIN"))

    trades = int(executions // 2)

    return {"ret%": ret_pct, "maxdd%": mdd, "sharpe": s, "trades": trades}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate 10 breakout+pullback iterations on XAUUSD 5m.")
    ap.add_argument("--root-5m", type=Path, default=Path("data/dukascopy_5m"))
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/breakout_pullback_iters"))

    ap.add_argument("--p1-start", type=parse_date, default=dt.date(2022, 1, 1))
    ap.add_argument("--p1-end", type=parse_date, default=dt.date(2022, 12, 31))
    ap.add_argument("--p2-start", type=parse_date, default=dt.date(2023, 1, 1))
    ap.add_argument("--p2-end", type=parse_date, default=dt.date(2025, 12, 31))
    ap.add_argument("--p3-start", type=parse_date, default=dt.date(2026, 1, 1))
    ap.add_argument("--p3-end", type=parse_date, default=None)

    args = ap.parse_args()
    p3_end = args.p3_end or dt.date.today()

    # Load per period (keeps it light)
    px_2022 = load_dukascopy_5m_mid_from_daily_parquet(symbol=args.symbol, start=args.p1_start, end=args.p1_end, root=args.root_5m)
    px_2325 = load_dukascopy_5m_mid_from_daily_parquet(symbol=args.symbol, start=args.p2_start, end=args.p2_end, root=args.root_5m)
    px_2026 = load_dukascopy_5m_mid_from_daily_parquet(symbol=args.symbol, start=args.p3_start, end=p3_end, root=args.root_5m)

    # 10 iterations: each one adds a practical guardrail / exit tweak.
    iters: list[IterCfg] = [
        # IMPORTANT: match the original aggressive baseline as closely as possible.
        # The original strategy had *no pullback timeout*; using a timeout massively reduces entries.
        IterCfg(
            "aggressive",
            dict(
                n_breakout=16,
                breakout_buffer_atr=0.0,
                pullback_timeout_bars=None,
                ema_band_width_atr_min=0.0,
                reclaim_requires_momo=False,
                use_chandelier=False,
                take_partial_tp1=False,
                allow_pyramid=True,
            ),
        ),
        IterCfg(
            "iter02_breakout_buffer",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=12, use_chandelier=False, take_partial_tp1=False, allow_pyramid=True),
        ),
        IterCfg(
            "iter03_timeout_tighter",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, use_chandelier=False, take_partial_tp1=False, allow_pyramid=True),
        ),
        IterCfg(
            "iter04_ema_band_filter",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, ema_band_width_atr_min=0.10, use_chandelier=False, take_partial_tp1=False, allow_pyramid=True),
        ),
        IterCfg(
            "iter05_reclaim_momentum",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, ema_band_width_atr_min=0.10, reclaim_requires_momo=True, use_chandelier=False, take_partial_tp1=False, allow_pyramid=True),
        ),
        IterCfg(
            "iter06_chandelier",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, ema_band_width_atr_min=0.10, reclaim_requires_momo=True, use_chandelier=True, chandelier_atr=2.2, take_partial_tp1=False, allow_pyramid=True),
        ),
        IterCfg(
            "iter07_partial_tp1",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, ema_band_width_atr_min=0.10, reclaim_requires_momo=True, use_chandelier=True, chandelier_atr=2.2, take_partial_tp1=True, tp1_fraction=0.5, allow_pyramid=True),
        ),
        IterCfg(
            "iter08_max_hold",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, ema_band_width_atr_min=0.10, reclaim_requires_momo=True, use_chandelier=True, chandelier_atr=2.2, take_partial_tp1=True, tp1_fraction=0.5, max_hold_bars=96, allow_pyramid=True),
        ),
        IterCfg(
            "iter09_no_pyramid",
            dict(n_breakout=16, breakout_buffer_atr=0.15, pullback_timeout_bars=8, ema_band_width_atr_min=0.10, reclaim_requires_momo=True, use_chandelier=True, chandelier_atr=2.2, take_partial_tp1=True, tp1_fraction=0.5, max_hold_bars=96, allow_pyramid=False),
        ),
        IterCfg(
            "iter10_wider_stop",
            dict(n_breakout=20, breakout_buffer_atr=0.20, pullback_timeout_bars=8, ema_band_width_atr_min=0.12, reclaim_requires_momo=True, use_chandelier=True, chandelier_atr=2.4, take_partial_tp1=True, tp1_fraction=0.5, max_hold_bars=96, stop_atr=1.35, tp2_atr=2.2, allow_pyramid=False),
        ),
    ]

    rows = []
    for cfg in iters:
        r1 = run_one(px_2022, cfg)
        r2 = run_one(px_2325, cfg)
        r3 = run_one(px_2026, cfg)

        # simple score: average Sharpe (nan-safe) with a soft drawdown penalty
        sharpe_avg = np.nanmean([r1["sharpe"], r2["sharpe"], r3["sharpe"]])
        dd_pen = np.nanmean([abs(r1["maxdd%"]), abs(r2["maxdd%"]), abs(r3["maxdd%"])] )
        score = float(sharpe_avg) - 0.02 * float(dd_pen)

        rows.append(
            {
                "iter": cfg.name,
                "score": score,
                "2022_ret%": r1["ret%"],
                "2022_dd%": r1["maxdd%"],
                "2022_sh": r1["sharpe"],
                "2022_tr": r1["trades"],
                "2325_ret%": r2["ret%"],
                "2325_dd%": r2["maxdd%"],
                "2325_sh": r2["sharpe"],
                "2325_tr": r2["trades"],
                "2026_ret%": r3["ret%"],
                "2026_dd%": r3["maxdd%"],
                "2026_sh": r3["sharpe"],
                "2026_tr": r3["trades"],
            }
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "iterations_summary.csv"
    df.to_csv(csv_path, index=False)

    best = iters[[c.name for c in iters].index(df.loc[0, "iter"]) ]

    # Write an HTML report for the best iteration (so you can inspect equity curves)
    bt_2022, _, ex_2022 = breakout_pullback_iterative(px_2022, **best.params)
    bt_2325, _, ex_2325 = breakout_pullback_iterative(px_2325, **best.params)
    bt_2026, _, ex_2026 = breakout_pullback_iterative(px_2026, **best.params)

    n_trades = {"2022": int(ex_2022 // 2), "2023-2025": int(ex_2325 // 2), "2026": int(ex_2026 // 2)}

    title = f"XAUUSD breakout+pullback BEST: {best.name}"
    out_html = args.out_dir / f"best_{best.name}.html"
    report_periods_equity_only(
        periods={"2022": bt_2022, "2023-2025": bt_2325, "2026": bt_2026},
        out_path=out_html,
        title=title,
        freq="MIN",
        initial_capital=1.0,
        n_trades=n_trades,
    )

    print("Wrote:", csv_path)
    print("Best:", best.name)
    print("Best params:", best.params)
    print("Best report:", out_html)


if __name__ == "__main__":
    main()
