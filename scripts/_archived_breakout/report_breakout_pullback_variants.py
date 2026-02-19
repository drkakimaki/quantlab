from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.report_periods import report_periods_equity_only
from quantlab.strategies.breakout_based import blueprint_a_v2_ohlc


def load_dukascopy_5m_ohlc_from_daily_parquet(*, symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.DataFrame:
    paths = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(str(p))
        cur += one

    if not paths:
        raise SystemExit(f"No 5m OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    import polars as pl

    df = (
        pl.scan_parquet(paths)
        .select(["ts", "open", "high", "low", "close"])
        .sort("ts")
        .collect(engine="streaming")
    )
    return df.to_pandas().set_index("ts").sort_index()


@dataclass(frozen=True)
class Variant:
    name: str
    params: dict


def main() -> None:
    root = Path("data/dukascopy_5m_ohlc")
    out_dir = Path("reports/breakdowpullbacks")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol = "XAUUSD"

    p3_end = dt.date(2026, 2, 10)
    bars_2022 = load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=dt.date(2022, 1, 1), end=dt.date(2022, 12, 31), root=root)
    bars_2325 = load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=dt.date(2023, 1, 1), end=dt.date(2025, 12, 31), root=root)
    bars_2026 = load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=dt.date(2026, 1, 1), end=p3_end, root=root)

    # Variant set: each corresponds to a subset of your suggestions.
    variants = [
        Variant(
            "v00_base",
            dict(
                # close-only breakout definition (no extra filters)
                breakout_close_atr=None,
                breakout_body_frac=None,
                breakout_range_atr=None,
                pullback_timeout_bars=None,
                no_reentry_into_range_atr=None,
                entry_2step=False,
                pullback_retrace_min=None,
                pullback_retrace_max=None,
                chop_crosses_max=None,
                htf=None,
                adaptive_n=False,
            ),
        ),
        Variant(
            "v01_break_close_buf",
            dict(breakout_close_atr=0.15),
        ),
        Variant(
            "v02_break_close_buf_body",
            dict(breakout_close_atr=0.15, breakout_body_frac=0.55),
        ),
        Variant(
            "v03_break_close_buf_body_impulse",
            dict(breakout_close_atr=0.15, breakout_body_frac=0.55, breakout_range_atr=1.2),
        ),
        Variant(
            "v04_pullback_timeout_6",
            dict(pullback_timeout_bars=6),
        ),
        Variant(
            "v05_no_reentry_range",
            dict(no_reentry_into_range_atr=0.10),
        ),
        Variant(
            "v06_entry_2step",
            dict(entry_2step=True),
        ),
        Variant(
            "v07_chop_filter",
            dict(chop_crosses_max=7),
        ),
        Variant(
            "v08_htf_15m",
            dict(htf="15min"),
        ),
        Variant(
            "v09_adaptive_n",
            dict(adaptive_n=True, n_breakout_low=16, n_breakout_high=26, adaptive_atr_ratio=1.25),
        ),
        Variant(
            "v10_combo",
            dict(
                breakout_close_atr=0.15,
                breakout_body_frac=0.55,
                breakout_range_atr=1.2,
                pullback_timeout_bars=6,
                no_reentry_into_range_atr=0.10,
                entry_2step=True,
                chop_crosses_max=7,
                htf="15min",
                adaptive_n=True,
            ),
        ),
    ]

    summary_rows = []

    for v in variants:
        params = v.params.copy()
        # keep defaults from blueprint_a_v2_ohlc unless overridden

        bt22, _, ex22 = blueprint_a_v2_ohlc(bars_2022, long_only=True, **params)
        bt23, _, ex23 = blueprint_a_v2_ohlc(bars_2325, long_only=True, **params)
        bt26, _, ex26 = blueprint_a_v2_ohlc(bars_2026, long_only=True, **params)

        n_trades = {"2022": int(ex22 // 2), "2023-2025": int(ex23 // 2), "2026": int(ex26 // 2)}

        out_path = out_dir / f"{v.name}.html"
        report_periods_equity_only(
            periods={"2022": bt22, "2023-2025": bt23, "2026": bt26},
            out_path=out_path,
            title=f"XAUUSD breakout_pullb {v.name}",
            freq="5MIN",
            initial_capital=1.0,
            n_trades=n_trades,
        )

        # Extract headline metrics from the generated bt frames
        def headline(bt: pd.DataFrame) -> dict:
            eq = bt["equity"].astype(float)
            ret = (float(eq.iloc[-1]) - 1.0) * 100.0
            peak = eq.cummax()
            dd = (eq / peak - 1.0).min() * 100.0
            return {"ret%": ret, "mdd%": float(dd)}

        r22 = headline(bt22)
        r23 = headline(bt23)
        r26 = headline(bt26)

        summary_rows.append(
            {
                "variant": v.name,
                "2022_ret%": r22["ret%"],
                "2022_mdd%": r22["mdd%"],
                "2022_tr": n_trades["2022"],
                "2325_ret%": r23["ret%"],
                "2325_mdd%": r23["mdd%"],
                "2325_tr": n_trades["2023-2025"],
                "2026_ret%": r26["ret%"],
                "2026_mdd%": r26["mdd%"],
                "2026_tr": n_trades["2026"],
            }
        )

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {len(variants)} reports to {out_dir}")
    print("Summary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
