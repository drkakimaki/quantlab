from __future__ import annotations

import datetime as dt
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


def main() -> None:
    root = Path("data/dukascopy_5m_ohlc")
    out_dir = Path("reports/breakdowpullbacks")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol = "XAUUSD"
    p3_end = dt.date(2026, 2, 10)

    bars_2022 = load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=dt.date(2022, 1, 1), end=dt.date(2022, 12, 31), root=root)
    bars_2325 = load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=dt.date(2023, 1, 1), end=dt.date(2025, 12, 31), root=root)
    bars_2026 = load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=dt.date(2026, 1, 1), end=p3_end, root=root)

    params = dict(
        entry_2step=True,
        allow_pyramid=True,
        pyramid_atr=0.7,
        pyramid_style="risk_neutral_second_reclaim",
        long_only=True,
        # keep other feature toggles OFF for clean comparison
        breakout_close_atr=None,
        breakout_body_frac=None,
        breakout_range_atr=None,
        pullback_timeout_bars=None,
        no_reentry_into_range_atr=None,
        pullback_retrace_min=None,
        pullback_retrace_max=None,
        chop_crosses_max=None,
        htf=None,
        adaptive_n=False,
    )

    bt22, _, ex22 = blueprint_a_v2_ohlc(bars_2022, **params)
    bt23, _, ex23 = blueprint_a_v2_ohlc(bars_2325, **params)
    bt26, _, ex26 = blueprint_a_v2_ohlc(bars_2026, **params)

    n_trades = {"2022": int(ex22 // 2), "2023-2025": int(ex23 // 2), "2026": int(ex26 // 2)}

    out_path = out_dir / "v11_risk_neutral_pyramid.html"
    report_periods_equity_only(
        periods={"2022": bt22, "2023-2025": bt23, "2026": bt26},
        out_path=out_path,
        title="XAUUSD breakout_pullb v11_risk_neutral_pyramid",
        freq="5MIN",
        initial_capital=1.0,
        n_trades=n_trades,
    )

    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
