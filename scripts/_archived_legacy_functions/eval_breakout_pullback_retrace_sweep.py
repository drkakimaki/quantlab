from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.metrics import sharpe
from quantlab.report_periods import _win_rate_from_position
from quantlab.strategies import breakout_pullback_ohlc


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


def max_dd_pct(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min() * 100.0)


@dataclass(frozen=True)
class Cfg:
    name: str
    rmin: float | None
    rmax: float | None


def run_period(bars: pd.DataFrame, cfg: Cfg) -> dict:
    bt, _pnl, executions = breakout_pullback_ohlc(
        bars,
        pullback_retrace_min=cfg.rmin,
        pullback_retrace_max=cfg.rmax,
        long_only=True,
    )
    eq = bt["equity"].astype(float)
    r = bt["returns_net"].astype(float)

    ret_pct = (float(eq.iloc[-1]) - 1.0) * 100.0
    mdd = max_dd_pct(eq)
    sh = float(sharpe(r, freq="5MIN"))
    wr = float(_win_rate_from_position(bt))
    trades = int(executions // 2)

    return {"ret%": ret_pct, "mdd%": mdd, "sh": sh, "wr%": wr, "tr": trades}


def main() -> None:
    root = Path("data/dukascopy_5m_ohlc")
    symbol = "XAUUSD"

    periods = {
        "2022": (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), dt.date(2026, 2, 10)),
    }

    bars = {k: load_dukascopy_5m_ohlc_from_daily_parquet(symbol=symbol, start=a, end=b, root=root) for k, (a, b) in periods.items()}

    cfgs = [
        Cfg("retrace_off", None, None),
        Cfg("0.20-0.70", 0.20, 0.70),
        Cfg("0.30-0.60", 0.30, 0.60),
        Cfg("0.25-0.65", 0.25, 0.65),
        Cfg("0.30-0.70", 0.30, 0.70),
    ]

    rows = []
    for cfg in cfgs:
        row = {"cfg": cfg.name}
        scores = []
        for pname, bdf in bars.items():
            r = run_period(bdf, cfg)
            row.update({f"{pname}_ret%": r["ret%"], f"{pname}_mdd%": r["mdd%"], f"{pname}_sh": r["sh"], f"{pname}_wr%": r["wr%"], f"{pname}_tr": r["tr"]})
            scores.append(r["sh"])
        row["avg_sh"] = float(np.nanmean(scores))
        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values("avg_sh", ascending=False)

    # Print concise markdown-ish output
    cols = [
        "cfg",
        "avg_sh",
        "2022_ret%",
        "2022_mdd%",
        "2022_sh",
        "2022_wr%",
        "2022_tr",
        "2023-2025_ret%",
        "2023-2025_mdd%",
        "2023-2025_sh",
        "2023-2025_wr%",
        "2023-2025_tr",
        "2026_ret%",
        "2026_mdd%",
        "2026_sh",
        "2026_wr%",
        "2026_tr",
    ]
    print(df_out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
