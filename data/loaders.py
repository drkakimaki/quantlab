from __future__ import annotations

"""Data loaders (no Polars dependency).

This module is intentionally Polars-free so that the backtest stack (rnd/webui)
can load existing parquet OHLC data without requiring the resampling toolchain.

Resampling code lives in :mod:`quantlab.data.resampling`.
"""

import datetime as dt
from pathlib import Path

import pandas as pd


def _daily_paths(*, root: Path, symbol: str, start: dt.date, end: dt.date) -> list[Path]:
    paths: list[Path] = []
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        p = root / symbol / str(cur.year) / f"{cur.isoformat()}.parquet"
        if p.exists():
            paths.append(p)
        cur += one
    return paths


def load_dukascopy_ohlc(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    root: str | Path = Path("data/dukascopy_5m_ohlc"),
) -> pd.DataFrame:
    """Load pre-resampled OHLC bars from daily parquet files.

    Expects files:
      {root}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Schema:
      ts, open, high, low, close

    Returns:
      DataFrame indexed by UTC timestamps with OHLC columns.

    Notes
    -----
    This is the "loader" path used by the backtest stack.
    It avoids Polars to keep imports lightweight.
    """
    root = Path(root)
    paths = _daily_paths(root=root, symbol=symbol, start=start, end=end)
    if not paths:
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} in {root} between {start} and {end}")

    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p, columns=["ts", "open", "high", "low", "close"])
        dfs.append(df)

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.sort_values("ts")
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.set_index("ts")

    # Ensure UTC tz-aware index.
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    return out


def resample_last(prices: pd.Series, rule: str) -> pd.Series:
    """Resample a price series to a lower frequency using last-observation in each bin."""
    px = prices.dropna().copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    out = px.resample(rule).last().dropna()
    out.name = prices.name
    return out
