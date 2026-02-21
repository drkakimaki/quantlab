from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lzma
import numpy as np
import polars as pl
import requests


DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"


@dataclass(frozen=True)
class DukascopySymbolSpec:
    symbol: str
    price_scale: int  # price = int_price / 10**price_scale


SYMBOL_SPECS: dict[str, DukascopySymbolSpec] = {
    "XAUUSD": DukascopySymbolSpec("XAUUSD", price_scale=3),
    "XAGUSD": DukascopySymbolSpec("XAGUSD", price_scale=3),
    # FX majors
    "EURUSD": DukascopySymbolSpec("EURUSD", price_scale=5),
}


def dukascopy_tick_url(symbol: str, day: dt.date, hour: int) -> str:
    # Dukascopy months are 0-indexed in the URL.
    month0 = day.month - 1
    return f"{DUKASCOPY_BASE}/{symbol}/{day.year}/{month0:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"


def _cache_path(cache_dir: Path, symbol: str, day: dt.date, hour: int) -> Path:
    month0 = day.month - 1
    return cache_dir / symbol / str(day.year) / f"{month0:02d}" / f"{day.day:02d}" / f"{hour:02d}h_ticks.bi5"


def download_bi5(url: str, out_path: Path, *, timeout_s: float = 60.0, retries: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    headers = {
        # Dukascopy sometimes behaves better with a browser-ish UA.
        "User-Agent": "Mozilla/5.0 (compatible; quantlab/0.1; +https://example.local)",
        "Accept": "*/*",
    }

    last_err: Exception | None = None
    for k in range(retries):
        try:
            with requests.get(url, headers=headers, timeout=timeout_s, stream=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, out_path)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            # simple backoff
            time.sleep(0.5 * (2**k))

    raise RuntimeError(f"Failed to download after {retries} tries: {url}") from last_err


_TICK_DTYPE = np.dtype(
    [
        ("ms", ">i4"),
        ("ask", ">i4"),
        ("bid", ">i4"),
        ("ask_vol", ">f4"),
        ("bid_vol", ">f4"),
    ]
)


def parse_ticks_bi5(path: Path) -> np.ndarray:
    """Parse a Dukascopy *tick* .bi5 file.

    Returns a numpy structured array with fields: ms, ask, bid, ask_vol, bid_vol.

    Notes:
    - Empty files (0 bytes) are common on weekends/holidays.
    """
    raw = path.read_bytes()
    if len(raw) == 0:
        return np.empty((0,), dtype=_TICK_DTYPE)

    # FORMAT_AUTO handles Dukascopy .bi5 (typically LZMA "alone" format)
    decompressed = lzma.decompress(raw, format=lzma.FORMAT_AUTO)
    if len(decompressed) == 0:
        return np.empty((0,), dtype=_TICK_DTYPE)

    if len(decompressed) % _TICK_DTYPE.itemsize != 0:
        raise ValueError(f"Unexpected decompressed size: {len(decompressed)} bytes")

    return np.frombuffer(decompressed, dtype=_TICK_DTYPE)


def ticks_to_1s_quotes(
    ticks: np.ndarray,
    *,
    symbol: str,
    day: dt.date,
    hour: int,
) -> pl.DataFrame:
    """Convert tick array -> 1s last-quote DataFrame for that hour.

    Output columns: ts (UTC), bid, ask, mid, spread
    """
    if ticks.size == 0:
        return pl.DataFrame(schema={
            "ts": pl.Datetime("ns", time_zone="UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "mid": pl.Float64,
            "spread": pl.Float64,
        })

    spec = SYMBOL_SPECS.get(symbol)
    if spec is None:
        raise KeyError(f"Unknown symbol spec for {symbol}. Add to SYMBOL_SPECS.")

    base = dt.datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=dt.UTC)
    base_ns = int(base.timestamp() * 1_000_000_000)

    ms = ticks["ms"].astype(np.int64)
    ts_ns = base_ns + ms * 1_000_000

    scale = 10 ** spec.price_scale
    bid = ticks["bid"].astype(np.float64) / scale
    ask = ticks["ask"].astype(np.float64) / scale

    df = pl.DataFrame({
        "ts_ns": ts_ns,
        "bid": bid,
        "ask": ask,
    })
    df = df.with_columns(
        pl.col("ts_ns").cast(pl.Datetime("ns", time_zone="UTC")).alias("ts"),
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    ).drop("ts_ns")

    # Dukascopy ticks are typically ordered, but sort just in case.
    df = df.sort("ts")

    # 1-second last quote
    df = (
        df.with_columns(pl.col("ts").dt.truncate("1s").alias("ts_1s"))
        .group_by("ts_1s")
        .agg(
            pl.col("bid").last().alias("bid"),
            pl.col("ask").last().alias("ask"),
            pl.col("mid").last().alias("mid"),
            pl.col("spread").last().alias("spread"),
        )
        .rename({"ts_1s": "ts"})
        .sort("ts")
    )

    return df


def iter_days(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    if end < start:
        raise ValueError("end must be >= start")
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def build_daily_1s(
    *,
    symbol: str,
    day: dt.date,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    timeout_s: float = 60.0,
    retries: int = 5,
) -> Path:
    """Download ticks (if needed) and write a daily 1s parquet file.

    Output file: {out_dir}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet
    """
    out_path = out_dir / symbol / f"{day.year}" / f"{day.isoformat()}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return out_path

    hour_frames: list[pl.DataFrame] = []
    for hour in range(24):
        url = dukascopy_tick_url(symbol, day, hour)
        p = _cache_path(cache_dir, symbol, day, hour)
        if not p.exists():
            download_bi5(url, p, timeout_s=timeout_s, retries=retries)
        ticks = parse_ticks_bi5(p)
        df1s = ticks_to_1s_quotes(ticks, symbol=symbol, day=day, hour=hour)
        if df1s.height:
            hour_frames.append(df1s)

    if hour_frames:
        df_day = pl.concat(hour_frames).unique(subset=["ts"], keep="last").sort("ts")
    else:
        df_day = pl.DataFrame(schema={
            "ts": pl.Datetime("ns", time_zone="UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "mid": pl.Float64,
            "spread": pl.Float64,
        })

    df_day.write_parquet(out_path, compression="zstd")
    return out_path


def build_daily_1m(
    *,
    symbol: str,
    day: dt.date,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    timeout_s: float = 60.0,
    retries: int = 5,
) -> Path:
    """Download ticks (if needed) and write a daily 1-minute OHLC parquet file.

    Output file: {out_dir}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Columns:
      ts (UTC minute), open, high, low, close, bid_close, ask_close, spread_close

    Notes:
    - This is still "direct" from Dukascopy ticks (not resampling your 1s files).
    - Minutes with no ticks are omitted for now (we can forward-fill later if desired).
    """
    out_path = out_dir / symbol / f"{day.year}" / f"{day.isoformat()}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return out_path

    minute_frames: list[pl.DataFrame] = []
    for hour in range(24):
        url = dukascopy_tick_url(symbol, day, hour)
        p = _cache_path(cache_dir, symbol, day, hour)
        if not p.exists():
            download_bi5(url, p, timeout_s=timeout_s, retries=retries)
        ticks = parse_ticks_bi5(p)
        if ticks.size == 0:
            continue

        spec = SYMBOL_SPECS.get(symbol)
        if spec is None:
            raise KeyError(f"Unknown symbol spec for {symbol}. Add to SYMBOL_SPECS.")

        base = dt.datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=dt.UTC)
        base_ns = int(base.timestamp() * 1_000_000_000)
        ms = ticks["ms"].astype(np.int64)
        ts_ns = base_ns + ms * 1_000_000

        scale = 10 ** spec.price_scale
        bid = ticks["bid"].astype(np.float64) / scale
        ask = ticks["ask"].astype(np.float64) / scale
        mid = (bid + ask) / 2.0
        spread = ask - bid

        df = pl.DataFrame({
            "ts_ns": ts_ns,
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "spread": spread,
        }).with_columns(
            pl.col("ts_ns").cast(pl.Datetime("ns", time_zone="UTC")).alias("ts"),
        ).drop("ts_ns").sort("ts")

        # 1-minute bucket OHLC on mid; keep last bid/ask/spread for close.
        df = (
            df.with_columns(pl.col("ts").dt.truncate("1m").alias("ts_1m"))
            .group_by("ts_1m")
            .agg(
                pl.col("mid").first().alias("open"),
                pl.col("mid").max().alias("high"),
                pl.col("mid").min().alias("low"),
                pl.col("mid").last().alias("close"),
                pl.col("bid").last().alias("bid_close"),
                pl.col("ask").last().alias("ask_close"),
                pl.col("spread").last().alias("spread_close"),
            )
            .rename({"ts_1m": "ts"})
            .sort("ts")
        )
        if df.height:
            minute_frames.append(df)

    if minute_frames:
        df_day = pl.concat(minute_frames).unique(subset=["ts"], keep="last").sort("ts")
    else:
        df_day = pl.DataFrame(schema={
            "ts": pl.Datetime("ns", time_zone="UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "bid_close": pl.Float64,
            "ask_close": pl.Float64,
            "spread_close": pl.Float64,
        })

    df_day.write_parquet(out_path, compression="zstd")
    return out_path


def latest_utc_date() -> dt.date:
    return dt.datetime.now(dt.UTC).date()
