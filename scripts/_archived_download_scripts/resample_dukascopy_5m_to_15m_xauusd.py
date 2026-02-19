from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"Expected column 'ts' in {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")
    return df


def resample_mid_5m_to_15m(in_path: Path, out_path: Path) -> None:
    df = _read_parquet(in_path)
    if "mid" not in df.columns:
        raise ValueError(f"Expected column 'mid' in {in_path}")
    s = df.set_index("ts")["mid"].astype(float)
    out = s.resample("15min", label="right", closed="right").last().dropna()
    out_df = out.reset_index().rename(columns={"index": "ts"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)


def resample_ohlc_5m_to_15m(in_path: Path, out_path: Path) -> None:
    df = _read_parquet(in_path)
    need = {"open", "high", "low", "close"}
    cols = {c.lower(): c for c in df.columns}
    if not need.issubset(set(cols.keys())):
        raise ValueError(f"Expected OHLC columns in {in_path}; have {sorted(df.columns)}")

    df = df.rename(columns={cols[k]: k for k in need})
    df = df.set_index("ts")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    out = df[list(need)].astype(float).resample("15min", label="right", closed="right").agg(agg).dropna()
    out_df = out.reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)


def _parse_date_from_name(name: str):
    import datetime as dt

    stem = name.replace(".parquet", "")
    return dt.date.fromisoformat(stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resample Dukascopy precomputed daily 5m data to daily 15m files.")
    ap.add_argument("--symbol", type=str, required=True)
    ap.add_argument("--root-5m", type=Path, default=Path("data/dukascopy_5m"))
    ap.add_argument("--root-5m-ohlc", type=Path, default=Path("data/dukascopy_5m_ohlc"))
    ap.add_argument("--out-15m", type=Path, default=Path("data/dukascopy_15m"))
    ap.add_argument("--out-15m-ohlc", type=Path, default=Path("data/dukascopy_15m_ohlc"))
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (optional; limit days)")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (optional; limit days)")
    args = ap.parse_args()

    sym = args.symbol

    start = _parse_date_from_name(f"{args.start}.parquet") if args.start else None
    end = _parse_date_from_name(f"{args.end}.parquet") if args.end else None

    mid_files = sorted((args.root_5m / sym).rglob("*.parquet"))
    ohlc_files = sorted((args.root_5m_ohlc / sym).rglob("*.parquet"))

    if not mid_files:
        raise SystemExit(f"No 5m mid parquet files found under {args.root_5m/sym}")
    if not ohlc_files:
        raise SystemExit(f"No 5m OHLC parquet files found under {args.root_5m_ohlc/sym}")

    # Map by date filename (YYYY-MM-DD.parquet)
    mid_map = {p.name: p for p in mid_files}
    ohlc_map = {p.name: p for p in ohlc_files}

    common = sorted(set(mid_map.keys()) & set(ohlc_map.keys()))
    if not common:
        raise SystemExit("No overlapping daily files between 5m and 5m_ohlc")

    if start is not None or end is not None:
        filt = []
        for name in common:
            d = _parse_date_from_name(name)
            if start is not None and d < start:
                continue
            if end is not None and d > end:
                continue
            filt.append(name)
        common = filt

    n = 0
    for name in common:
        mid_in = mid_map[name]
        ohlc_in = ohlc_map[name]
        # year folder inferred from parents
        year = mid_in.parent.name

        mid_out = args.out_15m / sym / year / name
        ohlc_out = args.out_15m_ohlc / sym / year / name

        # Always overwrite to keep it deterministic.
        resample_mid_5m_to_15m(mid_in, mid_out)
        resample_ohlc_5m_to_15m(ohlc_in, ohlc_out)

        n += 1
        if n % 200 == 0:
            print(f"Processed {n}/{len(common)} days...")

    print(f"Done. Wrote {n} daily 15m mid files to: {args.out_15m/sym}")
    print(f"Done. Wrote {n} daily 15m OHLC files to: {args.out_15m_ohlc/sym}")


if __name__ == "__main__":
    main()
