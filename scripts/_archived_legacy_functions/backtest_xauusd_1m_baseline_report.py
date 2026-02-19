from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from quantlab import PeriodResult, ReportConfig, buy_and_hold, write_html_report_periods


def _iter_days(start: dt.date, end: dt.date):
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def _load_1m_close(symbol: str, start: dt.date, end: dt.date, root: Path) -> pd.Series:
    parts = []
    for day in _iter_days(start, end):
        p = root / symbol / str(day.year) / f"{day.isoformat()}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # expected columns: ts, close
        if "ts" not in df.columns or "close" not in df.columns:
            raise KeyError(f"Unexpected schema in {p}: {df.columns}")
        d = df[["ts", "close"]].copy()
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        d = d.set_index("ts").sort_index()
        parts.append(d["close"].rename(day.isoformat()))

    if not parts:
        raise SystemExit(f"No 1m data found for {symbol} in {root} for {start}..{end}")

    s = pd.concat(parts).sort_index()
    s.name = symbol
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline XAUUSD buy&hold multi-period HTML report (1m bars).")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--data-root", type=Path, default=Path("data/dukascopy_1m"))
    ap.add_argument("--out", type=Path, default=Path("reports/xauusd_baseline_periods.html"))
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--lots", type=float, default=1.0)
    args = ap.parse_args()

    # Define your test windows
    windows = {
        "2022": (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
        "2023-2025": (dt.date(2023, 1, 1), dt.date(2025, 12, 31)),
        "2026": (dt.date(2026, 1, 1), dt.date(2026, 12, 31)),
    }

    # Load the full span needed
    start_all = min(s for s, _ in windows.values())
    end_all = max(e for _, e in windows.values())
    close = _load_1m_close(args.symbol, start_all, end_all, args.data_root)

    periods: dict[str, PeriodResult] = {}
    for name, (start, end) in windows.items():
        # Clamp to available data
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        sub = close.loc[(close.index >= start_ts) & (close.index <= end_ts)]
        if sub.empty:
            continue

        bt, pnl, n_trades = buy_and_hold(
            sub,
            start=sub.index[0],
            end=sub.index[-1],
            fee_bps=args.fee_bps,
            slippage_bps=args.slip_bps,
            lot_size=args.lots,
            contract_size=100.0,
        )
        periods[name] = PeriodResult(bt=bt, pnl=pnl, n_trades=n_trades)

    cfg = ReportConfig(
        title=f"{args.symbol} baseline (buy at start, sell at end) — 1 lot",
        freq="MIN",
    )
    p = write_html_report_periods(periods, args.out, config=cfg)
    print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
