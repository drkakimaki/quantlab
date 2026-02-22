"""Generate trade breakdown HTML + print key aggregates for current best_trend config.

Usage:
  ../.venv/bin/python scripts/generate_trades_breakdown_current.py \
    --config configs/trend_based/current.yaml \
    --out reports/trend_based/trades_breakdown.html \
    --mode three_block

This script is intentionally lightweight and reuses rnd/webui loaders for parity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from quantlab.rnd import _run_best_trend_periods
from quantlab.reporting.generate_trades_report import report_periods_trades_html
from quantlab.reporting.trade_breakdown import build_trade_ledger, agg_trade_table, duration_bin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="three_block", choices=["three_block", "yearly"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    # Override mode like rnd does
    cfg = dict(cfg)
    p = dict(cfg.get("periods", {}) or {})
    p["mode"] = args.mode
    cfg["periods"] = p

    period_dfs = _run_best_trend_periods(cfg)

    out_path = report_periods_trades_html(
        periods=period_dfs,
        out_path=args.out,
        title=f"Trade breakdown — best_trend ({args.mode})",
    )

    # Print compact aggregates for quick chat iteration
    ledgers = []
    for name, bt in period_dfs.items():
        tl = build_trade_ledger(bt)
        if tl is None or tl.empty:
            continue
        tl = tl.copy()
        tl["period"] = name
        et = pd.to_datetime(tl["entry_time"])
        if getattr(et.dt, "tz", None) is not None:
            et = et.dt.tz_convert("UTC").dt.tz_localize(None)
        tl["entry_month"] = et.dt.month
        tl["duration_bin"] = duration_bin(tl["bars"])
        ledgers.append(tl)

    all_trades = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()

    if all_trades.empty:
        print("No trades.")
        print(f"Wrote: {out_path}")
        return 0

    by_period = agg_trade_table(all_trades, "period")
    by_dur = agg_trade_table(all_trades, "duration_bin")
    by_month = agg_trade_table(all_trades, "entry_month")

    # order months 1..12
    by_month = by_month.sort_values("entry_month")

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 140)

    print("=== By period ===")
    print(by_period.to_string(index=False))
    print("\n=== By duration_bin ===")
    print(by_dur.to_string(index=False))
    print("\n=== By entry month ===")
    print(by_month.to_string(index=False))

    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
