"""Compare trade breakdown aggregates between two configs.

Prints deltas for:
- by duration_bin
- by entry_month
- by period

Usage:
  ../.venv/bin/python scripts/compare_trade_breakdowns.py \
    --a <configA.yaml> --b <configB.yaml>

A is treated as baseline, B as candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from quantlab.rnd import _run_best_trend_periods
from quantlab.reporting.trade_breakdown import build_trade_ledger, agg_trade_table, duration_bin


def _load_period_dfs(cfg_path: str, mode: str = "three_block") -> dict[str, pd.DataFrame]:
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    cfg = dict(cfg)
    p = dict(cfg.get("periods", {}) or {})
    p["mode"] = mode
    cfg["periods"] = p
    return _run_best_trend_periods(cfg)


def _ledger(period_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    return pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()


def _cmp(a: pd.DataFrame, b: pd.DataFrame, key: str) -> pd.DataFrame:
    A = agg_trade_table(a, key).set_index(key)
    B = agg_trade_table(b, key).set_index(key)

    cols = ["n_trades", "sum_pnl", "win_rate", "avg_return", "profit_factor", "avg_bars"]
    A = A[cols]
    B = B[cols]

    out = A.join(B, lsuffix="_a", rsuffix="_b", how="outer").fillna(0.0)

    out["dn_trades"] = out["n_trades_b"] - out["n_trades_a"]
    out["dsum_pnl"] = out["sum_pnl_b"] - out["sum_pnl_a"]
    out["dwin_rate"] = out["win_rate_b"] - out["win_rate_a"]
    out["davg_return"] = out["avg_return_b"] - out["avg_return_a"]

    # keep readable subset
    keep = [
        "n_trades_a","n_trades_b","dn_trades",
        "sum_pnl_a","sum_pnl_b","dsum_pnl",
        "win_rate_a","win_rate_b","dwin_rate",
        "avg_return_a","avg_return_b","davg_return",
        "profit_factor_a","profit_factor_b",
        "avg_bars_a","avg_bars_b",
    ]
    return out[keep].reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline config")
    ap.add_argument("--b", required=True, help="candidate config")
    ap.add_argument("--mode", default="three_block", choices=["three_block","yearly"])
    args = ap.parse_args()

    pA = _load_period_dfs(args.a, mode=args.mode)
    pB = _load_period_dfs(args.b, mode=args.mode)

    a = _ledger(pA)
    b = _ledger(pB)

    if a.empty or b.empty:
        print("No trades in one side.")
        return 0

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)

    print("=== By period (A->B deltas) ===")
    print(_cmp(a, b, "period").to_string(index=False))

    print("\n=== By duration_bin (A->B deltas) ===")
    print(_cmp(a, b, "duration_bin").to_string(index=False))

    print("\n=== By entry_month (A->B deltas) ===")
    print(_cmp(a, b, "entry_month").sort_values("entry_month").to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
