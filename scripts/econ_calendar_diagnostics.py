"""Diagnostics for econ_calendar time filter.

Reports per period:
- blocked bars (by rule)
- blocked bars while pre-time-filter position is ON (i.e. would be forced-flat)

We compute "pre-time-filter position" by running TrendStrategyWithGates with the
same config but with time_filter disabled.

Usage:
  ../.venv/bin/python scripts/econ_calendar_diagnostics.py --config configs/trend_based/current.yaml
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantlab.webui.periods import build_periods
from quantlab.webui.runner import load_period_data, load_time_filter_mask
from quantlab.webui.config import WORKSPACE
from quantlab.strategies.trend_following import TrendStrategyWithGates


def _disable_time_filter(cfg: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    # Turn off time filter gate by removing the block.
    c["time_filter"] = None
    return c


def _count(mask: pd.Series) -> int:
    return int(pd.Series(mask).fillna(False).astype(bool).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    periods = build_periods(cfg)

    symbol = cfg.get("symbol", "XAUUSD")
    corr_symbol = cfg.get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg.get("corr2_symbol", "EURUSD")

    tf = cfg.get("time_filter", {}) or {}
    kind = (tf.get("kind") or "fomc").strip().lower()

    if kind not in {"econ_calendar", "econ", "calendar"}:
        print(f"time_filter.kind={kind!r} (not econ_calendar) — nothing to diagnose")
        return 0

    ec = tf.get("econ_calendar", {}) or {}
    rules = ec.get("rules", {}) or {}

    rows = []

    for name, start, end in periods:
        data = load_period_data(
            symbol,
            start,
            end,
            need_htf=True,
            need_corr=True,
            corr_symbol=corr_symbol,
            corr2_symbol=corr2_symbol,
        )
        idx = pd.DatetimeIndex(data["prices"].index)

        # Pre-time-filter positions: same config but time_filter disabled.
        cfg_pre = _disable_time_filter(cfg)
        strat_pre = TrendStrategyWithGates.from_config(cfg_pre, allow_mask=None)
        context_pre = {
            "bars_15m": data.get("bars_15m"),
            "prices_xag": data.get("prices_xag"),
            "prices_eur": data.get("prices_eur"),
        }
        pos_pre = strat_pre.generate_positions(data["prices"], context=context_pre).fillna(0.0).astype(float)
        in_pos = pos_pre > 0.0

        # Combined allow mask (all rules)
        allow_all = load_time_filter_mask(idx, start, end, cfg=cfg, workspace=WORKSPACE)
        if allow_all is None:
            allow_all = pd.Series(True, index=idx)
        blocked_all = (~pd.Series(allow_all, index=idx).astype(bool))

        r = {
            "period": name,
            "bars": int(len(idx)),
            "blocked_all": _count(blocked_all),
            "blocked_all_in_pos": int((blocked_all & in_pos).sum()),
        }

        # Per-rule masks (build by temporarily keeping only one rule)
        for rule_name in sorted(rules.keys()):
            cfg_one = copy.deepcopy(cfg)
            cfg_one.setdefault("time_filter", {})
            cfg_one["time_filter"] = copy.deepcopy(tf)
            cfg_one.setdefault("time_filter", {}).setdefault("econ_calendar", {})
            cfg_one["time_filter"]["econ_calendar"] = copy.deepcopy(ec)
            cfg_one["time_filter"]["econ_calendar"]["rules"] = {rule_name: rules[rule_name]}

            allow_one = load_time_filter_mask(idx, start, end, cfg=cfg_one, workspace=WORKSPACE)
            if allow_one is None:
                allow_one = pd.Series(True, index=idx)
            blocked_one = (~pd.Series(allow_one, index=idx).astype(bool))

            key_b = f"blocked_{rule_name}"
            key_ip = f"blocked_{rule_name}_in_pos"
            r[key_b] = _count(blocked_one)
            r[key_ip] = int((blocked_one & in_pos).sum())

        rows.append(r)

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # Totals row
    num_cols = [c for c in df.columns if c != "period"]
    tot = {"period": "TOTAL"}
    for c in num_cols:
        try:
            tot[c] = float(df[c].sum())
        except Exception:
            pass
    df2 = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    print("\n-- Totals (sum across periods) --")
    print(df2.tail(1).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
