"""Gate ablation study with fixed position size=1.

Purpose
-------
Quantify which gates matter when the correlation gate is *not* allowed to change
leverage via sizing.

Method
------
- Load base config (typically configs/trend_based/current.yaml)
- Force sizing.confirm_size_one = sizing.confirm_size_both = 1.0
- Also force seasonality caps to <=1 (optional) by leaving them as-is; they become
  no-ops when size is already 1.
- Run baseline (all gates ON as in config)
- For each gate module, run with that module OFF (set its config block to None)

Outputs a compact table per variant with:
- sum_pnl, worst_maxdd, avg_sharpe
- delta vs baseline

Usage
-----
../.venv/bin/python scripts/ablate_gates_fixed_size.py --config configs/trend_based/current.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Reuse rnd internals for parity + cached period data.
from quantlab.rnd import _prepare_best_trend_inputs, _run_best_trend_periods, _score_candidate


MODULE_KEYS = [
    "htf_confirm",
    "ema_sep",
    "nochop",
    "corr",
    "time_filter",
    "seasonality",
    "churn",
    "risk",
]


def _force_size1(cfg: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c.setdefault("sizing", {})
    c["sizing"]["confirm_size_one"] = 1.0
    c["sizing"]["confirm_size_both"] = 1.0
    return c


def _variant(cfg: dict[str, Any], off_key: str | None) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    if off_key is not None:
        # Module OFF when block is null/missing.
        c[off_key] = None
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dd-cap", type=float, default=20.0)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    base_cfg = _force_size1(base_cfg)

    # Prepare period data once.
    prepared = _prepare_best_trend_inputs(base_cfg)

    def run(cfg: dict[str, Any]):
        period_dfs = _run_best_trend_periods(cfg, prepared=prepared)
        score, _results_df = _score_candidate(
            period_dfs,
            dd_cap_percent=float(args.dd_cap),
            initial_capital=float(args.initial_capital),
        )
        return score

    base_score = run(base_cfg)

    rows = []
    rows.append(
        {
            "variant": "baseline_all_on_size1",
            "ok": base_score.ok,
            "sum_pnl": base_score.sum_pnl,
            "worst_maxdd": base_score.worst_maxdd,
            "avg_sharpe": base_score.avg_sharpe,
            "d_sum_pnl": 0.0,
            "d_avg_sharpe": 0.0,
        }
    )

    for k in MODULE_KEYS:
        cfg_k = _variant(base_cfg, k)
        s = run(cfg_k)
        rows.append(
            {
                "variant": f"off_{k}_size1",
                "ok": s.ok,
                "sum_pnl": s.sum_pnl,
                "worst_maxdd": s.worst_maxdd,
                "avg_sharpe": s.avg_sharpe,
                "d_sum_pnl": s.sum_pnl - base_score.sum_pnl,
                "d_avg_sharpe": s.avg_sharpe - base_score.avg_sharpe,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["variant"]).reset_index(drop=True)

    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))

    # Convenience: sort by impact on avg_sharpe / sum_pnl
    df2 = df[df["variant"] != "baseline_all_on_size1"].copy()
    df2 = df2.sort_values(by=["d_avg_sharpe", "d_sum_pnl"], ascending=[True, True])
    print("\nWorst deltas (most important gates first):")
    print(df2[["variant", "d_avg_sharpe", "d_sum_pnl", "worst_maxdd"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
