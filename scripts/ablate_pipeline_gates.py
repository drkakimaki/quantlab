"""Ablate each gate in the *pipeline* of a best_trend config.

Unlike scripts/ablate_gates_fixed_size.py (legacy block-based config), this script
works with the canonical `pipeline:` list.

Method
------
- Load config YAML
- Run baseline (all gates as listed)
- For each pipeline gate i:
    - remove that gate from the pipeline
    - run backtest across the configured periods
    - score using quantlab.rnd._score_candidate

Outputs
-------
- One row per variant with TOTAL score + per-period stats.

Usage
-----
../.venv/bin/python scripts/ablate_pipeline_gates.py --config configs/trend_based/current.yaml --dd-cap 20
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantlab.rnd import _prepare_best_trend_inputs, _run_best_trend_periods, _score_candidate


def _pipeline(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    p = cfg.get("pipeline")
    if not isinstance(p, list) or not p:
        raise ValueError("Config must contain a non-empty `pipeline:` list")
    return [dict(x) for x in p]


def _gate_name(spec: dict[str, Any]) -> str:
    g = spec.get("gate") or spec.get("name")
    return str(g).strip()


def _variant_remove_gate(cfg: dict[str, Any], *, remove_idx: int) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    p = _pipeline(c)
    if remove_idx < 0 or remove_idx >= len(p):
        raise IndexError(remove_idx)
    del p[remove_idx]
    c["pipeline"] = p
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dd-cap", type=float, default=20.0)
    ap.add_argument("--initial-capital", type=float, default=1000.0)
    ap.add_argument(
        "--include-periods",
        action="store_true",
        help="Include per-period rows in the output (not just TOTAL)",
    )
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    p = _pipeline(base_cfg)

    prepared = _prepare_best_trend_inputs(base_cfg)

    def run(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
        period_dfs = _run_best_trend_periods(cfg, prepared=prepared)
        score_exclude = list(((cfg.get("periods", {}) or {}).get("score_exclude") or []) or [])
        score, results_df = _score_candidate(
            period_dfs,
            dd_cap_percent=float(args.dd_cap),
            initial_capital=float(args.initial_capital),
            score_exclude=score_exclude,
        )
        return score.__dict__, results_df

    base_score, base_results = run(base_cfg)

    rows: list[dict[str, Any]] = []

    def add_rows(variant: str, gate_off: str | None, score: dict[str, Any], results_df: pd.DataFrame):
        # Pull TOTAL row
        total = results_df[results_df["period"] == "TOTAL"].iloc[0].to_dict()
        row = {
            "variant": variant,
            "gate_off": gate_off or "",
            "ok": bool(score["ok"]),
            "sum_pnl": float(score["sum_pnl"]),
            "worst_maxdd": float(score["worst_maxdd"]),
            "avg_sharpe": float(score["avg_sharpe"]),
            "d_sum_pnl": float(score["sum_pnl"]) - float(base_score["sum_pnl"]),
            "d_avg_sharpe": float(score["avg_sharpe"]) - float(base_score["avg_sharpe"]),
            "TOTAL_pnl": float(total.get("pnl_percent")),
            "TOTAL_maxdd": float(total.get("maxdd_percent")),
            "TOTAL_sharpe": float(total.get("sharpe")),
        }
        rows.append(row)

        if args.include_periods:
            for _, r in results_df[results_df["period"] != "TOTAL"].iterrows():
                rows.append(
                    {
                        "variant": variant,
                        "gate_off": gate_off or "",
                        "period": r["period"],
                        "pnl_percent": float(r["pnl_percent"]),
                        "maxdd_percent": float(r["maxdd_percent"]),
                        "sharpe": float(r["sharpe"]),
                        "scored": bool(r["scored"]),
                    }
                )

    add_rows("baseline", None, base_score, base_results)

    for i, spec in enumerate(p):
        off = _gate_name(spec)
        cfg_i = _variant_remove_gate(base_cfg, remove_idx=i)
        s_i, res_i = run(cfg_i)
        add_rows(f"off_{off}", off, s_i, res_i)

    df = pd.DataFrame(rows)

    # Summary table
    summary = df[df["variant"].isin(["baseline"] + [f"off_{_gate_name(x)}" for x in p])].copy()
    summary = summary[[
        "variant",
        "gate_off",
        "ok",
        "sum_pnl",
        "worst_maxdd",
        "avg_sharpe",
        "d_sum_pnl",
        "d_avg_sharpe",
    ]]

    pd.set_option("display.max_columns", 50)
    print(summary.to_string(index=False))

    worst = summary[summary["variant"] != "baseline"].copy()
    worst = worst.sort_values(by=["d_avg_sharpe", "d_sum_pnl"], ascending=[True, True])
    print("\nWorst deltas (most important gates first):")
    print(worst[["variant", "d_avg_sharpe", "d_sum_pnl", "worst_maxdd"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
