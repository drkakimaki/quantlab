#!/usr/bin/env python3
"""Generate an HTML report for a specific trend_based config.

This script exists primarily to:
- regenerate reports outside the WebUI
- allow alternative reports without touching the canonical best_trend outputs

It is intentionally lightweight and uses the same core engine/reporting plumbing
as the WebUI runner.

Examples:
  python quantlab/scripts/report_trend_variants.py \
    --config quantlab/reports/trend_based/decisions/2026-02-25_ewma_holdout20_v1/best.yaml \
    --out-dir quantlab/reports/trend_based \
    --out-name alt_trend_ewma_holdout20_v1.html

Notes:
- For best_trend-style gate pipelines, the config is expected to define:
  symbol/periods/costs/trend/time_filter/pipeline
- The report is equity-only (with a perf table) + an optional trades breakdown.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--out-name", required=True, help="Output HTML filename")
    ap.add_argument(
        "--trades-name",
        default=None,
        help="Optional trades breakdown HTML filename (default: <out-name> with _trades)",
    )
    ap.add_argument("--title", default=None, help="Optional report title")
    ap.add_argument(
        "--record-executions",
        action="store_true",
        help="Record executions in the backtest engine (slower, usually unnecessary for reports)",
    )

    args = ap.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    cfg_path = (workspace / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    out_dir = (workspace / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_path = out_dir / args.out_name

    trades_name = args.trades_name
    if trades_name is None:
        stem = out_path.name
        if stem.lower().endswith(".html"):
            stem = stem[:-5]
        trades_name = stem + "_trades.html"
    trades_path = out_dir / trades_name

    # Ensure local imports work when called as a script.
    os.environ.setdefault("PYTHONPATH", str(workspace))

    # Lazy imports (keep CLI help snappy)
    import pandas as pd

    from quantlab.config.schema import validate_config_dict
    from quantlab.rnd import _prepare_best_trend_inputs, _run_best_trend_periods
    from quantlab.webui.periods import build_periods
    from quantlab.webui.runner import _run_buy_and_hold
    from quantlab.strategies import BacktestConfig
    from quantlab.reporting.generate_bt_report import report_periods_equity_only
    from quantlab.reporting.generate_trades_report import report_periods_trades_html

    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = validate_config_dict(cfg_raw)

    periods_list = build_periods(cfg)

    costs = cfg.get("costs", {}) or {}
    bt_cfg = BacktestConfig(
        fee_per_lot=float(costs.get("fee_per_lot", 0.0) or 0.0),
        spread_per_lot=float(costs.get("spread_per_lot", 0.0) or 0.0),
        record_executions=bool(args.record_executions),
    )

    # Speed: load inputs once (mirrors rnd sweeps).
    prepared = _prepare_best_trend_inputs(cfg)
    period_dfs = _run_best_trend_periods(cfg, record_executions=bool(args.record_executions), prepared=prepared)

    # Baseline (B&H) for vs_BH column.
    baseline_results = _run_buy_and_hold(periods_list, bt_cfg)
    baseline_dfs: dict[str, pd.DataFrame] = {k: df for k, (df, _pnl_abs, _n_exec) in baseline_results.items()}

    # Trade counts for header + table: derive from position diffs (same logic as report helpers).
    n_trades = {k: int((df["position"].diff().abs() > 0).sum()) for k, df in period_dfs.items() if df is not None and len(df)}

    score_exclude = list(((cfg.get("periods", {}) or {}).get("score_exclude") or []) or [])

    title = args.title
    if not title:
        # Include a tiny hyperparam string so the HTML isn't ambiguous.
        trend = cfg.get("trend", {}) or {}
        title = f"Alt Trend — {cfg.get('symbol','XAUUSD')} ({trend.get('ma_kind','ma')}, fast={trend.get('fast')}, slow={trend.get('slow')})"

    # Build report.
    report_periods_equity_only(
        periods=period_dfs,
        baseline_periods=baseline_dfs,
        out_path=out_path,
        title=title,
        initial_capital=1000.0,
        score_exclude=score_exclude,
        n_trades=n_trades,
    )

    # Trades breakdown (best effort; don't fail main report).
    try:
        report_periods_trades_html(
            periods=period_dfs,
            out_path=trades_path,
            title=title + " — trade breakdown",
            score_exclude=score_exclude,
        )
    except Exception:
        pass

    print(f"Generated: {out_path}")
    if trades_path.exists():
        print(f"Generated: {trades_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
