from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .engine.metrics import sharpe


# We intentionally reuse WebUI modules for period building + data loading + FOMC masking.
#
# Why this dependency direction is OK here:
# - Parity is the priority: the CLI R&D loop (rnd.py) must match what the WebUI runs,
#   otherwise we end up debugging "why don't the numbers match?" across pipelines.
# - If this ever becomes painful (extra deps / circular imports), extract the shared
#   functions into a neutral module (e.g. quantlab/inputs.py) and have both import it.
from .webui.periods import build_periods
from .webui.runner import load_period_data, load_fomc_mask  # type: ignore
from .webui.config import WORKSPACE
from .strategies.trend_following import TrendStrategyWithGates
from .strategies.base import BacktestConfig


@dataclass(frozen=True)
class CandidateScore:
    ok: bool
    sum_pnl: float
    worst_maxdd: float
    avg_sharpe: float
    # If constraint violated, how far over the cap (positive number)
    maxdd_violation: float


def _now_slug() -> str:
    # Europe/Berlin local time for folder naming
    now = dt.datetime.now()
    return now.strftime("%Y-%m-%d_%H%M%S")


def _set_in(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = cfg
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _get_in(cfg: dict[str, Any], dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    cur: Any = cfg
    for k in parts:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _max_drawdown_percent(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) * 100.0


def _period_stats(bt: pd.DataFrame, *, initial_capital: float, freq: str = "5MIN") -> tuple[float, float, float]:
    """Return (pnl_percent, maxdd_percent, sharpe)."""
    eq = bt["equity"].astype(float)
    r = bt["returns_net"].astype(float)

    cap0 = float(initial_capital)
    if cap0 <= 0:
        cap0 = float(eq.iloc[0]) if len(eq) else 1.0

    pnl = (float(eq.iloc[-1]) / cap0 - 1.0) * 100.0
    maxdd = _max_drawdown_percent(eq)
    s = float(sharpe(r, freq=freq))
    return pnl, maxdd, s


def _score_candidate(
    period_dfs: dict[str, pd.DataFrame],
    *,
    dd_cap_percent: float,
    initial_capital: float = 1000.0,
    freq: str = "5MIN",
) -> tuple[CandidateScore, pd.DataFrame]:
    rows = []
    sum_pnl = 0.0
    maxdds = []
    sharpes = []

    for name, bt in period_dfs.items():
        if bt is None or len(bt) == 0:
            pnl, maxdd, s = float("nan"), float("nan"), float("nan")
        else:
            pnl, maxdd, s = _period_stats(bt, initial_capital=initial_capital, freq=freq)

        rows.append({"period": name, "pnl_percent": pnl, "maxdd_percent": maxdd, "sharpe": s})

        if np.isfinite(pnl):
            sum_pnl += float(pnl)
        if np.isfinite(maxdd):
            maxdds.append(float(maxdd))
        if np.isfinite(s):
            sharpes.append(float(s))

    worst_maxdd = float(min(maxdds)) if maxdds else float("nan")
    avg_sh = float(np.mean(sharpes)) if sharpes else float("nan")

    # MaxDD is negative (e.g. -22). Cap is positive percent (e.g. 20).
    ok = (np.isfinite(worst_maxdd) and worst_maxdd >= -abs(dd_cap_percent))
    violation = 0.0
    if not ok and np.isfinite(worst_maxdd):
        violation = abs(worst_maxdd) - abs(dd_cap_percent)

    score = CandidateScore(
        ok=bool(ok),
        sum_pnl=float(sum_pnl),
        worst_maxdd=float(worst_maxdd),
        avg_sharpe=float(avg_sh),
        maxdd_violation=float(violation),
    )

    df = pd.DataFrame(rows)
    # Totals row for convenience
    df_tot = pd.DataFrame(
        [
            {
                "period": "TOTAL",
                "pnl_percent": score.sum_pnl,
                "maxdd_percent": score.worst_maxdd,
                "sharpe": score.avg_sharpe,
            }
        ]
    )
    df = pd.concat([df, df_tot], ignore_index=True)
    return score, df


@dataclass(frozen=True)
class PreparedPeriod:
    name: str
    start: dt.date
    end: dt.date
    prices: pd.Series | pd.DataFrame
    context: dict[str, Any]
    allow_mask: Any


def _prepare_best_trend_inputs(cfg: dict[str, Any]) -> list[PreparedPeriod]:
    """Load all data once per period for speed during sweeps."""
    periods = build_periods(cfg)

    symbol = cfg.get("symbol", "XAUUSD")
    corr_symbol = cfg.get("corr_symbol", "XAGUSD")
    corr2_symbol = cfg.get("corr2_symbol", "EURUSD")

    fomc_cfg = (cfg.get("time_filter", {}) or {}).get("fomc", {}) or {}
    days_csv = (cfg.get("time_filter", {}) or {}).get("fomc", {}).get(
        "days_csv", "quantlab/data/econ_calendar/fomc_decision_days.csv"
    )

    # Match WebUI semantics: paths in YAML are resolved relative to WORKSPACE.
    fomc_path = WORKSPACE / str(days_csv)

    out: list[PreparedPeriod] = []
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
        allow_mask = load_fomc_mask(data["prices"].index, start, end, fomc_path, fomc_cfg)
        context = {
            "bars_15m": data.get("bars_15m"),
            "prices_xag": data.get("prices_xag"),
            "prices_eur": data.get("prices_eur"),
        }
        out.append(
            PreparedPeriod(
                name=name,
                start=start,
                end=end,
                prices=data["prices"],
                context=context,
                allow_mask=allow_mask,
            )
        )
    return out


def _run_best_trend_periods(
    cfg: dict[str, Any],
    *,
    record_executions: bool = False,
    prepared: list[PreparedPeriod] | None = None,
) -> dict[str, pd.DataFrame]:
    costs = cfg.get("costs", {}) or {}
    bt_cfg = BacktestConfig(
        fee_per_lot=float(costs.get("fee_per_lot", 0.0) or 0.0),
        spread_per_lot=float(costs.get("spread_per_lot", 0.0) or 0.0),
        record_executions=bool(record_executions),
    )

    if prepared is None:
        prepared = _prepare_best_trend_inputs(cfg)

    out: dict[str, pd.DataFrame] = {}
    for p in prepared:
        strat = TrendStrategyWithGates.from_config(cfg, allow_mask=p.allow_mask)
        res = strat.run_backtest(p.prices, context=p.context, config=bt_cfg)
        out[p.name] = res.df

    return out


def _decision_dir(slug: str) -> Path:
    root = Path(__file__).resolve().parents[0] / "reports" / "trend_based" / "decisions"
    root.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now().strftime("%Y-%m-%d")
    folder = root / f"{today}_{slug}"
    return folder


def _write_decision_bundle(
    *,
    slug: str,
    cfg: dict[str, Any],
    topk: pd.DataFrame | None,
    best_results: pd.DataFrame,
    score: CandidateScore,
    notes: dict[str, Any],
) -> Path:
    d = _decision_dir(slug)
    d.mkdir(parents=True, exist_ok=True)

    (d / "best.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    best_results.to_csv(d / "results.csv", index=False)

    if topk is not None:
        topk.to_csv(d / "top_k.csv", index=False)

    (d / "notes.json").write_text(json.dumps(notes, indent=2, sort_keys=True), encoding="utf-8")

    decision_md = (
        f"# Decision: {slug}\n\n"
        f"Timestamp: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        f"Objective: maximize sum(period PnL%) subject to worst MaxDD <= {notes['dd_cap_percent']}%\n\n"
        f"## Best score\n\n"
        f"- OK under cap: {score.ok}\n"
        f"- Sum PnL%: {score.sum_pnl:.2f}\n"
        f"- Worst MaxDD%: {score.worst_maxdd:.2f}\n"
        f"- Avg Sharpe: {score.avg_sharpe:.2f}\n\n"
        f"## Files\n\n"
        f"- best.yaml\n"
        f"- results.csv\n"
        f"- top_k.csv (if sweep)\n"
        f"- notes.json\n"
    )
    (d / "DECISION.md").write_text(decision_md, encoding="utf-8")

    (d / "raw").mkdir(exist_ok=True)
    return d


def cmd_run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    if args.mode in {"three_block", "yearly"}:
        cfg = dict(cfg)
        p = dict(cfg.get("periods", {}) or {})
        p["mode"] = args.mode
        cfg["periods"] = p

    period_dfs = _run_best_trend_periods(cfg)
    score, results_df = _score_candidate(period_dfs, dd_cap_percent=float(args.dd_cap), initial_capital=float(args.initial_capital))

    print(results_df.to_string(index=False))
    print()
    print(f"OK={score.ok}  sum_pnl={score.sum_pnl:.2f}%  worst_maxdd={score.worst_maxdd:.2f}%  avg_sharpe={score.avg_sharpe:.2f}")

    if args.decision_slug:
        notes = {
            "kind": "run",
            "config_path": str(args.config),
            "mode": (cfg.get("periods", {}) or {}).get("mode"),
            "dd_cap_percent": float(args.dd_cap),
            "initial_capital": float(args.initial_capital),
        }
        d = _write_decision_bundle(
            slug=args.decision_slug,
            cfg=cfg,
            topk=None,
            best_results=results_df,
            score=score,
            notes=notes,
        )
        print(f"\nWrote decision bundle: {d}")

    return 0


def _better(a: CandidateScore | None, b: CandidateScore) -> bool:
    if a is None:
        return True
    # Feasible beats infeasible
    if a.ok != b.ok:
        return b.ok
    # If both infeasible: smaller violation wins
    if not b.ok:
        if b.maxdd_violation != a.maxdd_violation:
            return b.maxdd_violation < a.maxdd_violation
    # Maximize PnL
    if b.sum_pnl != a.sum_pnl:
        return b.sum_pnl > a.sum_pnl
    # Tiebreak: higher avg sharpe
    return b.avg_sharpe > a.avg_sharpe


def cmd_sweep(args: argparse.Namespace) -> int:
    sweep = yaml.safe_load(Path(args.sweep).read_text()) or {}

    base_path = sweep.get("base")
    if not base_path:
        raise ValueError("sweeps.yaml must include 'base' pointing to a config yaml")

    base_cfg = yaml.safe_load(Path(base_path).read_text()) or {}

    mode = sweep.get("mode", "")
    if mode in {"three_block", "yearly"}:
        base_cfg = dict(base_cfg)
        p = dict(base_cfg.get("periods", {}) or {})
        p["mode"] = mode
        base_cfg["periods"] = p

    dd_cap = float(sweep.get("dd_cap_percent", args.dd_cap))
    top_k = int(sweep.get("top_k", 10))
    initial_capital = float(sweep.get("initial_capital", 1000.0))

    kind = (sweep.get("kind") or "grid").strip().lower()

    grid: dict[str, list[Any]] = sweep.get("grid", {}) or {}
    keys = list(grid.keys())
    if not keys:
        raise ValueError("sweeps.yaml grid is empty")

    # Build candidate combos
    combos: list[tuple[Any, ...]]
    if kind == "grid":
        from itertools import product

        values_lists = [grid[k] for k in keys]
        combos = list(product(*values_lists))
    elif kind in {"sample", "random"}:
        n = int(sweep.get("n_candidates", 0) or 0)
        if n <= 0:
            raise ValueError("For kind=sample, set n_candidates > 0")

        import random

        seen: set[tuple[Any, ...]] = set()
        combos = []
        attempts = 0
        max_attempts = max(10_000, n * 50)
        while len(combos) < n and attempts < max_attempts:
            attempts += 1
            combo = tuple(random.choice(grid[k]) for k in keys)
            if combo in seen:
                continue
            seen.add(combo)
            combos.append(combo)

        if len(combos) < n:
            print(f"WARN: only generated {len(combos)}/{n} unique combos (grid may be too small)")
    else:
        raise ValueError(f"Unknown sweep kind: {kind!r} (use grid|sample)")

    rows = []
    best_cfg = None
    best_score: CandidateScore | None = None
    best_results_df: pd.DataFrame | None = None

    print("Preparing period data (cached for sweep)...", flush=True)
    prepared = _prepare_best_trend_inputs(base_cfg)
    print(f"Prepared {len(prepared)} periods. Starting {len(combos)} candidates...", flush=True)

    for idx, combo in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo, strict=True):
            _set_in(cfg, k, v)

        period_dfs = _run_best_trend_periods(cfg, prepared=prepared)
        score, results_df = _score_candidate(period_dfs, dd_cap_percent=dd_cap, initial_capital=initial_capital)

        row = {"i": idx}
        for k, v in zip(keys, combo, strict=True):
            row[k] = v
        row.update(
            {
                "ok": score.ok,
                "sum_pnl": score.sum_pnl,
                "worst_maxdd": score.worst_maxdd,
                "avg_sharpe": score.avg_sharpe,
                "maxdd_violation": score.maxdd_violation,
            }
        )
        rows.append(row)

        if _better(best_score, score):
            best_score = score
            best_cfg = cfg
            best_results_df = results_df

        if args.progress and (idx % int(args.progress) == 0):
            print(f"{idx}/{len(combos)} done...", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["ok", "maxdd_violation", "sum_pnl"], ascending=[False, True, False])

    topk_df = df.head(top_k).copy()
    print(topk_df.to_string(index=False))

    if args.decision_slug:
        assert best_cfg is not None and best_score is not None and best_results_df is not None
        notes = {
            "kind": "sweep",
            "sweep_path": str(args.sweep),
            "base": str(base_path),
            "mode": (best_cfg.get("periods", {}) or {}).get("mode"),
            "dd_cap_percent": dd_cap,
            "objective": "sum_pnl subject_to worst_maxdd_cap",
            "sweep_kind": kind,
            "n_candidates": len(combos),
            "grid": {k: grid[k] for k in keys},
        }
        d = _write_decision_bundle(
            slug=args.decision_slug,
            cfg=best_cfg,
            topk=topk_df,
            best_results=best_results_df,
            score=best_score,
            notes=notes,
        )
        print(f"\nWrote decision bundle: {d}")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantlab.rnd", description="Low-token R&D runner for best_trend")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run", help="Run best_trend once and print summary")
    ap_run.add_argument("--config", required=True, help="Path to YAML config")
    ap_run.add_argument("--mode", default="", help="Override periods.mode (three_block|yearly)")
    ap_run.add_argument("--dd-cap", type=float, default=20.0, help="MaxDD cap in percent (default 20)")
    ap_run.add_argument("--initial-capital", type=float, default=1000.0)
    ap_run.add_argument("--decision-slug", default="", help="If set, write decision bundle under reports/trend_based/decisions")
    ap_run.set_defaults(func=cmd_run)

    ap_sw = sub.add_parser("sweep", help="Run a grid sweep from sweeps.yaml")
    ap_sw.add_argument("--sweep", required=True, help="Path to sweeps.yaml")
    ap_sw.add_argument("--dd-cap", type=float, default=20.0, help="Default MaxDD cap if not set in YAML")
    ap_sw.add_argument("--decision-slug", default="", help="If set, write decision bundle")
    ap_sw.add_argument("--progress", default="", help="Print progress every N combos")
    ap_sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
