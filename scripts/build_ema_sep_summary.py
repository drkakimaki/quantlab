from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=Path, default=Path("reports/trend_based/ema_sep_search/ema_sep_results_agg.csv"))
    ap.add_argument("--grid", type=Path, default=Path("reports/trend_based/ema_sep_search/ema_sep_grid.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/trend_based/ema_sep_search/ema_sep_summary.md"))
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    df_agg = pd.read_csv(args.agg)
    meta = json.loads(args.grid.read_text()) if args.grid.exists() else {}

    df_rank = df_agg.sort_values(["mean_sharpe", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)

    cols_show = [
        "config_id",
        "mean_sharpe",
        "worst_max_drawdown",
        "sum_pnl_usd",
        "sharpe_2023_2025",
        "sharpe_2026",
        "ema_fast",
        "ema_slow",
        "atr_n",
        "sep_k",
    ]

    def tbl(df: pd.DataFrame, n: int = 15) -> str:
        return df[cols_show].head(n).to_markdown(index=False)

    md: list[str] = []
    md.append("# EMA separation filter hyperparam search (best_trend)\n")
    md.append(f"Generated: {meta.get('created_utc', dt.datetime.now(dt.UTC).isoformat().replace('+00:00','Z'))}\n")
    md.append(f"Agg CSV: `{args.agg}`\n")
    md.append(f"Grid spec: `{args.grid}`\n")

    md.append("\n## Top by mean Sharpe (across all 3 periods)\n")
    md.append(tbl(df_rank))

    by_worst_dd = df_agg.sort_values(["worst_max_drawdown", "mean_sharpe"], ascending=[False, False]).reset_index(drop=True)
    md.append("\n## Top by worst MaxDD (least-negative drawdown across periods)\n")
    md.append(tbl(by_worst_dd))

    by_2325 = df_agg.sort_values(["sharpe_2023_2025", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    md.append("\n## Top by Sharpe (2023-2025)\n")
    md.append(tbl(by_2325))

    by_2026 = df_agg.sort_values(["sharpe_2026", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    md.append("\n## Top by Sharpe (2026)\n")
    md.append(tbl(by_2026))

    # top htmls
    out_dir = args.out.parent
    htmls = sorted(out_dir.glob("top_*.html"))
    md.append("\n## HTML reports generated (top-k by mean Sharpe)\n")
    for p in htmls[: args.top_k]:
        md.append(f"- {p}")

    args.out.write_text("\n".join(md) + "\n")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
