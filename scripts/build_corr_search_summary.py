from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, *, cols: list[str], n: int = 15) -> str:
    view = df[cols].head(n).copy()

    # basic formatting
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    headers = list(view.columns)
    rows = view.values.tolist()

    def esc(x):
        s = str(x)
        return s.replace("|", "\\|")

    out = []
    out.append("| " + " | ".join(map(esc, headers)) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(map(lambda x: esc(x), r)) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=Path, default=Path("reports/trend_based/corr_search/corr_search_results_agg.csv"))
    ap.add_argument("--grid", type=Path, default=Path("reports/trend_based/corr_search/corr_search_grid.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/trend_based/corr_search/corr_search_summary.md"))
    args = ap.parse_args()

    df = pd.read_csv(args.agg)

    cols_show = [
        "config_id",
        "mean_sharpe",
        "worst_max_drawdown",
        "sum_pnl_usd",
        "sharpe_2023_2025",
        "sharpe_2026",
        "corr_logic",
        "corr_window_xag",
        "corr_min_abs_xag",
        "corr_flip_lookback_xag",
        "corr_max_flips_xag",
        "corr_window_eur",
        "corr_min_abs_eur",
        "corr_flip_lookback_eur",
        "corr_max_flips_eur",
    ]

    by_mean_sharpe = df.sort_values(["mean_sharpe", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    by_worst_dd = df.sort_values(["worst_max_drawdown", "mean_sharpe"], ascending=[False, False]).reset_index(drop=True)
    by_2325 = df.sort_values(["sharpe_2023_2025", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)
    by_2026 = df.sort_values(["sharpe_2026", "worst_max_drawdown"], ascending=[False, False]).reset_index(drop=True)

    md = []
    md.append("# Corr-stability hyperparam search (best_trend)\n")
    md.append(f"Agg CSV: `{args.agg}`\n")
    md.append(f"Grid spec: `{args.grid}`\n")

    md.append("## Top by mean Sharpe (across all 3 periods)\n")
    md.append(md_table(by_mean_sharpe, cols=cols_show, n=15))

    md.append("\n## Top by worst MaxDD (least-negative drawdown across periods)\n")
    md.append(md_table(by_worst_dd, cols=cols_show, n=15))

    md.append("\n## Top by Sharpe (2023-2025)\n")
    md.append(md_table(by_2325, cols=cols_show, n=15))

    md.append("\n## Top by Sharpe (2026-01-01..2026-02-13)\n")
    md.append(md_table(by_2026, cols=cols_show, n=15))

    args.out.write_text("\n".join(md) + "\n")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
