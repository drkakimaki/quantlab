from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.metrics import (
    sharpe,
    sharpe_bootstrap,
    exposure_from_position,
    avg_win_loss_from_position,
    profit_factor_from_position,
    win_rate_from_position,
    n_trades_from_position,
)

from .report_common import PeriodRow, safe_float, fmt_ts


def report_robustness(
    *,
    periods: dict[str, pd.DataFrame],
    out_path: str | Path,
    title: str,
    initial_capital: float | dict[str, float],
    returns_col: str = "returns_net",
    equity_col: str = "equity",
    baseline_periods: dict[str, pd.DataFrame] | None = None,
    n_trades: dict[str, int] | None = None,
    win_rate: dict[str, float] | None = None,
    score_exclude: list[str] | set[str] | None = None,
    table_title: str = "Yearly breakdown",
) -> Path:
    """Single-file HTML report for multiple periods (TABLE ONLY).

    This is a chartless variant intended for yearly breakdown / robustness-style reports.
    No matplotlib rendering, no chart CSS, no chart JS.

    Note: `table_title` controls the h2 label above the main table.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[PeriodRow] = []

    winners_rows: list[dict[str, object]] = []
    losers_rows: list[dict[str, object]] = []

    def _daily_pnl(bt: pd.DataFrame) -> pd.Series:
        """Daily PnL series derived from equity (end-of-day equity diffs)."""
        eq = bt[equity_col].astype(float).copy()
        eq.index = pd.to_datetime(eq.index)
        eq = eq.sort_index()
        # Use UTC days implicitly (index is expected UTC already).
        eq_d = eq.resample('1D').last().dropna()
        return eq_d.diff().dropna()


    for name, bt in periods.items():
        if bt is None or len(bt) == 0:
            rows.append(
                PeriodRow(
                    period=name,
                    pnl=float("nan"),
                    max_drawdown=float("nan"),
                    sharpe=float("nan"),
                    sharpe_ci_lo=float("nan"),
                    sharpe_ci_hi=float("nan"),
                    vs_bh=float("nan"),
                    win_rate=float("nan"),
                    profit_factor=float("nan"),
                    avg_win=float("nan"),
                    avg_loss=float("nan"),
                    exposure=float("nan"),
                    n_trades=int((n_trades or {}).get(name, 0)),
                    start="",
                    end="",
                )
            )
            continue

        if returns_col not in bt.columns or equity_col not in bt.columns:
            raise KeyError(f"Period {name!r} missing required columns")

        bt = bt.copy()
        bt.index = pd.to_datetime(bt.index)
        bt = bt.sort_index()

        eq = bt[equity_col].astype(float)

        cap0 = float(initial_capital[name] if isinstance(initial_capital, dict) else initial_capital)
        if cap0 <= 0:
            cap0 = float(eq.iloc[0]) if len(eq) else 1.0

        # Daily PnL breakdown (robustness surface)
        pnl_d = _daily_pnl(bt)
        wins_d = pnl_d[pnl_d > 0.0]
        losses_d = pnl_d[pnl_d < 0.0]

        sum_wins = float(wins_d.sum()) if len(wins_d) else 0.0
        sum_losses_abs = float((-losses_d).sum()) if len(losses_d) else 0.0

        denom_wins = max(1e-12, sum_wins)
        denom_losses = max(1e-12, sum_losses_abs)

        top5_wins = float(wins_d.sort_values(ascending=False).head(5).sum()) if len(wins_d) else 0.0
        worst5_losses_abs = float((-losses_d.sort_values().head(5)).sum()) if len(losses_d) else 0.0

        winners_rows.append(
            {
                "period": name,
                "p90_win_pct": float((wins_d / cap0).quantile(0.90)) if (len(wins_d) and cap0 > 0) else float("nan"),
                "n_win_days": int(len(wins_d)),
                "top5_win_share": float(top5_wins / denom_wins) if sum_wins > 0 else float("nan"),
            }
        )

        losers_rows.append(
            {
                "period": name,
                "p10_loss_pct": float((losses_d / cap0).quantile(0.10)) if (len(losses_d) and cap0 > 0) else float("nan"),
                "n_loss_days": int(len(losses_d)),
                "worst5_loss_share": float(worst5_losses_abs / denom_losses) if sum_losses_abs > 0 else float("nan"),
            }
        )

        pnl = (float(eq.iloc[-1]) / cap0 - 1.0) * 100.0

        vs_bh = float("nan")
        if baseline_periods is not None and name in baseline_periods:
            bh = baseline_periods.get(name)
            if bh is not None and len(bh) and equity_col in bh.columns:
                bh_eq = bh[equity_col].astype(float)
                bh_cap0 = cap0
                if bh_cap0 <= 0:
                    bh_cap0 = float(bh_eq.iloc[0]) if len(bh_eq) else 1.0
                bh_pnl = (float(bh_eq.iloc[-1]) / bh_cap0 - 1.0) * 100.0
                vs_bh = pnl - bh_pnl

        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        max_dd = float(dd.min()) * 100.0

        s = float(sharpe(eq))
        s_ci = sharpe_bootstrap(eq)
        s_lo = float(s_ci.get("lo", float("nan")))
        s_hi = float(s_ci.get("hi", float("nan")))

        if n_trades is None:
            trades = n_trades_from_position(bt, pos_col="position")
        else:
            trades = int(n_trades.get(name, 0))

        wr = float(win_rate.get(name)) if (win_rate is not None and name in win_rate) else win_rate_from_position(bt)
        pf = profit_factor_from_position(bt, pos_col="position", returns_col=returns_col)
        avg_win, avg_loss = avg_win_loss_from_position(bt, pos_col="position", returns_col=returns_col)

        exposure = float(exposure_from_position(bt, pos_col="position"))

        rows.append(
            PeriodRow(
                period=name,
                pnl=pnl,
                max_drawdown=max_dd,
                sharpe=s,
                sharpe_ci_lo=s_lo,
                sharpe_ci_hi=s_hi,
                vs_bh=vs_bh,
                win_rate=wr,
                profit_factor=pf,
                avg_win=avg_win,
                avg_loss=avg_loss,
                exposure=exposure,
                n_trades=trades,
                start=fmt_ts(bt.index.min()),
                end=fmt_ts(bt.index.max()),
            )
        )

    def pct(x: float) -> str:
        x = safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:,.2f}%"

    def pct_signed(x: float) -> str:
        x = safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:+,.2f}%"

    def num(x: float) -> str:
        x = safe_float(x)
        if np.isnan(x):
            return "nan"
        return f"{x:,.2f}"

    def pf_fmt(x: float) -> str:
        x = safe_float(x)
        if np.isnan(x):
            return "nan"
        if np.isinf(x):
            return "inf"
        return f"{x:,.2f}"

    def money(x: float) -> str:
        x = safe_float(x)
        if np.isnan(x):
            return 'nan'
        if np.isinf(x):
            return 'inf' if x > 0 else '-inf'
        return f"{x:,.2f}"

    def pct_small(x: float) -> str:
        x = safe_float(x)
        if np.isnan(x):
            return 'nan'
        if np.isinf(x):
            return 'inf' if x > 0 else '-inf'
        return f"{x*100:,.2f}%"


    win_tr = []
    for r in winners_rows:
        win_tr.append(
            '<tr>'
            f"<td class='period'>{r['period']}</td>"
            f"<td class='num mono'>{pct_small(r['p90_win_pct'])}</td>"
            f"<td class='num mono'>{r['n_win_days']}</td>"
            f"<td class='num mono'>{pct_small(r['top5_win_share'])}</td>"
            '</tr>'
        )

    loss_tr = []
    for r in losers_rows:
        loss_tr.append(
            '<tr>'
            f"<td class='period'>{r['period']}</td>"
                        f"<td class='num mono'>{pct_small(r['p10_loss_pct'])}</td>"
            f"<td class='num mono'>{r['n_loss_days']}</td>"
            f"<td class='num mono'>{pct_small(r['worst5_loss_share'])}</td>"
            '</tr>'
        )

    tr = []
    for r in rows:
        tr.append(
            "<tr>"
            f"<td class='period'>{r.period}</td>"
            f"<td class='num mono'>{pct(r.pnl)}</td>"
            f"<td class='num mono'>{pct_signed(r.vs_bh)}</td>"
            f"<td class='num mono'>{pct(r.max_drawdown)}</td>"
            f"<td class='num mono'>{num(r.sharpe)}</td>"
            f"<td class='num mono'>{num(r.sharpe_ci_lo)}/{num(r.sharpe_ci_hi)}</td>"
            f"<td class='num mono'>{pct(r.win_rate)}</td>"
            f"<td class='num mono'>{pf_fmt(r.profit_factor)}</td>"
            f"<td class='num mono'>{pct(r.avg_win)}</td>"
            f"<td class='num mono'>{pct(r.avg_loss)}</td>"
            f"<td class='num mono'>{pct(r.exposure)}</td>"
            f"<td class='num mono'>{r.n_trades}</td>"
            "</tr>"
        )

    exclude = set(score_exclude or [])
    scored_rows = [r for r in rows if r.period not in exclude]
    total_trades = int(sum((r.n_trades or 0) for r in scored_rows))
    sum_pnl = float(sum((r.pnl for r in scored_rows if np.isfinite(r.pnl)), 0.0))
    worst_dd = float(min((r.max_drawdown for r in scored_rows if np.isfinite(r.max_drawdown)), default=float("nan")))
    sh_list = [r.sharpe for r in scored_rows if np.isfinite(r.sharpe)]
    avg_sharpe = float(np.mean(sh_list)) if sh_list else float("nan")

    holdout_note = ""
    if exclude:
        holdout_note = f"Holdout excluded from header stats: {sorted(exclude)}"

    archetype = title.split("(", 1)[0].strip() if title else "Report"
    parts = title.split(" + ") if title else []
    base_line = parts[0] if parts else title

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>{archetype}</title>
  <style>
    :root {{
      --fg:#e5e7eb;
      --muted:#9ca3af;
      --bg:#0b1220;
      --border:#223047;
      --card:#111a2b;
      --card2:#0f172a;
      --shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      color:var(--fg);
      background:var(--bg);
      margin: 0;
      line-height: 1.45;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 26px 18px 34px; }}
    .header {{
      background: radial-gradient(900px 420px at 20% -20%, rgba(96,165,250,0.18), rgba(0,0,0,0) 55%),
                  linear-gradient(180deg, rgba(17,26,43,0.98), rgba(15,23,42,0.92));
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 18px 18px 16px;
      margin-bottom: 14px;
    }}
    h1 {{ margin: 0 0 6px 0; font-size: 30px; letter-spacing: -0.6px; }}
    .sub {{ color:var(--muted); font-size: 12px; }}
    .subtitle {{ color:var(--muted); font-size: 13px; margin-top: 2px; }}
    .section {{ margin: 14px 0; }}

    .summary-strip {{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 10px; }}
    .pill {{ background: var(--card2); border:1px solid var(--border); border-radius: 999px; padding: 8px 10px; font-size: 12px; color: var(--muted); }}
    .pill b {{ color: var(--fg); font-weight: 700; font-variant-numeric: tabular-nums; }}

    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow); overflow: hidden; }}
    .box {{ padding: 14px 14px; }}
    .box h2 {{ margin: 0 0 10px 0; font-size: 14px; color: var(--fg); letter-spacing: -0.2px; }}

    table {{ border-collapse: separate; border-spacing: 0; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 12px; text-align: left; }}
    th.num {{ text-align: right; }}
    th {{
      font-size: 12px;
      color: var(--muted);
      background: var(--card2);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tbody td {{ background: transparent; }}
    tbody tr:hover td {{ background: rgba(96,165,250,0.10); }}

    td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .mono {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
    td.period {{ font-weight: 600; }}

    .footer {{ margin-top: 10px; }}
  </style>
</head>
<body>
  <div class='page'>
    <div class='header'>
      <h1>{archetype}</h1>
      <div class='subtitle'>{base_line}</div>
      <div class='summary-strip'>
        <div class='pill'>Sum PnL: <b>{pct(sum_pnl)}</b></div>
        <div class='pill'># Trades: <b>{total_trades:,}</b></div>
        <div class='pill'>Avg Sharpe: <b>{num(avg_sharpe)}</b></div>
        <div class='pill'>Worst MaxDD: <b>{pct(worst_dd)}</b></div>
      </div>
      <div class='sub'>{holdout_note}</div>
    </div>

    <div class='section'>
      <div class='card'>
        <div class='box'>
          <h2>{table_title}</h2>
          <table id='perf-table'>
            <thead>
              <tr>
                <th data-sort='text'>Period</th>
                <th class='num' data-sort='num'>PnL</th>
                <th class='num' data-sort='num'>vs B&H</th>
                <th class='num' data-sort='num'>Max DD</th>
                <th class='num' data-sort='num'>Sharpe</th>
                <th class='num' data-sort='text'>Sharpe CI</th>
                <th class='num' data-sort='num'>Win Rate</th>
                <th class='num' data-sort='num'>Profit Factor</th>
                <th class='num' data-sort='num'>Avg Win</th>
                <th class='num' data-sort='num'>Avg Loss</th>
                <th class='num' data-sort='num'>Exposure%</th>
                <th class='num' data-sort='num'># Trades</th>
              </tr>
            </thead>
            <tbody>
              {''.join(tr)}
            </tbody>
          </table>
          <div class='sub' style='padding:10px 12px;'>Tip: click headers to sort.</div>
        </div>
      </div>
    </div>

    <script>
      // Sortable table
      (function() {{
        const table = document.getElementById('perf-table');
        if (!table) return;
        const tbody = table.querySelector('tbody');
        const getCell = (tr, idx) => tr.children[idx].innerText.trim();
        const parseNum = (s) => {{
          if (!s) return Number.NEGATIVE_INFINITY;
          const x = s.replace(/%/g,'').replace(/,/g,'');
          const v = parseFloat(x);
          return Number.isFinite(v) ? v : Number.NEGATIVE_INFINITY;
        }};

        let sortState = {{ idx: -1, asc: false }};

        table.querySelectorAll('thead th').forEach((th, idx) => {{
          th.style.cursor = 'pointer';
          th.addEventListener('click', () => {{
            const type = th.getAttribute('data-sort') || 'text';
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const asc = (sortState.idx === idx) ? !sortState.asc : false;
            sortState = {{ idx, asc }};

            rows.sort((a,b) => {{
              const A = getCell(a, idx);
              const B = getCell(b, idx);
              if (type === 'num') {{
                return (parseNum(A) - parseNum(B)) * (asc ? 1 : -1);
              }}
              return A.localeCompare(B) * (asc ? 1 : -1);
            }});

            rows.forEach(r => tbody.appendChild(r));
          }});
        }});
      }})();
    </script>



    <div class='section'>
      <div class='card'>
        <div class='box'>
          <h2>Winners breakdown (daily)</h2>
          <table id='winners-table'>
            <thead>
              <tr>
                <th>Period</th>
                <th class='num'>P90 Win (%)</th>
                <th class='num'># Win Days</th>
                <th class='num'>Top-5 Wins / Wins</th>
              </tr>
            </thead>
            <tbody>
              {''.join(win_tr)}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class='section'>
      <div class='card'>
        <div class='box'>
          <h2>Losers breakdown (daily)</h2>
          <table id='losers-table'>
            <thead>
              <tr>
                <th>Period</th>
                                <th class='num'>P10 Loss (%)</th>
                <th class='num'># Loss Days</th>
                <th class='num'>Worst-5 Losses / Losses</th>
              </tr>
            </thead>
            <tbody>
              {''.join(loss_tr)}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    <div class='footer sub'>Generated by quantlab.generate_robustness_report.report_robustness</div>
  </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path
