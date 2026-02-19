from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VAR_DIR = Path('reports/trend_based/ablation/variants')
OUT_DIR = Path('reports/trend_based/ablation')

PERIODS = ['2021-2022', '2023-2025', '2026']


def variant_from_name(name: str) -> dict:
    # best_trend__all_on.html OR best_trend__no-ema_sep__no-corr.html
    stem = Path(name).stem
    parts = stem.split('__')
    assert parts[0] == 'best_trend'
    disabled = set()
    if len(parts) >= 2 and parts[1] != 'all_on':
        for p in parts[1:]:
            if p.startswith('no-'):
                disabled.add(p.removeprefix('no-'))
    feats = ['ema_sep', 'nochop', 'base_slope', 'corr']
    out = {
        'variant': stem,
        'file': name,
    }
    for f in feats:
        out[f] = 0 if f in disabled else 1
    out['disabled'] = ','.join(sorted(disabled)) if disabled else '(none)'
    return out


def parse_table(html_path: Path) -> pd.DataFrame:
    """Parse the first summary table from the HTML report without lxml.

    We use BeautifulSoup (available in env) to avoid pandas.read_html dependency on lxml/html5lib.
    """

    from bs4 import BeautifulSoup

    html = html_path.read_text(encoding='utf-8', errors='replace')
    soup = BeautifulSoup(html, 'html.parser')

    table = soup.find('table')
    if table is None:
        raise ValueError(f'No <table> found in {html_path}')

    # headers
    headers = []
    thead = table.find('thead')
    if thead:
        ths = thead.find_all('th')
        headers = [th.get_text(strip=True) for th in ths]

    # rows
    body = table.find('tbody') or table
    rows = []
    for tr in body.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds:
            continue
        rows.append([td.get_text(strip=True) for td in tds])

    if not rows:
        raise ValueError(f'No rows parsed from {html_path}')

    if headers and len(headers) == len(rows[0]):
        df = pd.DataFrame(rows, columns=headers)
    else:
        df = pd.DataFrame(rows)

    # normalize column names
    df.columns = [
        str(c)
        .replace('PnL (%)', 'PnL_pct')
        .replace('Max DD (%)', 'MaxDD_pct')
        .replace('Win Rate', 'WinRate')
        .replace('Profit Factor', 'ProfitFactor')
        .replace('Avg Win', 'AvgWin')
        .replace('Avg Loss', 'AvgLoss')
        .replace('# Trades', 'Trades')
        for c in df.columns
    ]

    # Convert string metrics like "-0.24%" to floats
    def pct_to_float(x):
        if isinstance(x, str) and x.strip().endswith('%'):
            return float(x.strip().replace('%', ''))
        return x

    for col in ['PnL_pct', 'MaxDD_pct', 'WinRate', 'AvgWin', 'AvgLoss']:
        if col in df.columns:
            df[col] = df[col].map(pct_to_float)

    for col in ['Sharpe', 'ProfitFactor']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Trades' in df.columns:
        df['Trades'] = pd.to_numeric(df['Trades'], errors='coerce').astype('Int64')

    return df


def main() -> None:
    rows = []
    for html_path in sorted(VAR_DIR.glob('best_trend__*.html')):
        meta = variant_from_name(html_path.name)
        df = parse_table(html_path)
        # keep only the periods we care about
        df = df[df['Period'].isin(PERIODS)].copy()
        if df.shape[0] != 3:
            raise ValueError(f'{html_path}: expected 3 period rows, got {df.shape[0]}')

        for _, r in df.iterrows():
            row = dict(meta)
            row['period'] = r['Period']
            for k in ['PnL_pct', 'MaxDD_pct', 'Sharpe', 'WinRate', 'ProfitFactor', 'AvgWin', 'AvgLoss', 'Trades']:
                row[k] = r.get(k)
            rows.append(row)

    out = pd.DataFrame(rows)

    # Add simple aggregate scores (across periods)
    # - total pnl (sum of period pnl %)
    # - avg sharpe (mean)
    # - worst drawdown (most negative MaxDD)
    agg = (
        out.groupby('variant')
        .agg(
            ema_sep=('ema_sep', 'max'),
            nochop=('nochop', 'max'),
            base_slope=('base_slope', 'max'),
            corr=('corr', 'max'),
            disabled=('disabled', 'first'),
            pnl_total_pct=('PnL_pct', 'sum'),
            sharpe_mean=('Sharpe', 'mean'),
            maxdd_worst_pct=('MaxDD_pct', 'min'),
            trades_total=('Trades', 'sum'),
        )
        .reset_index()
    )

    # Write long-form CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / 'ablation_metrics_long.csv'
    out.to_csv(out_csv, index=False)

    agg_csv = OUT_DIR / 'ablation_metrics_agg.csv'
    agg.sort_values(['sharpe_mean', 'pnl_total_pct'], ascending=False).to_csv(agg_csv, index=False)

    # Markdown summary
    md_path = OUT_DIR / 'ablation_summary.md'

    def topn(df: pd.DataFrame, period: str, metric: str, n: int = 5, asc: bool = False) -> pd.DataFrame:
        d = df[df['period'] == period].copy()
        return d.sort_values(metric, ascending=asc).head(n)

    lines = []
    lines.append('# Best-trend filter ablation (ema_sep / nochop / base_slope / corr)')
    lines.append('')
    lines.append(f'- Variants dir: `{VAR_DIR}`')
    lines.append(f'- Parsed long metrics CSV: `{out_csv}`')
    lines.append(f'- Parsed aggregate CSV: `{agg_csv}`')
    lines.append('')
    lines.append('## Aggregate ranking (across periods)')
    lines.append('Sorted by mean Sharpe (desc), then total PnL% (desc).')
    lines.append('')
    def df_to_md_table(d: pd.DataFrame) -> str:
        # Minimal markdown table writer (avoid pandas optional deps)
        cols = list(d.columns)
        def fmt(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ''
            if isinstance(v, float):
                return f"{v:.4g}"
            return str(v)
        out_lines = []
        out_lines.append('| ' + ' | '.join(cols) + ' |')
        out_lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
        for _, row in d.iterrows():
            out_lines.append('| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |')
        return '\n'.join(out_lines)

    lines.append(df_to_md_table(agg.sort_values(['sharpe_mean', 'pnl_total_pct'], ascending=False).head(16)))
    lines.append('')

    for period in PERIODS:
        lines.append(f'## Period: {period}')
        lines.append('')
        for metric, asc, title in [
            ('Sharpe', False, 'Top Sharpe'),
            ('PnL_pct', False, 'Top PnL (%)'),
            ('MaxDD_pct', False, 'Least drawdown (MaxDD closest to 0)'),
        ]:
            if metric == 'MaxDD_pct':
                # MaxDD is negative; higher is better
                asc = False
            lines.append(f'### {title}')
            lines.append('')
            d = out[out['period'] == period].copy()
            if metric == 'MaxDD_pct':
                d = d.sort_values(metric, ascending=False)
            else:
                d = d.sort_values(metric, ascending=False)
            lines.append(
                df_to_md_table(
                    d[[
                        'variant', 'disabled', 'PnL_pct', 'MaxDD_pct', 'Sharpe', 'WinRate', 'ProfitFactor', 'AvgWin', 'AvgLoss', 'Trades'
                    ]].head(8)
                )
            )
            lines.append('')

    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out_csv}')
    print(f'Wrote {agg_csv}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
