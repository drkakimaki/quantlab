# Quantlab - Trading Strategy Backtesting

Repo: https://github.com/drkakimaki/quantlab

Modular backtesting with Strategy classes, composable gates, and unified engine.

## File Structure

```
quantlab/
├── engine/                  # Backtest engine + metrics
│   ├── backtest.py
│   ├── metrics.py
│   └── trades.py            # Canonical trade log
├── strategies/              # Strategy classes + gates
│   ├── base.py              # StrategyBase, BacktestResult
│   ├── buy_and_hold.py
│   ├── mean_reversion.py
│   └── trend_following.py   # TrendStrategy, TrendStrategyWithGates, gates
├── time_filter/             # FOMC/session blocking
├── data/
│   ├── dukascopy.py         # Tick download, 1s builder
│   ├── resample.py          # 5m/15m mid + OHLC
│   ├── download.py          # CLI: download + resample
│   └── validate.py          # CLI: check integrity
├── configs/trend_based/current.yaml
├── webui/                   # Browser interface
├── reporting/               # Report generation (HTML)
├── rnd.py                   # Low-token CLI runner (agent-friendly)
└── reports/                 # Output reports + decision bundles
```

## Quick Start

### WebUI (human inspection)
```bash
.venv/bin/python quantlab/webui/backtest_ui.py --port 8080
```

### CLI R&D loop (agent-friendly)
```bash
# Single run from experiment config
.venv/bin/python -m quantlab.rnd run \
  --config quantlab/configs/trend_based/experiment.yaml \
  --mode yearly \
  --dd-cap 20

# Grid sweep (writes decision bundle if --decision-slug is set)
.venv/bin/python -m quantlab.rnd sweep \
  --sweep quantlab/configs/trend_based/sweeps.yaml \
  --decision-slug sweep_fast_slow
```

## Strategy Classes

| Class | Purpose |
|-------|---------|
| `BuyAndHoldStrategy` | Buy at start, hold to end |
| `MeanReversionStrategy` | Z-score mean reversion |
| `TrendStrategy` | Simple MA crossover |
| `TrendStrategyWithGates` | Trend + composable filters |

```python
from quantlab.strategies import TrendStrategyWithGates, BacktestConfig
from quantlab.engine.trades import extract_trade_log

strategy = TrendStrategyWithGates.from_config(config, allow_mask=fomc_mask)
result = strategy.run_backtest(prices, context={
    "bars_15m": bars_15m,
    "prices_xag": xag_prices,
    "prices_eur": eur_prices,
})

# Canonical trade log (one row per trade segment)
trade_log = extract_trade_log(result.df)
```

## Composable Gates

`TrendStrategyWithGates` applies: `base → HTF → EMASep → NoChop → Corr → TimeFilter → SeasonalitySizeCap → Churn → Risk`

| Gate | Purpose |
|------|---------|
| `HTFConfirmGate` | 15m SMA alignment |
| `EMASeparationGate` | EMA separation > k×ATR |
| `NoChopGate` | Avoid choppy markets |
| `CorrelationGate` | XAG/EUR correlation stability |
| `TimeFilterGate` | FOMC force-flat windows |
| `SeasonalitySizeCapGate` | Month-based size cap (e.g. June size<=1) |
| `ChurnGate` | Entry debounce + re-entry cooldown |
| `RiskGate` | Shock exits (+ optional cooldown) |

Gate is ON when config block present, OFF when missing.

## Data Management

```bash
# Download recent data
.venv/bin/python quantlab/data/download.py --symbols XAUUSD XAGUSD EURUSD --days 7

# Download specific period
.venv/bin/python quantlab/data/download.py --symbol XAUUSD --start 2020-01-01 --end 2020-12-31

# Validate integrity
.venv/bin/python quantlab/data/validate.py
```

**Data formats:**
| Folder | Columns |
|--------|---------|
| `dukascopy_1s` | ts, bid, ask, mid, spread |
| `dukascopy_5m` | ts, bid, ask, mid, spread |
| `dukascopy_5m_ohlc` | ts, open, high, low, close |
| `dukascopy_15m` | ts, bid, ask, mid, spread |
| `dukascopy_15m_ohlc` | ts, open, high, low, close |

## Configuration

`quantlab/configs/trend_based/current.yaml`:
```yaml
symbol: XAUUSD
trend: {fast: 30, slow: 75}
htf_confirm: {rule: "15min"}
ema_sep: {ema_fast: 40, ema_slow: 300, sep_k: 0.05}
nochop: {ema: 20, lookback: 40, min_closes: 24}
corr: {logic: "or", xag: {min_abs: 0.10}, eur: {min_abs: 0.10}}
time_filter: {kind: "fomc"}
churn: {min_on_bars: 3, cooldown_bars: 8}
risk: {shock_exit_abs_ret: 0.006}
costs: {fee_per_lot: 3.50, spread_per_lot: 3.50}  # IC Markets rough baseline
```

## Web UI

Browser-based backtest runner at http://localhost:8080

**Reports served by the UI:**
- Equity/performance report: `/report/<strategy_id>`
- Trade breakdown report: `/trades/<strategy_id>`
- Yearly variant: add `?mode=yearly`

**Available strategies:**
- Buy & Hold
- Trend (MA 20/100) - baseline
- Mean Reversion (Z-score)
- Best Trend (all filters)

## Development

```bash
.venv/bin/python quantlab/webui/backtest_ui.py --port 8080
```

- **API Key:** FRED_API_KEY for economic calendar
- **Design:** Strategy classes with composable gates, single backtest engine

## Reports

### HTML generation code
- `quantlab/reporting/generate_bt_report.py` (multi-period single-file equity/performance HTML)
- `quantlab/reporting/generate_trades_report.py` (multi-period single-file trade breakdown HTML)

### Outputs
```
quantlab/reports/
├── baselines/           # Simple strategy reports (overwritten)
│   ├── buy_and_hold.html
│   ├── trend.html
│   └── mean_reversion.html
├── trend_based/         # Best trend variants (overwritten)
│   ├── best_trend.html
│   ├── BEST_TREND_STRATEGY.md
│   └── decisions/       # Decision bundles (artifacts we keep)
│       └── YYYY-MM-DD_<slug>/
│           ├── DECISION.md
│           ├── best.yaml
│           ├── results.csv
│           ├── top_k.csv        # sweep only
│           ├── notes.json
│           └── raw/
```
