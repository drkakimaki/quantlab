# Quantlab - Trading Strategy R&D

Quantlab is a modular trading-strategy development and backtesting framework with a unified execution engine and a config-driven gate pipeline.
It includes a WebUI runner for HTML reports and an agent-friendly `quantlab.rnd` CLI.

## File Structure

```
quantlab/
├── engine/                  # Backtest engine + metrics
│   ├── backtest.py
│   ├── metrics.py
│   └── trades.py            # Canonical trade log
├── strategies/              # Strategy classes + gates
│   ├── base.py              # StrategyBase, BacktestResult, BacktestConfig
│   ├── buy_and_hold.py
│   ├── mean_reversion.py
│   ├── trend_following.py   # TrendStrategy, TrendStrategyWithGates
│   └── gates/               # Composable gate implementations
├── time_filter/             # Time-blocking infra (FOMC + econ calendar)
├── data/
│   ├── dukascopy.py         # Tick download, 1s builder
│   ├── resample.py          # 5m/15m mid + OHLC
│   ├── download.py          # CLI: download + resample
│   └── validate.py          # CLI: check integrity
├── configs/
│   └── trend_based/
│       ├── current.yaml     # Canonical config
│       └── (sweeps/experiments)
├── webui/                   # Browser interface
├── reporting/               # Report generation (HTML)
├── reports/                 # Output reports + decision bundles
├── tests/                   # Unit tests + regression (golden series)
├── rnd.py                   # Low-token CLI runner (agent-friendly)
└── check.sh                 # Unit + regression checks (one command)
```

## Quick Start

### WebUI (human inspection)
```bash
.venv/bin/python quantlab/webui/backtest_ui.py --port 8080
```

### Report generation (canonical HTML outputs)
```bash
.venv/bin/python -c "from quantlab.webui.runner import run_backtest; run_backtest('best_trend')"
```

### CLI R&D loop (agent-friendly)
```bash
# JSON by default; use --format text for humans.

# Configs
# - canonical run:     quantlab/configs/trend_based/current.yaml
# - experiments:       quantlab/configs/trend_based/experiment*.yaml
# - sweeps (grids):    quantlab/configs/trend_based/sweeps*.yaml

# Single scoring run
.venv/bin/python -m quantlab.rnd run --config quantlab/configs/trend_based/current.yaml

# Grid sweep (writes decision bundle if --decision-slug is set)
.venv/bin/python -m quantlab.rnd sweep --sweep quantlab/configs/trend_based/sweeps.yaml --decision-slug sweep_fast_slow
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

`TrendStrategyWithGates` applies (current order):

`base → HTF → EMASep → NoChop → (Corr optional) → TimeFilter → EMA-strength sizing → SeasonalitySizeCap → Churn → Mid-loss limiter → Time-stop → ShockExit`

### Entry filters (regime / permission)
| Gate | Purpose |
|------|---------|
| `HTFConfirmGate` | 15m SMA alignment |
| `EMASeparationGate` | EMA separation > k×ATR |
| `NoChopGate` | Avoid choppy markets |
| `CorrelationGate` | XAG/EUR correlation stability (**currently disabled in canonical**) |
| `TimeFilterGate` | Force-flat during blocked windows (FOMC / econ_calendar) |

### Sizing overlays (do not create entries)
| Gate | Purpose |
|------|---------|
| `EMAStrengthSizingGate` | Size=2 on strong EMA separation (segment-held) |
| `SeasonalitySizeCapGate` | Month-based size cap (e.g. June size<=1) |

### Trade frequency control
| Gate | Purpose |
|------|---------|
| `ChurnGate` | Entry debounce + re-entry cooldown |

### Post-entry exits
| Gate | Purpose |
|------|---------|
| `MidDurationLossLimiterGate` | Kill mid-duration losers (e.g. 13–48 bars under -1%) |
| `NoRecoveryExitGate` | Kill trades that fail to recover by N bars |
| `ShockExitGate` | Shock exits (+ optional cooldown) |

Gate is ON when it appears in the `pipeline:` list (order = list order).

Note: `time_filter.kind` supports `fomc` and `econ_calendar` (exact values).

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

Canonical config (tracked):
- `quantlab/configs/trend_based/current.yaml`

Gate pipeline config:
- `current.yaml` uses `pipeline:`, a list of `{gate, params}` entries.
- Gate is ON if it appears in `pipeline:`.
- Gate order is the list order.

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

### Regression & parity (mandatory when refactoring execution semantics)

When touching anything that can alter signals/execution (gates, costs, time-filtering, return math, trade extraction), regress on **position + equity series**, not just headline metrics.

Golden series regression (best_trend 2024–2025):

```bash
# run regression
.venv/bin/python -m pytest -q tests/regression/test_best_trend_2024_2025_series.py

# update golden (ONLY when the change is intentional)
.venv/bin/python tests/regression/update_golden_best_trend_2024_2025.py
```

Notes:
- Golden artifacts live under `tests/regression/golden/`.
- We bias toward CLI/WebUI parity; the runner intentionally reuses WebUI loaders.

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
