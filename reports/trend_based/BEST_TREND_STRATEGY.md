# Best Trend — Strategy Overview + Canonical Pipeline (XAUUSD, EMA)

**Last updated:** 2026-02-25 (Europe/Berlin)

This document describes the current **best_trend** canonical variant.

- **Current canonical config:** `configs/trend_based/current.yaml` (**EMA**)
- Previous SMA canonical config (archived): `configs/trend_based/best_sma_trend.yaml`

Reference reports:
- Equity/perf: `reports/trend_based/best_ema_trend.html`
- Trades: `reports/trend_based/best_ema_trend_trades.html`
- Yearly equity/perf: `reports/trend_based/best_ema_trend_yearly.html`

---

## 0) Current EMA results (snapshot)

Holdout policy: **2026 is excluded from header score aggregates** (`score_exclude: ['2026']`).

From `reports/trend_based/best_ema_trend.html`:
- Sum PnL (scored): **437.56%**
- Worst MaxDD (scored): **-15.63%**
- Avg Sharpe (scored): **1.52**
- # Trades: **1,867**
- Per period:
  - 2020–2022: **50.23%** (vs B&H **+30.25%**), MaxDD **-15.63%**, Sharpe **0.74** (CI **-0.31/1.63**)
  - 2023–2025: **387.33%** (vs B&H **+250.99%**), MaxDD **-13.43%**, Sharpe **2.29** (CI **1.35/3.09**)
  - 2026 (holdout): **148.89%** (vs B&H **+127.89%**), MaxDD **-18.95%**, Sharpe **5.23** (CI **1.61/7.87**)

---

## 1) Overview (what it is)

**Hypothesis:** long-only trend following on XAUUSD where most value comes from (a) staying out of chop, (b) avoiding macro windows, and (c) sizing up only when the HTF regime is strong.

**Base signal:** EMA crossover (fast/slow) on 5m close.

**Meta-order (recommended):**

`base signal → entry filters → time filter → sizing overlays → post-entry exits`

---

## 2) Canonical config (where truth lives)

Canonical config:
- `configs/trend_based/current.yaml`

Holdout policy:
- `periods.score_exclude: ['2026']` (holdout is shown in tables but excluded from header scoring aggregates)

---

## 3) Canonical pipeline (exact order + intent)

Source of truth: `configs/trend_based/current.yaml` → `trend:` + `pipeline:`

**Base signal:** EMA crossover on 5m close (`fast=30`, `slow=110`).

1. **ema_sep** — HTF EMA separation filter (ATR-scaled)
   - Purpose: avoid chop; require fast>slow and meaningful separation

2. **nochop** — HTF NoChop filter
   - Purpose: require sustained strength (enough closes above EMA)

3. **time_filter** — force-flat allow-mask
   - Purpose: block FOMC window; force-flat June (month block)

4. **ema_strength_sizing** — size up on strong separation (segment-held)
   - Purpose: increase size only in strong regimes

5. **shock_exit** — shock day kill-switch (+ optional cooldown)
   - Purpose: exit on large adverse bar moves

---

## 4) Notes / guardrails

- The previous SMA canonical description is preserved in:
  - `reports/trend_based/BEST_SMA_TREND_STRATEGY.md`
- If you change anything that affects execution semantics (gates/costs/time filter/return math/trade extraction), run the golden regression:
  - `./check.sh`
