# 2026-02-15 — Loss / Drawdown Deep Dive (best_trend)

Scope: canonical **best_trend** (includes **ShockExit abs(5m ret) ≥ 0.006**, shifted by 1 bar) vs a baseline with shock exit disabled.

Artifacts in this folder:
- `drawdown_summary_shock_abs0p006.csv`
- `drawdown_summary_baseline_no_shock.csv`
- `worst_daily_in_position.csv`
- `segments_highlights.csv` (worst MAE / longest / best MFE segments)
- `drawdown_episodes_head.csv` (first 500 episodes)
- `raw/bt_*.parquet` (per-period backtest frames; archived separately if needed)

---

## 1) Drawdown depth + time-under-water (TUW)

### Max drawdown depth (min DD)
Shock exit reduces the worst drawdown depth in all periods (most noticeably in 2021–2025):

- **2021–2022:** min DD **-10.87%** vs **-12.76%** (improvement ≈ **+1.88pp**)
- **2023–2025:** min DD **-15.51%** vs **-17.99%** (improvement ≈ **+2.48pp**)
- **2026:** min DD **-11.75%** vs **-12.07%** (improvement ≈ **+0.32pp**)

(See `drawdown_summary_*.csv`.)

### Time under water is enormous in 2021–2025
This is important: TUW fraction is ~0.99 in 2021–2025. That means equity spends most bars below its all-time peak once it starts trending up.
So “reduce MaxDD” is doable; “be at new highs frequently” is intrinsically hard for this style.

Comparative episode stats (shock vs no-shock):
- Max episode duration is slightly shorter with shock in 2021–2025.
- Episode count increases with shock in 2021–2025 (more but smaller underwater episodes).

Interpretation:
- ShockExit mainly **chips away the tail** and may fragment long DD regimes into shorter episodes.

---

## 2) Worst daily returns while in-position
From `worst_daily_in_position.csv` (shock variant):

### 2021–2022 (worst days)
- 2021-05-11: **-3.51%** (shock trigger: False)
- 2022-09-13: **-2.96%** (shock trigger: True)
- 2022-12-27: **-2.68%** (shock trigger: False)

### 2023–2025 (worst days)
- 2025-10-17: **-4.84%** (shock trigger: True)
- 2024-08-02: **-4.68%** (shock trigger: True)
- 2023-03-16: **-3.78%** (shock trigger: False)

### 2026 (worst days)
- 2026-01-07: **-5.95%** (shock trigger: False)
- 2026-01-26: **-5.72%** (shock trigger: False)
- 2026-01-14: **-5.06%** (shock trigger: False)

Key takeaway:
- ShockExit helps, but **many of the worst in-position days are not single-bar 0.6% shocks** (or the shock happens when we are already flat / not aligned). Those are candidates for a *multi-bar* loss limiter.

---

## 3) Segment-level tail risk (MAE)
Computed directly from `raw/bt_*.parquet`.

### Pattern: the worst MAE segments are almost always size=2.0 (0.02 lots)
Examples (shock variant):
- 2023–2025 worst segment MAE: **-5.94%** (2024-04-12 entry, size=2.0)
- 2026 worst segment MAE: **-6.17%** (2026-01-26 entry, size=2.0)

Shock vs no-shock comparison:
- Some worst-segment MAE values improve (e.g. 2024-04-12 MAE -6.53% → -5.94%).
- Some segments are unchanged (suggesting losses accrued without a single-bar shock trigger).

This points to a clean, discrete control lever:
- **Protect the 0.02 exposure** (either stricter upgrade criteria or an explicit “risk-off after X adverse move”).

See: `segments_highlights.csv`.

---

## 4) What thresholds might cut the worst tails (without coding yet)

Based on the observed worst daily returns:
- A **daily loss limit** around **-2% to -3%** would have cut many of the worst days across periods.
- A segment-level max-loss (MAE cap) around **-3% to -4%** would catch the worst segments, but must be applied carefully (risk of chopping good trends).

The fact that 2026 has multiple very bad days without the 0.6% shock trigger suggests a second mechanism:
- **multi-bar adverse streak / rolling loss window** (e.g. if last 12 bars return < -X, flatten)

---

## Recommendations
1) Implement **daily loss limit** first (most live-stable, directly targets DD depth).
2) Consider a discrete rule to reduce tail risk from **size=2.0 segments**.
3) If still needed, add a multi-bar loss window exit to catch non-shock cascades.
