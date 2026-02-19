# 2026-02-14 — Shock exits (kill-switch experiments)

## Purpose
Test a simple, discrete **shock exit** mechanism to reduce the damage from the biggest drawdown-deepening days.
This respects the constraint that position sizes are discrete (0.01 / 0.02 lots only).

## Mechanism
When a shock is detected (computed on 5m returns and **shifted by 1 bar** to avoid lookahead), we:
1) if currently in-position: **force flat for the rest of the segment**
2) optionally: **cooldown** (force flat for N bars after a shock)

Shock trigger variants tested:
- Absolute return: `abs(5m_return) >= threshold`
- Sigma shock: `abs(5m_return) >= k * rolling_std(returns, window)`

## Reports generated
- `reports/trend_based/best_trend_shock_abs0p004.html`
- `reports/trend_based/best_trend_shock_abs0p006.html`
- `reports/trend_based/best_trend_shock_4sigma_96b.html`
- `reports/trend_based/best_trend_shock_5sigma_96b.html`
- `reports/trend_based/best_trend_shock_4sigma_96b_cd12.html`

## Headline outcomes vs baseline (qualitative)
- `abs_ret>=0.006` was the only variant that looked plausibly competitive overall:
  - improved 2021–2022 and 2023–2025, mildly reduced 2026 PnL
  - improved drawdown across all periods
- Sigma-based shocks (4σ/5σ) tended to **hurt 2023–2025 PnL badly** despite improving 2021–2022 drawdown.
- Adding cooldown (12 bars) made the sigma variant even more restrictive.

## Parameter search (absolute-return shock)
We ran a small sweep over `shock_exit_abs_ret` in:
`{0.0035, 0.0040, 0.0045, 0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075, 0.0080}`.

Summary CSV:
- `shock_abs_search.csv`

### Best by mean Sharpe (across periods)
- **0.0045**
  - mean Sharpe: **4.272** (baseline 3.979, +0.293)
  - worst MaxDD: **-15.27%** (baseline -16.98%, improvement +1.71pp)
  - sum PnL%: **483.36** (baseline 508.50, -25.14)

### Best “balanced” pick (keeps PnL while improving DD/Sharpe)
- **0.0060** (PROMOTED to canonical best_trend)
  - mean Sharpe: **4.162** (+0.183)
  - worst MaxDD: **-15.22%** (+1.77pp)
  - sum PnL%: **510.79** (+2.29)

Interpretation:
- Smaller thresholds (e.g. 0.0035–0.0040) become too aggressive and start sacrificing too much 2026 PnL.
- Thresholds around **0.006–0.0075** are the sweet spot if we want a kill-switch that doesn’t cripple the core trend engine.

## Next step
If we want to continue, focus on a tighter local sweep around 0.006–0.0075 and validate it
against the specific top drawdown-deepening days/episodes from the attribution study.
