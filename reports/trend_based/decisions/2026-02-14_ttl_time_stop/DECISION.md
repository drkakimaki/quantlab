# 2026-02-14 — Segment TTL (time-stop) sweep

## Purpose
Test a simple **time stop** to cap slow bleed / stale segments:
- If in-position for >= N 5m bars since entry, force flat for the rest of the segment.

This is intended to improve drawdown depth and live stability without adding more indicator gates.

## Sweep
- TTL bars tested (big sweep): 0, 72, 144, 216, 288, 360, 432, 576, 720, 864, 1152, 1440
- Canonical best_trend config used (including ShockExit abs_ret>=0.006).

## Results (headline)
Only TTL values <= 288 had any effect; larger TTLs did not bind (identical to baseline).

Notable settings:
- **TTL=288 (~1 day)**
  - mean Sharpe: **+0.0167**
  - sum PnL%: **+6.41**
  - worst MaxDD%: unchanged
  - Tradeoff: 2021–2022 PnL down, 2023–2025 & 2026 PnL up.

- TTL=216
  - worst MaxDD% improvement: **+0.35pp**
  - but mean Sharpe worse and sum PnL slightly down.

- TTL=144 / TTL=72 were clearly harmful (worse worst drawdown and/or big PnL hit).

## Artifacts
- `ttl_sweep.csv` (earlier small sweep)
- `ttl_sweep_big.csv` (full sweep)
- Raw reports are archived (see `reports/_archive_*/trend_based_decisions_raw/`).
