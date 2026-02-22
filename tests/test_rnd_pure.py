from __future__ import annotations

import pandas as pd

import quantlab.rnd as rnd


def test_set_in_get_in_roundtrip() -> None:
    cfg: dict = {}
    rnd._set_in(cfg, "a.b.c", 123)
    assert cfg == {"a": {"b": {"c": 123}}}
    assert rnd._get_in(cfg, "a.b.c") == 123
    assert rnd._get_in(cfg, "a.b") == {"c": 123}
    assert rnd._get_in(cfg, "a.x") is None


def test_better_feasible_beats_infeasible() -> None:
    a = rnd.CandidateScore(ok=False, sum_pnl=999, worst_maxdd=-50, avg_sharpe=10, maxdd_violation=30)
    b = rnd.CandidateScore(ok=True, sum_pnl=-1, worst_maxdd=-1, avg_sharpe=-1, maxdd_violation=0)
    assert rnd._better(a, b) is True


def test_better_infeasible_smaller_violation_wins() -> None:
    a = rnd.CandidateScore(ok=False, sum_pnl=10, worst_maxdd=-30, avg_sharpe=1, maxdd_violation=10)
    b = rnd.CandidateScore(ok=False, sum_pnl=999, worst_maxdd=-25, avg_sharpe=0, maxdd_violation=5)
    assert rnd._better(a, b) is True


def test_score_candidate_basic() -> None:
    # Synthetic intraday equity; Sharpe is computed from daily closes.
    idx = pd.date_range("2025-01-01", periods=24 * 12 * 5, freq="5min", tz="UTC")  # 5 days

    bt1 = pd.DataFrame(
        {
            "equity": (1000.0 + pd.Series(range(len(idx)), index=idx) * 0.1).astype(float),
            "returns_net": 0.0,
            "position": 1.0,
        },
        index=idx,
    )

    bt2 = pd.DataFrame(
        {
            "equity": (1000.0 + pd.Series(range(len(idx)), index=idx) * 0.05).astype(float),
            "returns_net": 0.0,
            "position": 1.0,
        },
        index=idx,
    )

    score, results = rnd._score_candidate(
        {"p1": bt1, "p2": bt2},
        dd_cap_percent=20.0,
        initial_capital=1000.0,
    )

    # Has p1, p2, TOTAL
    assert set(results["period"].tolist()) == {"p1", "p2", "TOTAL"}

    # Feasible under cap (drawdown should be near 0 for monotonic equity)
    assert score.ok is True

    # Total pnl is sum of period pnls
    p1_pnl = float(results.loc[results["period"] == "p1", "pnl_percent"].iloc[0])
    p2_pnl = float(results.loc[results["period"] == "p2", "pnl_percent"].iloc[0])
    tot_pnl = float(results.loc[results["period"] == "TOTAL", "pnl_percent"].iloc[0])
    assert abs((p1_pnl + p2_pnl) - tot_pnl) < 1e-9
