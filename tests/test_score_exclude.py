import pandas as pd

from quantlab.rnd import _score_candidate


def _bt(eq_vals):
    # Minimal bt df for scoring
    idx = pd.date_range("2020-01-01", periods=len(eq_vals), freq="D")
    return pd.DataFrame({"equity": eq_vals}, index=idx)


def test_score_exclude_omits_holdout_from_total_and_constraints():
    period_dfs = {
        "train": _bt([1000, 1100]),   # +10%
        "holdout": _bt([1000, 2000]), # +100%
    }

    score, df = _score_candidate(period_dfs, dd_cap_percent=20.0, initial_capital=1000.0, score_exclude=["holdout"])

    # Total should reflect only train (+10%)
    assert abs(score.sum_pnl - 10.0) < 1e-6

    # Ensure dataframe includes both periods and scored flag is correct
    rows = {r["period"]: r for r in df.to_dict(orient="records") if r["period"] != "TOTAL"}
    assert rows["train"]["scored"] is True
    assert rows["holdout"]["scored"] is False
