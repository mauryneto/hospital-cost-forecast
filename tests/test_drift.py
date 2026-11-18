"""PSI and the retrain rule: numbers in, decisions out."""

import numpy as np
import pandas as pd
import pytest

from costmodel.drift import (
    PSI_ALERT,
    PSI_STABLE,
    RetrainRules,
    error_by_group,
    known_through,
    monthly_metrics,
    pair_monthly_metrics,
    price_level,
    price_shifts,
    psi,
    should_retrain,
)
from costmodel.evaluate import pair_summary, summary

rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_the_same_distribution():
    a, b = pd.Series(rng.normal(50, 10, 20_000)), pd.Series(rng.normal(50, 10, 20_000))
    assert psi(a, b) < 0.01


def test_psi_flags_a_shifted_numeric_distribution():
    older = pd.Series(rng.normal(50, 10, 20_000))
    assert psi(older, older + 2) < PSI_STABLE
    assert psi(older, older + 10) > PSI_ALERT


def test_psi_flags_a_new_mix_of_categories():
    before = pd.Series(rng.choice(["A", "B", "C"], 10_000, p=[0.6, 0.3, 0.1]))
    after = pd.Series(rng.choice(["A", "B", "C", "NEW"], 10_000, p=[0.2, 0.2, 0.1, 0.5]))
    assert psi(before, after) > PSI_ALERT
    assert psi(before, before.sample(frac=1, random_state=1)) < 0.01


def test_psi_counts_missing_values_as_a_bucket():
    full = pd.Series(rng.normal(0, 1, 10_000))
    holes = full.mask(rng.random(10_000) < 0.4)
    assert psi(full, holes) > PSI_ALERT


RULES = RetrainRules(label_delay_months=3, mae_tolerance=0.10, patience=2)


def month(period, mae, total=0.0):
    return {"period": period, "mae": mae, "total_error_pct": total}


def test_rule_needs_patience_months_in_a_row():
    # Reference 800, limit 880.
    assert not should_retrain([month("2024-01", 900)], 800, RULES, 15)[0]  # only one mature month
    assert not should_retrain([month("2024-01", 900), month("2024-02", 850)], 800, RULES, 15)[0]
    assert not should_retrain([month("2024-01", 900), month("2024-02", 850), month("2024-03", 900)], 800, RULES, 15)[0]
    fire, reason = should_retrain([month("2024-01", 850), month("2024-02", 900), month("2024-03", 950)], 800, RULES, 15)
    assert fire and "2024-03" in reason


def test_rule_also_fires_on_the_monthly_total():
    under = [month("2024-01", 800, -16), month("2024-02", 800, -18)]
    assert should_retrain(under, 800, RULES, 15)[0]
    assert not should_retrain(under, 800, RULES, None)[0]  # without a ceiling, only MAE counts


def test_labels_are_known_only_after_the_delay():
    assert known_through(pd.Period("2024-06", "M"), RULES) == pd.Period("2024-03", "M")


def test_monthly_metrics():
    period = pd.Series(["2024-01", "2024-01", "2024-02"])
    out = monthly_metrics(np.array([100.0, 300.0, 200.0]), np.array([110.0, 270.0, 200.0]), period)
    assert [m["period"] for m in out] == ["2024-01", "2024-02"]
    assert out[0]["mae"] == pytest.approx(20.0)
    assert out[0]["total_error_pct"] == pytest.approx(-5.0)
    assert out[1]["mae"] == 0.0


def repriced_year():
    """Two procedures, 100 admissions of each per month. A is paid 100 all year; B doubles from 200 to 400 in July."""
    rows = []
    for month in range(1, 13):
        for code, price in (("A", 100.0), ("B", 200.0 if month < 7 else 400.0)):
            kind = "urgent" if code == "A" else "elective"
            rows += [{"period": f"2024-{month:02d}", "code": code, "y": price, "type": kind}]
    return pd.concat([pd.DataFrame(rows)] * 100, ignore_index=True)


def test_error_by_group_separates_the_group_that_went_wrong():
    df = repriced_year()
    pred = df["code"].map({"A": 100.0, "B": 200.0}).to_numpy()  # a model that learned the old prices
    out = error_by_group(df["y"].to_numpy(), pred, df["period"], df["type"])
    assert out["2024-06"]["elective"]["total_error_pct"] == pytest.approx(0)
    assert out["2024-07"]["elective"]["total_error_pct"] == pytest.approx(-50)
    assert out["2024-07"]["urgent"]["total_error_pct"] == pytest.approx(0)
    assert out["2024-07"]["urgent"]["n"] == 100


def test_price_level_ignores_a_change_of_mix_and_shows_a_change_of_price():
    df = repriced_year()
    last_seen = pd.Series({"A": 100.0, "B": 200.0})
    levels = {m["period"]: m for m in price_level(df["y"].to_numpy(), df["period"], df["code"], last_seen)}
    assert levels["2024-06"]["price_index_pct"] == pytest.approx(0)
    assert levels["2024-07"]["price_index_pct"] == pytest.approx(100 * (500 / 300 - 1))  # B doubled, A did not move
    # More of the expensive procedure at the same prices is a change of mix, not of price.
    more_b = pd.concat([df, df[(df["code"] == "B") & (df["period"] == "2024-03")]])
    march = price_level(more_b["y"].to_numpy(), more_b["period"], more_b["code"], last_seen)[2]
    assert march["period"] == "2024-03" and march["price_index_pct"] == pytest.approx(0)
    # A procedure the model never saw has no reference price: it is left out and reported as not covered.
    new = pd.concat([df, pd.DataFrame({"period": ["2024-01"] * 100, "code": ["NEW"] * 100, "y": [300.0] * 100})])
    january = price_level(new["y"].to_numpy(), new["period"], new["code"], last_seen)[0]
    assert january["price_index_pct"] == pytest.approx(0)
    assert january["covered_pct"] == pytest.approx(50)


def test_price_shifts_ranks_the_repriced_procedure_first():
    df = repriced_year()
    pred = df["code"].map({"A": 100.0, "B": 200.0}).to_numpy()
    reference = pd.DataFrame({"mean": {"A": 100.0, "B": 200.0}, "median": {"A": 100.0, "B": 200.0}})
    shifts = price_shifts(df["y"].to_numpy(), pred, df["period"], df["code"], reference)
    assert [s["code"] for s in shifts] == ["B", "A"]
    b = shifts[0]
    assert b["extra"] == pytest.approx(6 * 100 * 200)  # six months, 100 admissions, 200 above the last-seen price
    assert b["reference_median"] == 200
    assert b["monthly_median"]["2024-06"] == 200 and b["monthly_median"]["2024-07"] == 400
    assert shifts[1]["extra"] == pytest.approx(0)


def test_two_numbers_are_each_judged_by_their_own_metric():
    # Ten admissions in one month: nine cost 500 and one 10,500. The typical cost is 500, the expected cost 1,500.
    y = np.array([500.0] * 9 + [10_500.0])
    period = pd.Series(["2024-01"] * 10)
    typical, expected = np.full(10, 500.0), np.full(10, 1_500.0)
    months = pair_monthly_metrics(y, typical, expected, period)
    assert months[0]["mae"] == 1_000.0  # from the typical cost (the expected cost would miss by 1,800)
    assert months[0]["predicted"] == 15_000.0 and months[0]["total_error_pct"] == 0.0  # from the expected cost
    assert monthly_metrics(y, typical, period)[0]["total_error_pct"] == pytest.approx(-66.67, abs=0.01)

    both = pair_summary(y, typical, expected, period)
    assert both["mae"] == 1_000.0 and both["monthly_total_mape"] == 0.0 and both["bias_pct"] == 0.0
    assert pair_summary(y, typical, typical, period) == summary(y, typical, period)  # one number: nothing changes
