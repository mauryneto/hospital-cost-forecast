"""PSI and the retrain rule: numbers in, decisions out."""

import numpy as np
import pandas as pd
import pytest

from costmodel.drift import PSI_ALERT, PSI_STABLE, RetrainRules, known_through, monthly_metrics, psi, should_retrain

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
