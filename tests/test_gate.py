"""The promotion gate: numbers in, decision out."""

import pytest

from costmodel.gate import GateRules, should_promote

BASELINE = {"mae": 849.0, "monthly_total_mape": 12.0}
RULES = GateRules(min_mae_gain=0.02, max_monthly_total_mape=15.0)


def test_better_candidate_is_promoted():
    ok, reasons = should_promote({"mae": 800.0, "monthly_total_mape": 10.0}, {"baseline": BASELINE}, RULES)
    assert ok
    assert len(reasons) == 3


def test_gain_below_minimum_is_not_promoted():
    # 1% better per admission is within noise: not worth a more complex model.
    ok, reasons = should_promote({"mae": 840.0, "monthly_total_mape": 10.0}, {"baseline": BASELINE}, RULES)
    assert not ok
    assert any("needs 2.0%" in r for r in reasons)


def test_worse_monthly_total_is_not_promoted_even_with_better_mae():
    ok, reasons = should_promote({"mae": 700.0, "monthly_total_mape": 13.0}, {"baseline": BASELINE}, RULES)
    assert not ok
    assert any("is worse than baseline" in r for r in reasons)


def test_ceiling_applies_even_when_reference_is_poor_at_totals():
    median_baseline = {"mae": 849.0, "monthly_total_mape": 40.0}
    ok, reasons = should_promote({"mae": 700.0, "monthly_total_mape": 30.0}, {"median": median_baseline}, RULES)
    assert not ok
    assert any("above the 15.0% ceiling" in r for r in reasons)
    assert should_promote({"mae": 700.0, "monthly_total_mape": 30.0}, {"median": median_baseline}, GateRules(0.02, None))[0]


def test_candidate_must_beat_production_too():
    production = {"mae": 790.0, "monthly_total_mape": 9.0}
    ok, _ = should_promote({"mae": 800.0, "monthly_total_mape": 10.0}, {"baseline": BASELINE, "production": production})
    assert not ok


def test_needs_a_reference():
    with pytest.raises(ValueError):
        should_promote({"mae": 1.0, "monthly_total_mape": 1.0}, {})
