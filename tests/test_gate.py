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


def test_candidate_must_be_worth_replacing_production_too():
    production = ("production", {"mae": 790.0, "monthly_total_mape": 9.0})
    ok, reasons = should_promote({"mae": 800.0, "monthly_total_mape": 10.0}, {"baseline": BASELINE}, RULES, production)
    assert not ok
    assert "not worth replacing production: worse on MAE and monthly total" in reasons


IN_USE = ("model in use", {"mae": 800.0, "monthly_total_mape": 12.0})


@pytest.mark.parametrize(
    ("mae", "total", "expected", "why"),
    [
        (780.0, 12.0, True, "better on MAE, worse on neither"),  # 2.5% better per admission, same total
        (800.0, 10.0, True, "better on monthly total, worse on neither"),  # same per admission, total 17% better
        (780.0, 10.0, True, "better on MAE and monthly total, worse on neither"),
        (800.0, 12.0, False, "not better on either by its margin"),  # the same model again
        (790.0, 11.5, False, "not better on either by its margin"),  # a little better on both: within noise
        (700.0, 12.5, False, "worse on monthly total"),  # neither number may pay for the other
        (810.0, 8.0, False, "worse on MAE"),
    ],
)
def test_replacing_the_model_in_use_needs_one_number_better_and_none_worse(mae, total, expected, why):
    ok, reasons = should_promote({"mae": mae, "monthly_total_mape": total}, {}, RULES, IN_USE)
    assert ok is expected
    assert any(r.endswith(why) for r in reasons), reasons


def test_a_tie_with_the_model_in_use_is_not_worse():
    # The same typical-cost model refitted: its MAE differs only by rounding noise.
    ok, _ = should_promote({"mae": 800.0 * (1 + 1e-9), "monthly_total_mape": 10.0}, {}, RULES, IN_USE)
    assert ok


def test_ceiling_applies_against_the_model_in_use():
    in_use = ("model in use", {"mae": 800.0, "monthly_total_mape": 30.0})
    ok, reasons = should_promote({"mae": 800.0, "monthly_total_mape": 20.0}, {}, RULES, in_use)
    assert not ok and any("above the 15.0% ceiling" in r for r in reasons)


def test_needs_a_reference():
    with pytest.raises(ValueError):
        should_promote({"mae": 1.0, "monthly_total_mape": 1.0}, {})
