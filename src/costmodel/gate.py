"""The promotion gate: may a candidate model replace what is in use today?

A candidate is promoted only if, on the same held-out data, it beats **every** reference: the best baseline and, when
there is one, the model currently in production. Two things must hold against each reference:

1. **Per admission:** its MAE is at least ``min_mae_gain`` (e.g. 2%) lower. A smaller gain is within noise and not worth
   the complexity of a model over a lookup table.
2. **Monthly total:** its monthly-total error is not worse. A model that is better per admission but worse at the
   month's total would mislead whoever provisions money with it.

On top of that, ``max_monthly_total_mape`` is an absolute ceiling, because the best-MAE baseline (the median per
procedure) is itself poor at totals, and beating a poor number is not enough.

The function is pure (numbers in, decision out) so it can be tested without data or MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateRules:
    min_mae_gain: float = 0.02
    max_monthly_total_mape: float | None = 15.0


def should_promote(candidate: dict, references: dict[str, dict], rules: GateRules = GateRules()) -> tuple[bool, list[str]]:
    """Return (promote?, reasons). ``candidate`` and each reference hold ``mae`` and ``monthly_total_mape``."""
    if not references:
        raise ValueError("the gate needs at least one reference to compare with")
    reasons: list[str] = []
    ok = True
    for name, ref in references.items():
        gain = 1 - candidate["mae"] / ref["mae"]
        if gain >= rules.min_mae_gain:
            reasons.append(f"MAE {candidate['mae']:.2f} is {100 * gain:.1f}% below {name} ({ref['mae']:.2f})")
        else:
            ok = False
            reasons.append(
                f"MAE {candidate['mae']:.2f} is only {100 * gain:.1f}% below {name} ({ref['mae']:.2f}); "
                f"needs {100 * rules.min_mae_gain:.1f}%"
            )
        if candidate["monthly_total_mape"] <= ref["monthly_total_mape"]:
            reasons.append(
                f"monthly total error {candidate['monthly_total_mape']:.2f}% is not worse than {name} "
                f"({ref['monthly_total_mape']:.2f}%)"
            )
        else:
            ok = False
            reasons.append(
                f"monthly total error {candidate['monthly_total_mape']:.2f}% is worse than {name} "
                f"({ref['monthly_total_mape']:.2f}%)"
            )
    ceiling = rules.max_monthly_total_mape
    if ceiling is not None:
        if candidate["monthly_total_mape"] <= ceiling:
            reasons.append(
                f"monthly total error {candidate['monthly_total_mape']:.2f}% is within the {ceiling:.1f}% ceiling"
            )
        else:
            ok = False
            reasons.append(f"monthly total error {candidate['monthly_total_mape']:.2f}% is above the {ceiling:.1f}% ceiling")
    return ok, reasons
