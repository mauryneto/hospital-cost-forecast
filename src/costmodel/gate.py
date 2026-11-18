"""The promotion gate: may a candidate model replace what is in use today?

A candidate gives two numbers per admission, and each is judged by the metric it is for: the **typical cost** by the
error per admission (MAE), the **expected cost** by the error on the month's total. The gate reads both from one
``candidate`` dict (``mae`` and ``monthly_total_mape``); a model that gives a single number is judged on it for both.

**Against every baseline** (lookup tables an analyst could build by hand), both must hold:

1. **Per admission:** its MAE is at least ``min_mae_gain`` (e.g. 2%) lower. A smaller gain is within noise and not worth
   the complexity of a model over a lookup table.
2. **Monthly total:** its monthly-total error is not worse. A model that is better per admission but worse at the
   month's total would mislead whoever provisions money with it.

**Against the model in use**, the question is whether replacing it is worth it: at least one of the two numbers must
be better by its margin (``min_mae_gain`` for the MAE, ``min_total_gain`` for the monthly total, both relative), and
the other must not be worse. Either number improving is a reason to deploy; neither may pay for the other.

On top of that, ``max_monthly_total_mape`` is an absolute ceiling, because the best-MAE baseline (the median per
procedure) is itself poor at totals, and beating a poor number is not enough.

The function is pure (numbers in, decision out) so it can be tested without data or MLflow.
"""

from __future__ import annotations

from dataclasses import dataclass

TIE = 1e-6  # relative difference below which two numbers count as the same (not better, not worse)


@dataclass(frozen=True)
class GateRules:
    min_mae_gain: float = 0.02
    max_monthly_total_mape: float | None = 15.0
    min_total_gain: float = 0.10


def should_promote(candidate: dict, baselines: dict[str, dict], rules: GateRules = GateRules(),
                   in_use: tuple[str, dict] | None = None) -> tuple[bool, list[str]]:  # fmt: skip
    """Return (promote?, reasons). ``candidate``, each baseline and the model in use (``(name, metrics)``) hold ``mae``
    and ``monthly_total_mape``."""
    if not baselines and in_use is None:
        raise ValueError("the gate needs at least one reference to compare with")
    reasons: list[str] = []
    ok = True
    mae, total = candidate["mae"], candidate["monthly_total_mape"]
    for name, ref in baselines.items():
        gain = 1 - mae / ref["mae"]
        if gain >= rules.min_mae_gain:
            reasons.append(f"MAE {mae:.2f} is {100 * gain:.1f}% below {name} ({ref['mae']:.2f})")
        else:
            ok = False
            reasons.append(
                f"MAE {mae:.2f} is only {100 * gain:.1f}% below {name} ({ref['mae']:.2f}); "
                f"needs {100 * rules.min_mae_gain:.1f}%"
            )
        if total <= ref["monthly_total_mape"]:
            reasons.append(f"monthly total error {total:.2f}% is not worse than {name} ({ref['monthly_total_mape']:.2f}%)")
        else:
            ok = False
            reasons.append(f"monthly total error {total:.2f}% is worse than {name} ({ref['monthly_total_mape']:.2f}%)")
    if in_use is not None:
        name, ref = in_use
        mae_gain = 1 - mae / ref["mae"]
        total_gain = 1 - total / ref["monthly_total_mape"] if ref["monthly_total_mape"] > 0 else 0.0
        gains = (("MAE", mae_gain, rules.min_mae_gain), ("monthly total", total_gain, rules.min_total_gain))
        better = [label for label, gain, needed in gains if gain >= needed]
        worse = [label for label, gain, _ in gains if gain < -TIE]
        reasons.append(
            f"against {name}: MAE {mae:.2f} vs {ref['mae']:.2f} ({100 * mae_gain:+.1f}%,"
            f" {100 * rules.min_mae_gain:.1f}% counts as better); monthly total error {total:.2f}% vs"
            f" {ref['monthly_total_mape']:.2f}% ({100 * total_gain:+.1f}%, {100 * rules.min_total_gain:.1f}% counts as"
            " better)"
        )
        if better and not worse:
            reasons.append(f"worth replacing {name}: better on {' and '.join(better)}, worse on neither")
        else:
            ok = False
            why = f"worse on {' and '.join(worse)}" if worse else "not better on either by its margin"
            reasons.append(f"not worth replacing {name}: {why}")
    ceiling = rules.max_monthly_total_mape
    if ceiling is not None:
        if total <= ceiling:
            reasons.append(f"monthly total error {total:.2f}% is within the {ceiling:.1f}% ceiling")
        else:
            ok = False
            reasons.append(f"monthly total error {total:.2f}% is above the {ceiling:.1f}% ceiling")
    return ok, reasons
