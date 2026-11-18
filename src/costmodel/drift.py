"""Pure functions for monitoring: the PSI drift measure and the retrain rule. Numbers in, decisions out (no data loading,
no MLflow), so each can be tested on its own. Used by ``monitor.py`` and ``replay.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PSI_STABLE, PSI_ALERT = 0.10, 0.25


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, top: int = 20) -> float:
    """Population Stability Index of ``actual`` against ``expected``.

    Numbers are cut at the deciles of ``expected``; categories keep the ``top`` most frequent of ``expected`` and pool the
    rest. Missing values are a bucket of their own. Rule of thumb: below 0.10 stable, 0.10-0.25 moderate, above 0.25 a
    real shift.
    """
    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
        e, a = expected.astype(float), actual.astype(float)
        edges = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)[1:-1])) if e.notna().any() else np.array([])

        def bucket(s: pd.Series) -> pd.Series:
            return pd.Series(np.where(s.isna(), -1, np.searchsorted(edges, s, side="right")), index=s.index)

        e_b, a_b = bucket(e), bucket(a)
    else:
        e_s, a_s = expected.astype("string").fillna("<NA>"), actual.astype("string").fillna("<NA>")
        keep = e_s.value_counts().index[:top]
        e_b, a_b = e_s.where(e_s.isin(keep), "<other>"), a_s.where(a_s.isin(keep), "<other>")
    e_p = e_b.value_counts(normalize=True)
    a_p = a_b.value_counts(normalize=True)
    idx = e_p.index.union(a_p.index)
    e_p = e_p.reindex(idx, fill_value=0).clip(lower=1e-4)
    a_p = a_p.reindex(idx, fill_value=0).clip(lower=1e-4)
    return float(((a_p - e_p) * np.log(a_p / e_p)).sum())


@dataclass(frozen=True)
class RetrainRules:
    label_delay_months: int = 3
    mae_tolerance: float = 0.10
    patience: int = 2


def breaches(month: dict, reference_mae: float, rules: RetrainRules, ceiling: float | None) -> list[str]:
    """What is wrong with one mature month (empty list: nothing)."""
    out = []
    limit = (1 + rules.mae_tolerance) * reference_mae
    if month["mae"] > limit:
        out.append(f"MAE {month['mae']:.2f} > {limit:.2f}")
    if ceiling is not None and abs(month["total_error_pct"]) > ceiling:
        out.append(f"monthly total error {month['total_error_pct']:+.1f}% beyond {ceiling:.0f}%")
    return out


def should_retrain(
    mature_months: list[dict], reference_mae: float, rules: RetrainRules, ceiling: float | None
) -> tuple[bool, str]:
    """Fire when the last ``patience`` mature months (chronological) all breach. Returns (retrain?, reason)."""
    recent = mature_months[-rules.patience :]
    if len(recent) < rules.patience:
        return False, f"only {len(mature_months)} mature month(s); the rule needs {rules.patience}"
    problems = {m["period"]: breaches(m, reference_mae, rules, ceiling) for m in recent}
    if all(problems.values()):
        detail = "; ".join(f"{p}: {', '.join(b)}" for p, b in problems.items())
        return True, f"{rules.patience} months in a row out of bounds ({detail})"
    return False, "within bounds"


def known_through(decision: pd.Period, rules: RetrainRules) -> pd.Period:
    """At the end of month ``decision``, the last admission month whose costs are (almost all) known."""
    return decision - rules.label_delay_months


def monthly_metrics(y: np.ndarray, pred: np.ndarray, period: pd.Series) -> list[dict]:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "pred": np.asarray(pred, dtype=float), "period": period.values})
    df["abs_err"] = (df["pred"] - df["y"]).abs()
    g = df.groupby("period", sort=True).agg(
        n=("y", "size"), actual=("y", "sum"), predicted=("pred", "sum"), mae=("abs_err", "mean")
    )
    g["total_error_pct"] = 100 * (g["predicted"] - g["actual"]) / g["actual"]
    return [{"period": str(p), **{k: float(v) for k, v in row.items()}} for p, row in g.iterrows()]


def pair_monthly_metrics(y: np.ndarray, typical: np.ndarray, expected: np.ndarray, period: pd.Series) -> list[dict]:
    """``monthly_metrics`` for a model that gives two numbers: the month's total (and its error) from the expected
    cost, the MAE from the typical cost."""
    months = monthly_metrics(y, expected, period)
    if expected is not typical:
        for month, per_admission in zip(months, monthly_metrics(y, typical, period), strict=True):
            month["mae"] = per_admission["mae"]
    return months


def error_by_group(y: np.ndarray, pred: np.ndarray, period: pd.Series, group: pd.Series) -> dict[str, dict[str, dict]]:
    """Per month and per group: admissions, paid and predicted totals, and the error on the total. {month: {group: ...}}"""
    df = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float),
            "pred": np.asarray(pred, dtype=float),
            "period": period.to_numpy(),
            "group": group.astype("string").fillna("NA").to_numpy(),
        }
    )
    g = df.groupby(["period", "group"], sort=True).agg(n=("y", "size"), actual=("y", "sum"), predicted=("pred", "sum"))
    g["total_error_pct"] = 100 * (g["predicted"] - g["actual"]) / g["actual"]
    out: dict[str, dict[str, dict]] = {}
    for (p, name), row in g.iterrows():
        out.setdefault(str(p), {})[str(name)] = {k: float(v) for k, v in row.items()}
    return out


def price_level(y: np.ndarray, period: pd.Series, code: pd.Series, reference_mean: pd.Series) -> list[dict]:
    """Per month: what was paid against what the same admissions would cost at the prices the model last saw.

    ``reference_mean`` is the mean paid per code in the last months of the model's data. Holding the mix of codes
    fixed, ``price_index_pct`` is how much dearer (+) or cheaper (-) the month is than that price level. Admissions whose
    code the model never saw have no reference price and are left out; ``covered_pct`` is the share of money kept.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "period": period.to_numpy()})
    df["ref"] = code.astype("string").map(reference_mean).to_numpy(dtype=float)
    out = []
    for p, g in df.groupby("period", sort=True):
        known = g["ref"].notna()
        paid, at_ref = float(g.loc[known, "y"].sum()), float(g.loc[known, "ref"].sum())
        out.append(
            {
                "period": str(p),
                "paid": paid,
                "at_reference_prices": at_ref,
                "price_index_pct": 100 * (paid / at_ref - 1) if at_ref else float("nan"),
                "covered_pct": 100 * paid / float(g["y"].sum()),
            }
        )
    return out


def price_shifts(y: np.ndarray, pred: np.ndarray, period: pd.Series, code: pd.Series, reference: pd.DataFrame,
                 top: int = 10) -> list[dict]:  # fmt: skip
    """The codes that cost most above the prices the model last saw, with the median paid for each month by month.

    ``reference`` has the ``mean`` and ``median`` paid per code in the last months of the model's data. ``extra`` is
    what was paid minus what the same number of admissions would cost at that mean. A code whose monthly median jumps
    is being paid differently than the model learned: a price change, which no input drift can show.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float),
            "pred": np.asarray(pred, dtype=float),
            "period": period.to_numpy(),
            "code": code.astype("string").to_numpy(),
        }
    )
    g = df.groupby("code").agg(n=("y", "size"), paid=("y", "sum"), predicted=("pred", "sum"))
    g = g.join(reference[["mean", "median"]], how="inner")  # codes the model saw
    g["at_reference_prices"] = g["n"] * g["mean"]
    g["extra"] = g["paid"] - g["at_reference_prices"]
    monthly = df.groupby(["code", "period"])["y"].median()
    return [
        {
            "code": str(name),
            "n": int(row["n"]),
            "paid": float(row["paid"]),
            "at_reference_prices": float(row["at_reference_prices"]),
            "extra": float(row["extra"]),
            "predicted": float(row["predicted"]),
            "reference_median": float(row["median"]),
            "monthly_median": {str(p): float(v) for p, v in monthly[name].items()},
        }
        for name, row in g.sort_values("extra", ascending=False).head(top).iterrows()
    ]
