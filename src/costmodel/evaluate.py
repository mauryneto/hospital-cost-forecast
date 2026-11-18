"""Metrics, read the way a payer would read them.

* MAE / RMSE per admission.
* MAE inside each decile of the *actual* cost: being wrong on the expensive admissions is what hurts.
* Error on the month's total cost (sum predicted vs. sum paid). This is how the forecast would be used to provision money,
  and it rewards getting the mean right, which per-admission MAE does not (a median minimises MAE but underestimates
  totals on a right-skewed target).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_ae": float(np.median(np.abs(err))),
        "bias_pct": float(100 * err.sum() / y.sum()),
    }


def decile_table(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Per decile of actual cost: mean actual, mean predicted (calibration) and MAE."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "pred": np.asarray(pred, dtype=float)})
    df["decile"] = pd.qcut(df["y"].rank(method="first"), 10, labels=range(1, 11))
    df["abs_err"] = (df["pred"] - df["y"]).abs()
    return df.groupby("decile", observed=True).agg(
        mean_actual=("y", "mean"), mean_pred=("pred", "mean"), mae=("abs_err", "mean"), n=("y", "size")
    )


def monthly_totals(y: np.ndarray, pred: np.ndarray, period: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "pred": np.asarray(pred, dtype=float), "period": period.values})
    out = df.groupby("period").agg(actual=("y", "sum"), predicted=("pred", "sum"), n=("y", "size"))
    out["error_pct"] = 100 * (out["predicted"] - out["actual"]) / out["actual"]
    return out


def segment_mae(y: np.ndarray, pred: np.ndarray, segment: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "pred": np.asarray(pred, dtype=float), "seg": segment.values})
    df["abs_err"] = (df["pred"] - df["y"]).abs()
    return df.groupby("seg", observed=True).agg(mae=("abs_err", "mean"), mean_actual=("y", "mean"), n=("y", "size"))


def to_markdown(df: pd.DataFrame, **kwargs) -> str:
    """Markdown table that keeps counts as integers (pandas would print them as 1234.00 next to float columns)."""
    ints = {c: object for c in df.select_dtypes("integer").columns}
    return df.astype(ints).to_markdown(floatfmt=".2f", **kwargs)


def summary(y: np.ndarray, pred: np.ndarray, period: pd.Series) -> dict[str, float]:
    out = regression_metrics(y, pred)
    deciles = decile_table(y, pred)
    out["mae_top_decile"] = float(deciles.loc[10, "mae"])
    months = monthly_totals(y, pred, period)
    out["monthly_total_mape"] = float(months["error_pct"].abs().mean())
    return out


def pair_summary(y: np.ndarray, typical: np.ndarray, expected: np.ndarray, period: pd.Series) -> dict[str, float]:
    """``summary`` for a model that gives two numbers, each judged by what it is for: the per-admission metrics come
    from the typical cost, the bias and the error on the month's total from the expected cost."""
    out = summary(y, typical, period)
    if expected is not typical:
        totals = summary(y, expected, period)
        out |= {"bias_pct": totals["bias_pct"], "monthly_total_mape": totals["monthly_total_mape"]}
    return out
