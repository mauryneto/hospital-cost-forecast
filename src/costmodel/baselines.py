"""Baselines, from the floor to the hard one.

On SUS, what a hospital is paid follows the SIGTAP price table of the procedure, so "median cost of the requested
procedure" is close to the business rule itself (private plans price by procedure too, through contract tables). That is
the baseline any model has to beat to justify itself.

All baselines take the output of ``features.build_features`` and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from costmodel import features as F

AGE_BANDS = [-1, 0, 4, 14, 29, 44, 59, 74, 200]
AGE_LABELS = ["<1", "1-4", "5-14", "15-29", "30-44", "45-59", "60-74", "75+"]


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, labels=AGE_LABELS).astype("string").fillna("NA")


class GlobalMean:
    name = "global_mean"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GlobalMean:
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value_)


class GroupStat:
    """Median (or mean) cost per group, backing off to coarser groups when a group is rare or unseen.

    ``levels`` goes from the finest grouping to the coarsest; the global statistic is the last resort. A group is used
    only if it had at least ``min_count`` training admissions, so a procedure seen twice does not set its own price.

    The median minimises the error per admission; the mean is what makes the month's *total* come out right on a
    right-skewed target. Both are reported because a payer cares about both.
    """

    def __init__(self, name: str, levels: list[list[str]], stat: str = "median", min_count: int = 30):
        if stat not in ("median", "mean"):
            raise ValueError("stat must be 'median' or 'mean'")
        self.name = name
        self.levels = levels
        self.stat = stat
        self.min_count = min_count

    @staticmethod
    def _frame(X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(age_band=age_band(X["age_years"]))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GroupStat:
        df = self._frame(X).assign(_y=np.asarray(y, dtype=float))
        self.tables_ = []
        for keys in self.levels:
            stats = df.groupby(keys, observed=True)["_y"].agg([self.stat, "size"])
            self.tables_.append(stats.loc[stats["size"] >= self.min_count, self.stat].rename("_pred").reset_index())
        self.global_ = float(df["_y"].agg(self.stat))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df = self._frame(X).reset_index(drop=True)
        pred = pd.Series(np.nan, index=df.index)
        self.coverage_ = {}
        for keys, table in zip(self.levels, self.tables_, strict=True):
            found = df[keys].merge(table, on=keys, how="left")["_pred"]
            newly = pred.isna() & found.notna()
            self.coverage_["+".join(keys)] = float(newly.mean())
            pred = pred.fillna(found)
        self.coverage_["global"] = float(pred.isna().mean())
        return pred.fillna(self.global_).to_numpy()


class LogLinear:
    """Ridge regression on log(cost), one-hot categories plus an age spline.

    Exponentiating a log-scale prediction estimates the *median*, not the mean; Duan's smearing factor (the mean of
    exp(residuals) on training data) rescales it back towards the mean so monthly totals are not biased low.
    """

    name = "log_linear"

    def __init__(self, alpha: float = 1.0, min_frequency: int = 30):
        self.alpha = alpha
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogLinear:
        prep = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=self.min_frequency),
                    F.CATEGORICAL,
                ),
                ("age", SplineTransformer(n_knots=8, degree=3), ["age_years"]),
                ("month", OneHotEncoder(handle_unknown="ignore"), ["admission_month"]),
            ],
            sparse_threshold=1.0,
        )
        self.pipe_ = Pipeline([("prep", prep), ("ridge", Ridge(alpha=self.alpha, solver="sparse_cg"))])
        X = X.assign(age_years=X["age_years"].fillna(X["age_years"].median()))
        log_y = np.log(np.asarray(y, dtype=float))
        self.pipe_.fit(X, log_y)
        self.smearing_ = float(np.mean(np.exp(log_y - self.pipe_.predict(X))))
        self.age_fill_ = float(X["age_years"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.assign(age_years=X["age_years"].fillna(self.age_fill_))
        return np.exp(self.pipe_.predict(X)) * self.smearing_


PROC = [["requested_procedure"], ["procedure_prefix4"]]
PROC_AGE_TYPE = [["requested_procedure", "age_band", "admission_type"], *PROC]


def all_baselines() -> list:
    return [
        GlobalMean(),
        GroupStat("median_by_proc", PROC, "median"),
        GroupStat("median_by_proc_age_type", PROC_AGE_TYPE, "median"),
        GroupStat("mean_by_proc", PROC, "mean"),
        GroupStat("mean_by_proc_age_type", PROC_AGE_TYPE, "mean"),
        LogLinear(),
    ]
