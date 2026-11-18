"""The candidate model: gradient-boosted trees (LightGBM) on the admission-time features.

Three ways to fit a right-skewed, strictly positive cost, chosen per run in the config:

* ``tweedie`` and ``gamma``: predict the mean cost directly (log link), so monthly totals come out unbiased.
* ``log_l2``: least squares on log(cost). Exponentiating gives the *median*; Duan's smearing factor rescales it towards
  the mean, as in the log-linear baseline.

With ``use_time``, the model also sees ``admission_period`` (months elapsed). Trees cannot extrapolate, so for a future
date they apply the most recent level they saw, which is what a price that keeps rising calls for. Whether that helps
is decided on validation, like any other setting.

Category codes (procedure, ICD-10, hospital...) are learned from the training data only. A code seen fewer than
``min_category_count`` times, or never, is treated as missing, so the model falls back on the coarser columns
(procedure group, ICD-10 chapter) that ``features.build_features`` already provides.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from costmodel import features as F

OBJECTIVES = ("tweedie", "gamma", "log_l2")


class CostModel:
    def __init__(
        self,
        objective: str = "tweedie",
        num_leaves: int = 63,
        learning_rate: float = 0.1,
        n_estimators: int = 2000,
        min_category_count: int = 20,
        seed: int = 42,
        use_time: bool = False,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.objective = objective
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_category_count = min_category_count
        self.seed = seed
        self.use_time = use_time

    @property
    def name(self) -> str:
        return f"lightgbm_{self.objective}"

    def params(self) -> dict:
        return {
            "objective": self.objective,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_category_count": self.min_category_count,
            "seed": self.seed,
            "use_time": self.use_time,
        }

    @property
    def columns(self) -> list[str]:
        return [c for c in F.FEATURES if self.use_time or c != "admission_period"]

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in F.CATEGORICAL:
            values = X[col].astype("string")
            known = values.isin(self.categories_[col])  # rare or unseen codes become missing
            out[col] = pd.Categorical(values.where(known), categories=self.categories_[col])
        for col in F.NUMERIC:
            if col in self.columns:
                out[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        return out

    def _target(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.log(y) if self.objective == "log_l2" else y

    def _booster(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective="regression" if self.objective == "log_l2" else self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=1.0,
            cat_smooth=10,
            min_data_per_group=100,
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y,
        X_valid: pd.DataFrame | None = None,
        y_valid=None,
        early_stopping_rounds: int | None = None,
    ) -> CostModel:
        """Fit on features from ``features.build_features``. With a validation set, stop when it stops improving."""
        self.categories_ = {}
        for col in F.CATEGORICAL:
            counts = X[col].astype("string").value_counts()
            self.categories_[col] = sorted(counts[counts >= self.min_category_count].index)
        self.booster_ = self._booster()
        Xe = self._encode(X)
        fit_kwargs = {}
        if X_valid is not None and early_stopping_rounds:
            fit_kwargs = {
                "eval_X": self._encode(X_valid),
                "eval_y": self._target(y_valid),
                "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
            }
        self.booster_.fit(Xe, self._target(y), **fit_kwargs)
        self.best_iteration_ = int(self.booster_.best_iteration_ or self.booster_.n_estimators_)
        self.smearing_ = 1.0
        if self.objective == "log_l2":
            residuals = self._target(y) - self.booster_.predict(Xe, num_iteration=self.best_iteration_)
            self.smearing_ = float(np.mean(np.exp(residuals)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster_.predict(self._encode(X), num_iteration=self.best_iteration_)
        return np.exp(raw) * self.smearing_ if self.objective == "log_l2" else raw

    def feature_importance(self) -> pd.Series:
        """Total gain per feature: how much each column reduced the loss over all trees."""
        gain = self.booster_.booster_.feature_importance(importance_type="gain", iteration=self.best_iteration_)
        return pd.Series(gain, index=self.columns).sort_values(ascending=False)
