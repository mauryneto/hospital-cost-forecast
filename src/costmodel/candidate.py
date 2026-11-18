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
        sample_weight=None,
    ) -> CostModel:
        """Fit on features from ``features.build_features``. With a validation set, stop when it stops improving.

        ``sample_weight`` makes some admissions count more than others (e.g. recent ones); the default is all equal.
        """
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
        self.booster_.fit(Xe, self._target(y), sample_weight=sample_weight, **fit_kwargs)
        self.best_iteration_ = int(self.booster_.best_iteration_ or self.booster_.n_estimators_)
        self.smearing_ = 1.0
        if self.objective == "log_l2":
            residuals = self._target(y) - self.booster_.predict(Xe, num_iteration=self.best_iteration_)
            self.smearing_ = float(np.average(np.exp(residuals), weights=sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster_.predict(self._encode(X), num_iteration=self.best_iteration_)
        return np.exp(raw) * self.smearing_ if self.objective == "log_l2" else raw

    def feature_importance(self) -> pd.Series:
        """Total gain per feature: how much each column reduced the loss over all trees."""
        gain = self.booster_.booster_.feature_importance(importance_type="gain", iteration=self.best_iteration_)
        return pd.Series(gain, index=self.columns).sort_values(ascending=False)


TYPICAL, EXPECTED = "typical_cost", "expected_cost"


class CostPair:
    """Two fitted models behind one registered version, one for each question a cost prediction answers.

    * ``typical``: what this admission should cost. Chosen for the lowest error per admission.
    * ``expected``: the number to add up. Chosen for the lowest error on the month's total.

    On a right-skewed cost no single number is best at both. The one closest to each admission sits near the typical
    case; the one that sums right sits near the mean, which the few expensive admissions pull up. When the search
    finds one setting that is best at both, the two are the same model.
    """

    def __init__(self, typical: CostModel, expected: CostModel):
        self.typical, self.expected = typical, expected

    @property
    def name(self) -> str:
        return self.typical.name if self.expected is self.typical else f"{self.typical.name}+{self.expected.name}"

    def fit(self, X: pd.DataFrame, y) -> CostPair:
        self.typical.fit(X, y)
        if self.expected is not self.typical:
            self.expected.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """The typical cost: what code that wants one number per admission gets."""
        return self.typical.predict(X)

    def predict_both(self, X: pd.DataFrame) -> pd.DataFrame:
        typical = self.typical.predict(X)
        expected = typical if self.expected is self.typical else self.expected.predict(X)
        return pd.DataFrame({TYPICAL: typical, EXPECTED: expected}, index=X.index)


def predict_both(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(typical, expected) from any model. One that gives a single number (a baseline, an older version) gives it
    for both."""
    if hasattr(model, "predict_both"):
        both = model.predict_both(X)
        return both[TYPICAL].to_numpy(dtype=float), both[EXPECTED].to_numpy(dtype=float)
    pred = np.asarray(model.predict(X), dtype=float)
    return pred, pred
