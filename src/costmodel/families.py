"""``costmodel families``: is gradient boosting with LightGBM the right kind of model here?

Other families are fitted on the same training window and measured on the same validation window, with the same
features and the same two metrics as the candidate search (``costmodel train``):

* **Random forest**: many deep trees fitted independently on random samples of rows and columns, then averaged. Here
  it is LightGBM's ``rf`` mode, so categories are handled exactly as in the candidate.
* **CatBoost**: gradient boosting built for columns with many categories (it encodes each category with statistics of
  the target instead of splitting on the raw codes). Optional: ``uv sync --extra families``.

This is a comparison, not a new search: the LightGBM row is the setting ``costmodel train`` chose among all the ones
it tried, while each other family gets two settings. It never touches the test window or the registry. A family that
won here by a clear margin would become a candidate and go through the gate like any other.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from costmodel import features as F
from costmodel import registry
from costmodel.candidate import CostModel
from costmodel.config import Config
from costmodel.evaluate import summary, to_markdown
from costmodel.train import choose, load_splits

FOREST_TREES = 300
FOREST_LEAVES = 4095  # forests want deep trees; each tree may use up to this many leaves
CATBOOST_DEPTH = 8
CATBOOST_MAX_TREES = 1000  # CatBoost is slow on many high-cardinality columns; this bounds the run


class ForestModel(CostModel):
    """Random forest on the same features: LightGBM's ``rf`` mode, on log(cost) or on the cost itself."""

    def __init__(self, log_target: bool = True, n_trees: int = FOREST_TREES, min_category_count: int = 20, seed: int = 42,
                 use_time: bool = True):  # fmt: skip
        super().__init__("log_l2", FOREST_LEAVES, 1.0, n_trees, min_category_count, seed, use_time)
        # In rf mode only least squares gives a true forest (an average of trees fitted to the target itself).
        self.objective = "log_l2" if log_target else "l2"

    @property
    def name(self) -> str:
        return f"random_forest_{self.objective}"

    def fit(self, X: pd.DataFrame, y, X_valid=None, y_valid=None, early_stopping_rounds=None) -> ForestModel:
        return super().fit(X, y)  # a forest has no rounds to stop early: every tree is independent

    def _booster(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            boosting_type="rf",
            objective="regression",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            min_child_samples=20,
            subsample=0.632,  # each tree sees a random 63% of the rows...
            subsample_freq=1,
            colsample_bynode=0.5,  # ...and each split a random half of the columns
            cat_smooth=10,
            min_data_per_group=100,
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )


class CatBoostModel:
    """CatBoost on the same features. ``loss`` is ``log_rmse`` (least squares on log(cost), smeared) or ``tweedie``."""

    def __init__(self, loss: str = "log_rmse", learning_rate: float = 0.1, iterations: int = 2000, seed: int = 42,
                 use_time: bool = True):  # fmt: skip
        if loss not in ("log_rmse", "tweedie"):
            raise ValueError("loss must be 'log_rmse' or 'tweedie'")
        self.loss, self.learning_rate, self.iterations = loss, learning_rate, iterations
        self.seed, self.use_time = seed, use_time

    @property
    def name(self) -> str:
        return f"catboost_{self.loss}"

    @property
    def columns(self) -> list[str]:
        return [c for c in F.FEATURES if self.use_time or c != "admission_period"]

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.columns].copy()
        for col in F.CATEGORICAL:
            out[col] = out[col].astype("string").fillna("NA").astype(object)
        return out

    def _target(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.log(y) if self.loss == "log_rmse" else y

    def fit(self, X: pd.DataFrame, y, X_valid: pd.DataFrame | None = None, y_valid=None,
            early_stopping_rounds: int | None = None) -> CatBoostModel:  # fmt: skip
        from catboost import CatBoostRegressor, Pool

        self.model_ = CatBoostRegressor(
            loss_function="RMSE" if self.loss == "log_rmse" else "Tweedie:variance_power=1.5",
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=CATBOOST_DEPTH,
            random_seed=self.seed,
            allow_writing_files=False,
            verbose=False,
        )
        train = Pool(self._frame(X), self._target(y), cat_features=F.CATEGORICAL)
        kwargs = {}
        if X_valid is not None and early_stopping_rounds:
            kwargs = {
                "eval_set": Pool(self._frame(X_valid), self._target(y_valid), cat_features=F.CATEGORICAL),
                "early_stopping_rounds": early_stopping_rounds,
                "use_best_model": True,
            }
        self.model_.fit(train, **kwargs)
        self.best_iteration_ = int(self.model_.tree_count_)
        self.smearing_ = 1.0
        if self.loss == "log_rmse":
            self.smearing_ = float(np.mean(np.exp(self._target(y) - self._raw(train))))
        return self

    def _raw(self, data) -> np.ndarray:
        """The model's raw output, which is on the log scale for both losses (asked for explicitly: the default of
        ``predict`` differs between loss functions)."""
        return np.asarray(self.model_.predict(data, prediction_type="RawFormulaVal"), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self._raw(self._frame(X))) * self.smearing_


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except ImportError:
        return False
    return True


def contenders(cfg: Config) -> list[tuple[str, str, object]]:
    """(family, setting, model) for every alternative to try."""
    tc = cfg.train
    out: list[tuple[str, str, object]] = [
        ("random forest", "on log(cost)", ForestModel(True, FOREST_TREES, tc.min_category_count, tc.seed)),
        ("random forest", "on the cost", ForestModel(False, FOREST_TREES, tc.min_category_count, tc.seed)),
    ]
    if catboost_available():
        rounds = min(tc.max_rounds, CATBOOST_MAX_TREES)
        out += [
            ("CatBoost", "log(cost), least squares", CatBoostModel("log_rmse", tc.learning_rate, rounds, tc.seed)),
            ("CatBoost", "Tweedie", CatBoostModel("tweedie", tc.learning_rate, rounds, tc.seed)),
        ]
    return out


def run(cfg: Config) -> pd.DataFrame:
    searched = cfg.reports_dir / "train_valid.json"
    if not searched.exists():
        raise RuntimeError(f"{searched} not found; run `costmodel train` first (its chosen setting is the LightGBM row)")
    search = pd.DataFrame(json.loads(searched.read_text(encoding="utf-8")))
    best = choose(search, cfg)
    rows = [
        {
            "family": "LightGBM (gradient boosting)",
            "setting": f"{best['objective']}, {int(best['num_leaves'])} leaves, {best['window']}"
            f"{', with time' if best['use_time'] else ''}; chosen among {len(search)} settings",
            "trees": int(best["trees"]),
            **{k: float(best[k]) for k in ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")},
            "seconds": float(best["seconds"]),
        }
    ]

    df = load_splits(cfg)
    train, valid = df[df["split"] == "train"], df[df["split"] == "valid"]
    X, y = F.build_features(train), train[F.TARGET]
    X_valid, y_valid = F.build_features(valid), valid[F.TARGET]
    registry.setup(cfg)
    with mlflow.start_run(run_name="model families"):
        mlflow.set_tags({"step": "families", "dataset": cfg.name})
        for family, setting, model in contenders(cfg):
            t0 = time.perf_counter()
            with mlflow.start_run(run_name=f"{family}: {setting}", nested=True):
                model.fit(X, y, X_valid, y_valid, cfg.train.early_stopping_rounds)
                metrics = summary(y_valid.to_numpy(), model.predict(X_valid), valid["period"])
                seconds = round(time.perf_counter() - t0, 1)
                mlflow.log_params({"family": family, "setting": setting, "model": model.name})
                mlflow.log_metrics({f"valid_{k}": v for k, v in metrics.items()} | {"trees": model.best_iteration_})
            keep = ("mae", "rmse", "mae_top_decile", "bias_pct", "monthly_total_mape")
            rows.append({"family": family, "setting": setting, "trees": model.best_iteration_,
                         **{k: metrics[k] for k in keep}, "seconds": seconds})  # fmt: skip
            print(f"{family:<14} {setting:<26} trees {model.best_iteration_:<5} MAE {metrics['mae']:9.2f}"
                  f"  monthly total err {metrics['monthly_total_mape']:5.2f}%  ({seconds}s)")  # fmt: skip
    results = pd.DataFrame(rows)
    write_report(cfg, results, len(train), len(valid))
    return results


def write_report(cfg: Config, results: pd.DataFrame, n_train: int, n_valid: int) -> None:
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(results.to_json(orient="records"))
    (out / "families_valid.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8", newline="\n")
    best = choose(results, cfg)
    ceiling = cfg.gate.max_monthly_total_mape
    lines = [
        f"# Model families on the validation window ({cfg.name})",
        "",
        f"Generated by `costmodel families --config {cfg.path.as_posix() if cfg.path else ''}`. Every row was fitted on the"
        f" same {n_train:,} training admissions and measured on the same {n_valid:,} validation admissions, with the same"
        " features. The LightGBM row is the setting `costmodel train` chose; each other family got two settings, so they"
        " are less tuned. Boosted models stop adding trees when validation stops improving (CatBoost at"
        f" {CATBOOST_MAX_TREES:,} trees at most); the forest has a fixed {FOREST_TREES} trees.",
        "",
        to_markdown(results.sort_values("mae"), index=False),
        "",
        "**Best by the search's own rule** (lowest MAE"
        + (f" among rows with monthly total error at most {ceiling:.0f}%" if ceiling is not None else "")
        + f"): {best['family']}, {best['setting']}.",
    ]
    if not catboost_available():
        lines += ["", "CatBoost is not installed (`uv sync --extra families`), so it is missing from this comparison."]
    (out / "families_valid.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
