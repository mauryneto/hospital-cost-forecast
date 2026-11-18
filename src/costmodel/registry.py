"""Everything that talks to MLflow: where runs are stored, how a model is packaged, and the registry aliases.

* **Tracking:** a SQLite file, ``mlflow.db`` in the work folder (``$MLFLOW_TRACKING_URI`` overrides it). Model files go to
  ``mlartifacts/`` next to it. Both are ignored by Git.
* **Packaging:** the registered model takes the *raw admission columns* (as in the contract) and builds the features
  itself, so whoever calls it (the API, a batch job) cannot build them differently from training. A candidate gives
  two numbers per admission, the typical cost and the expected cost (see ``candidate.CostPair``); a baseline, or a
  version registered before that, gives one, which then counts as both.
* **Aliases:** ``staging`` points at the latest trained candidate; ``production`` at the version that passed the gate.
  Moving an alias is the deploy; moving it back is the rollback. Versions are never deleted.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from costmodel import features as F
from costmodel.candidate import EXPECTED, TYPICAL
from costmodel.config import Config
from costmodel.paths import HOME

STAGING = "staging"
PRODUCTION = "production"

logging.getLogger("mlflow").setLevel(logging.ERROR)


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(HOME / 'mlflow.db').as_posix()}"


def setup(cfg: Config) -> str:
    """Point MLflow at the tracking database and select (or create) the dataset's experiment. Returns its id."""
    mlflow.set_tracking_uri(tracking_uri())
    exp = mlflow.get_experiment_by_name(cfg.name)
    if exp is None:
        exp_id = mlflow.create_experiment(cfg.name, artifact_location=(HOME / "mlartifacts" / cfg.name).as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


class CostModelPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: raw admission columns in; out, the typical and the expected cost (or one cost, from a model
    that gives a single number)."""

    def load_context(self, context):
        with open(context.artifacts["cost_model"], "rb") as f:
            self.model = pickle.load(f)

    # The type hint keeps MLflow quiet (it warns on every import without one); the input schema itself is the explicit
    # signature set in ``log_model``.
    def predict(self, context, model_input: pd.DataFrame, params=None):
        X = F.build_features(pd.DataFrame(model_input))
        both = getattr(self.model, "predict_both", None)
        return both(X).reset_index(drop=True) if both is not None else self.model.predict(X)


def unwrap(loaded) -> object:
    """The project's own model object inside a model loaded from the registry."""
    return loaded.unwrap_python_model().model


def input_columns(df: pd.DataFrame) -> list[str]:
    """The contract's admission-time columns present in this dataset (optional ones may be missing)."""
    return [c for c in F.ALLOWED_INPUTS if c in df.columns]


def log_model(cfg: Config, model, example: pd.DataFrame) -> str:
    """Log ``model`` in the active run and register it as a new version. Returns the version number."""
    example = example[input_columns(example)].head(200)
    X = F.build_features(example)
    output = model.predict_both(X).reset_index(drop=True) if hasattr(model, "predict_both") else model.predict(X)
    signature = infer_signature(example, output)
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = Path(tmp) / "cost_model.pkl"
        path.write_bytes(pickle.dumps(model))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CostModelPyfunc(),
            artifacts={"cost_model": str(path)},
            signature=signature,
            input_example=example.head(5),
            registered_model_name=cfg.model_name,
        )
    return str(info.registered_model_version)


def client() -> mlflow.MlflowClient:
    return mlflow.MlflowClient(tracking_uri=tracking_uri())


def version_for(cfg: Config, alias: str):
    """The model version an alias points at, or None."""
    try:
        return client().get_model_version_by_alias(cfg.model_name, alias)
    except MlflowException:
        return None


def set_alias(cfg: Config, alias: str, version: str) -> None:
    client().set_registered_model_alias(cfg.model_name, alias, version)


def tag_version(cfg: Config, version: str, tags: dict[str, str]) -> None:
    for key, value in tags.items():
        client().set_model_version_tag(cfg.model_name, version, key, value)


def load(cfg: Config, alias: str):
    """Load the model an alias points at, or None if the alias is not set."""
    if version_for(cfg, alias) is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlflow.pyfunc.load_model(f"models:/{cfg.model_name}@{alias}")


def load_version(cfg: Config, version: str):
    """Load one exact version. Unlike ``load``, the result cannot change if an alias moves meanwhile."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlflow.pyfunc.load_model(f"models:/{cfg.model_name}/{version}")


def predict_both(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score admissions with a loaded registry model, passing only the columns it was registered with. Returns the
    typical and the expected cost; a model that gives a single number gives it for both."""
    out = model.predict(df[input_columns(df)])
    if isinstance(out, pd.DataFrame):
        return out[TYPICAL].to_numpy(dtype=float), out[EXPECTED].to_numpy(dtype=float)
    out = np.asarray(out, dtype=float)
    return out, out


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """The typical cost of each admission (one number per admission)."""
    return predict_both(model, df)[0]
