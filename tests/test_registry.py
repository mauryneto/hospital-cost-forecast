"""Register a model in a throwaway MLflow database, move aliases, load it back by alias."""

import numpy as np
import pytest

from costmodel import features as F
from costmodel import registry
from costmodel.candidate import CostModel, CostPair
from costmodel.config import Config
from costmodel.synthetic import generate


@pytest.mark.slow
def test_register_alias_and_load_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    cfg = Config(name="registry_test", source={"type": "file"})
    df = generate(3_000, "2023-01-01", "2023-12-31", seed=3)
    model = CostModel("tweedie", 7, 0.1, 50, 5).fit(F.build_features(df), df[F.TARGET])

    registry.setup(cfg)
    assert registry.load(cfg, registry.PRODUCTION) is None
    with registry.mlflow.start_run():
        v1 = registry.log_model(cfg, model, df)
    registry.set_alias(cfg, registry.STAGING, v1)
    registry.tag_version(cfg, v1, {"gate": "passed"})

    loaded = registry.load(cfg, registry.STAGING)
    # The registered model takes raw contract rows, with post-admission columns present, and builds its own features.
    np.testing.assert_allclose(registry.predict(loaded, df), model.predict(F.build_features(df)))
    assert registry.version_for(cfg, registry.STAGING).tags["gate"] == "passed"

    with registry.mlflow.start_run():
        v2 = registry.log_model(cfg, model, df)
    registry.set_alias(cfg, registry.PRODUCTION, v1)
    registry.set_alias(cfg, registry.STAGING, v2)
    assert str(registry.version_for(cfg, registry.PRODUCTION).version) == v1
    assert str(registry.version_for(cfg, registry.STAGING).version) == v2

    # A candidate with two numbers per admission is one version; a model with one number gives it for both.
    X = F.build_features(df)
    pair = CostPair(CostModel("log_l2", 7, 0.1, 50, 5), CostModel("tweedie", 7, 0.1, 50, 5)).fit(X, df[F.TARGET])
    with registry.mlflow.start_run():
        v3 = registry.log_model(cfg, pair, df)
    typical, expected = registry.predict_both(registry.load_version(cfg, v3), df)
    np.testing.assert_allclose(typical, pair.typical.predict(X))
    np.testing.assert_allclose(expected, pair.expected.predict(X))
    np.testing.assert_allclose(registry.predict(registry.load_version(cfg, v3), df), typical)
    single_typical, single_expected = registry.predict_both(registry.load_version(cfg, v1), df)
    np.testing.assert_array_equal(single_typical, single_expected)
