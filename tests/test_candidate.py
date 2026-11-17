"""The LightGBM candidate, on synthetic admissions whose cost is driven by the procedure price."""

import pickle

import numpy as np
import pytest

from costmodel import features as F
from costmodel.baselines import GlobalMean
from costmodel.candidate import CostModel
from costmodel.synthetic import generate


@pytest.fixture(scope="module")
def data():
    df = generate(12_000, "2022-01-01", "2023-12-31", seed=1)
    train, valid = df[df["admission_date"] < "2023-07-01"], df[df["admission_date"] >= "2023-07-01"]
    return F.build_features(train), train[F.TARGET], F.build_features(valid), valid[F.TARGET]


def small(objective: str) -> CostModel:
    return CostModel(objective, num_leaves=15, learning_rate=0.1, n_estimators=300, min_category_count=5)


@pytest.mark.parametrize("objective", ["tweedie", "gamma", "log_l2"])
def test_beats_global_mean_and_gets_totals_roughly_right(data, objective):
    X, y, Xv, yv = data
    model = small(objective).fit(X, y, Xv, yv, early_stopping_rounds=20)
    pred = model.predict(Xv)
    assert np.isfinite(pred).all() and (pred > 0).all()
    floor = np.abs(GlobalMean().fit(X, y).predict(Xv) - yv).mean()
    assert np.abs(pred - yv).mean() < 0.8 * floor
    # The synthetic price rises 6% a year and trees cannot extrapolate, so allow some under-prediction.
    assert abs(pred.sum() / yv.sum() - 1) < 0.12


def test_early_stopping_picks_fewer_trees(data):
    X, y, Xv, yv = data
    model = CostModel("tweedie", 15, 0.3, 2000, 5).fit(X, y, Xv, yv, early_stopping_rounds=10)
    assert model.best_iteration_ < 2000


def test_unseen_and_rare_codes_fall_back_instead_of_failing(data):
    X, y, Xv, _ = data
    model = small("tweedie").fit(X, y)
    new = Xv.head(5).assign(requested_procedure="9999999999", provider_id="NEVER_SEEN")
    assert np.isfinite(model.predict(new)).all()
    assert "9999999999" not in model.categories_["requested_procedure"]


def test_survives_pickling(data):
    X, y, Xv, _ = data
    model = small("log_l2").fit(X, y)
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(again.predict(Xv), model.predict(Xv))


def test_feature_importance_names_every_feature(data):
    X, y, _, _ = data
    model = small("tweedie").fit(X, y)
    imp = model.feature_importance()
    assert set(imp.index) == set(model.columns) == set(F.FEATURES) - {"admission_period"}
    assert imp.index[0] in {"requested_procedure", "procedure_prefix2", "procedure_prefix4", "diagnosis_icd10"}


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError):
        CostModel("poisson")


def test_time_feature_is_used_only_when_asked(data):
    X, y, Xv, _ = data
    without, with_time = small("tweedie").fit(X, y), CostModel("tweedie", 15, 0.1, 300, 5, use_time=True).fit(X, y)
    assert "admission_period" not in without.feature_importance().index
    assert "admission_period" in with_time.feature_importance().index
    # Synthetic prices rise 6% a year; with the date, the model carries the latest level forward instead of the average.
    assert with_time.predict(Xv).mean() > without.predict(Xv).mean()
