"""The API and the batch job: input checks, and that both price an admission exactly like the registered model."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from costmodel import config as config_module
from costmodel import contract, registry
from costmodel import features as F
from costmodel.api import MAX_ADMISSIONS, Admission, create_app
from costmodel.candidate import EXPECTED, TYPICAL, CostModel, CostPair
from costmodel.config import Config
from costmodel.score import Scorer, run_batch
from costmodel.synthetic import generate

CFG = Config(name="serving_test", source={"type": "file"})


def rows(df: pd.DataFrame) -> list[dict]:
    """Admissions as an API client would send them: admission-time fields only, dates as text."""
    out = df[[c for c in contract.ADMISSION_TIME if c in df.columns]].copy()
    out["admission_date"] = out["admission_date"].dt.strftime("%Y-%m-%d")
    return out.astype(object).where(out.notna(), None).to_dict(orient="records")


class Doubler:
    """Stands in for a registry model: predicts twice the age, so the order of the answers can be checked."""

    def predict(self, df):
        return 2 * df["age_years"].astype(float).to_numpy()


@pytest.fixture
def client():
    scorer = Scorer("cost-serving_test", "7", Doubler(), {"test_mae": "123.4", "fitted_on": "2023-01-01 to 2023-12-31"})
    with TestClient(create_app(CFG, scorer)) as c:
        yield c


def test_request_schema_is_the_contract_admission_time_columns():
    assert list(Admission.model_fields) == contract.ADMISSION_TIME


def test_validate_inputs_needs_no_cost_and_still_checks_the_rest():
    df = generate(50, "2024-01-01", "2024-01-31", seed=1)
    contract.validate_inputs(df.drop(columns=[F.TARGET]))  # not billed yet: fine
    with pytest.raises(contract.ContractError, match="sex"):
        contract.validate_inputs(df.assign(sex="X"))


def test_health_reports_the_loaded_version(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model": "cost-serving_test",
        "model_version": "7",
        "fitted_on": "2023-01-01 to 2023-12-31",
        "test_mae": 123.4,
    }


def test_predict_keeps_the_order_and_names_the_version(client):
    admissions = rows(generate(5, "2024-01-01", "2024-01-31", seed=2))
    body = client.post("/predict", json={"admissions": admissions}).json()
    assert body["model_version"] == "7"
    assert [p["admission_id"] for p in body["predictions"]] == [a["admission_id"] for a in admissions]
    # A model that gives one number per admission (a baseline, an older version): it counts as both.
    assert [p["typical_cost"] for p in body["predictions"]] == [2 * a["age_years"] for a in admissions]
    assert [p["expected_cost"] for p in body["predictions"]] == [2 * a["age_years"] for a in admissions]
    assert set(body["predictions"][0]) == {"admission_id", "typical_cost", "expected_cost"}


def test_predict_returns_the_two_numbers_of_a_model_that_has_them():
    class TwoNumbers:
        def predict(self, df):
            age = df["age_years"].astype(float).to_numpy()
            return pd.DataFrame({TYPICAL: 2 * age, EXPECTED: 3 * age})

    admissions = rows(generate(5, "2024-01-01", "2024-01-31", seed=2))
    with TestClient(create_app(CFG, Scorer("cost-serving_test", "8", TwoNumbers()))) as c:
        body = c.post("/predict", json={"admissions": admissions}).json()
    assert [p["typical_cost"] for p in body["predictions"]] == [2 * a["age_years"] for a in admissions]
    assert [p["expected_cost"] for p in body["predictions"]] == [3 * a["age_years"] for a in admissions]


@pytest.mark.parametrize(
    "change",
    [
        {"paid_amount": 1000.0},  # the cost is not known at admission
        {"length_of_stay_days": 3},  # nor anything about the stay
        {"age_years": 131},
        {"sex": "X"},
        {"diagnosis_icd10": "i21"},
        {"admission_type": "planned"},
        {"admission_date": "2024-02-30"},
    ],
)
def test_predict_rejects_what_the_contract_rejects(client, change):
    admission = rows(generate(1, "2024-01-01", "2024-01-31", seed=3))[0] | change
    assert client.post("/predict", json={"admissions": [admission]}).status_code == 422


def test_predict_rejects_missing_fields_duplicates_and_size(client):
    admission = rows(generate(1, "2024-01-01", "2024-01-31", seed=4))[0]
    no_sex = {k: v for k, v in admission.items() if k != "sex"}
    assert client.post("/predict", json={"admissions": [no_sex]}).status_code == 422
    duplicated = client.post("/predict", json={"admissions": [admission, admission]})
    assert duplicated.status_code == 422
    assert duplicated.json()["detail"][0]["column"] == "admission_id"
    assert client.post("/predict", json={"admissions": []}).status_code == 422
    too_many = [admission | {"admission_id": str(i)} for i in range(MAX_ADMISSIONS + 1)]
    assert client.post("/predict", json={"admissions": too_many}).status_code == 422


def test_optional_columns_may_be_left_out(client):
    admission = rows(generate(1, "2024-01-01", "2024-01-31", seed=5))[0]
    del admission["residence_area"], admission["bed_specialty"]
    assert client.post("/predict", json={"admissions": [admission]}).status_code == 200


@pytest.fixture
def registered(tmp_path, monkeypatch):
    """A model registered as 'production' in a throwaway work folder, and the admissions table it came from."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    monkeypatch.setattr(config_module, "DATA_DIR", tmp_path / "data")
    df = contract.validate(generate(4_000, "2023-01-01", "2024-03-31", seed=6))
    CFG.table_path.parent.mkdir(parents=True)
    df.to_parquet(CFG.table_path, index=False)
    train = df[df["admission_date"] < "2024-01-01"]
    pair = CostPair(CostModel("log_l2", 7, 0.1, 50, 5, use_time=True), CostModel("tweedie", 7, 0.1, 50, 5, use_time=True))
    model = pair.fit(F.build_features(train), train[F.TARGET])
    registry.setup(CFG)
    with registry.mlflow.start_run():
        version = registry.log_model(CFG, model, train)
    registry.set_alias(CFG, registry.PRODUCTION, version)
    return df, model, version


@pytest.mark.slow
def test_api_and_batch_price_like_the_registered_model(registered, tmp_path):
    df, model, version = registered
    march = df[df["admission_date"].dt.month.eq(3) & df["admission_date"].dt.year.eq(2024)]
    expected = model.predict_both(F.build_features(march))
    assert not np.allclose(expected[TYPICAL], expected[EXPECTED])  # two different numbers per admission

    path = run_batch(CFG, month="2024-03")
    batch = pd.read_parquet(path)
    assert path == CFG.predictions_dir / "2024-03.parquet"
    assert batch["admission_id"].tolist() == march["admission_id"].tolist()
    assert set(batch["model_version"]) == {version}
    for column in (TYPICAL, EXPECTED):
        np.testing.assert_allclose(batch[column], expected[column])

    # A file of admissions not billed yet (no cost column), scored from a CSV.
    unbilled = tmp_path / "unbilled.csv"
    march.drop(columns=[F.TARGET]).to_csv(unbilled, index=False, date_format="%Y-%m-%d")
    from_file = pd.read_parquet(run_batch(CFG, input_path=str(unbilled)))
    for column in (TYPICAL, EXPECTED):
        np.testing.assert_allclose(from_file[column], expected[column])

    with TestClient(create_app(CFG)) as client:
        assert client.get("/health").json()["model_version"] == version
        body = client.post("/predict", json={"admissions": rows(march.head(20))}).json()
    for column in (TYPICAL, EXPECTED):
        np.testing.assert_allclose([p[column] for p in body["predictions"]], expected[column][:20], atol=0.005)


@pytest.mark.slow
def test_api_keeps_its_version_until_restarted(registered):
    df, model, first = registered
    with TestClient(create_app(CFG)) as client:
        with registry.mlflow.start_run():
            second = registry.log_model(CFG, model, df)
        registry.set_alias(CFG, registry.PRODUCTION, second)  # a promotion while the API is running
        assert client.get("/health").json()["model_version"] == first
    with TestClient(create_app(CFG)) as client:
        assert client.get("/health").json()["model_version"] == second


@pytest.mark.slow
def test_no_production_model_means_no_api_and_no_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(registry, "HOME", tmp_path)
    with pytest.raises(RuntimeError, match="no 'production' model"), TestClient(create_app(CFG)):
        pass
    unbilled = tmp_path / "a.csv"
    generate(10, "2024-01-01", "2024-01-31").to_csv(unbilled, index=False, date_format="%Y-%m-%d")
    with pytest.raises(RuntimeError, match="no 'production' model"):
        run_batch(CFG, input_path=str(unbilled))
