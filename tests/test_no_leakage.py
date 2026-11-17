"""Fail the build if anything known only after admission can reach the model."""

import pandas as pd
import pytest

from costmodel import contract
from costmodel import features as F


def admission(**overrides) -> dict:
    row = {
        "admission_id": "1", "admission_date": pd.Timestamp("2022-03-10"), "age_years": 45.0, "sex": "M",
        "diagnosis_icd10": "I219", "requested_procedure": "0303060190", "procedure_code_system": "SIGTAP",
        "admission_type": "urgent", "provider_id": "2273411", "residence_area": "330455", "bed_specialty": "03",
    }  # fmt: skip
    row.update(overrides)
    return row


def test_inputs_are_admission_time_only():
    assert set(F.ALLOWED_INPUTS) <= set(contract.ADMISSION_TIME)
    assert not set(F.ALLOWED_INPUTS) & set(F.FORBIDDEN)
    assert not any(c.startswith(contract.SOURCE_PREFIX) for c in F.ALLOWED_INPUTS)


@pytest.mark.parametrize("col", contract.POST_ADMISSION + [contract.TARGET])
def test_post_admission_and_target_are_forbidden(col):
    assert col in F.FORBIDDEN
    assert col not in F.FEATURES


def test_admission_id_is_not_a_feature():
    assert "admission_id" not in F.ALLOWED_INPUTS
    assert "admission_id" not in F.FEATURES


def test_build_features_needs_only_admission_columns():
    """Given *only* the allowed columns, feature building must work: it cannot be reading anything else."""
    df = pd.DataFrame([admission()])[F.ALLOWED_INPUTS]
    assert list(F.build_features(df).columns) == F.FEATURES


def test_forbidden_and_source_columns_do_not_change_features():
    base = pd.DataFrame([admission()])
    leaky = base.assign(
        paid_amount=99999.0,
        discharge_date=pd.Timestamp("2022-04-01"),
        length_of_stay_days=22.0,
        icu_days=10.0,
        performed_procedure="0406020000",
        death=1.0,
        src_billing_month="2022-05",
    )
    pd.testing.assert_frame_equal(F.build_features(base), F.build_features(leaky))


def test_optional_columns_may_be_absent_required_may_not():
    no_optional = pd.DataFrame([admission()]).drop(columns=contract.ADMISSION_TIME_OPTIONAL)
    out = F.build_features(no_optional)
    assert out.loc[0, "residence_area"] == "NA"
    with pytest.raises(KeyError):
        F.build_features(pd.DataFrame([admission()]).drop(columns=["requested_procedure"]))
