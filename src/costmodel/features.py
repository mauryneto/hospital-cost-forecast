"""The single source of truth for what the model may see.

The prediction happens **at admission**, so features are built only from the contract's admission-time columns.
Post-admission columns (length of stay, ICU days, performed procedure, death, discharge), the target and every
source-specific ``src_*`` column are forbidden, and ``tests/test_no_leakage.py`` fails if any of them reaches the
features.

``build_features`` first narrows its input to ``ALLOWED_INPUTS``. So even a careless edit that references a forbidden
column inside it fails (``KeyError``) instead of silently leaking.
"""

from __future__ import annotations

import pandas as pd

from costmodel import contract

TARGET = contract.TARGET
ALLOWED_INPUTS = [c for c in contract.ADMISSION_TIME if c != "admission_id"]
FORBIDDEN = contract.POST_ADMISSION + [TARGET]

CATEGORICAL = [
    "sex",
    "residence_area",
    "diagnosis_icd10",
    "diagnosis_category",  # ICD-10 3-character category (e.g. I21)
    "diagnosis_chapter",  # ICD-10 letter (e.g. I = circulatory)
    "requested_procedure",
    "procedure_prefix2",  # coarser procedure family (SIGTAP group, e.g. 04 = surgery)
    "procedure_prefix4",  # SIGTAP subgroup
    "procedure_code_system",
    "admission_type",
    "bed_specialty",
    "provider_id",
]
NUMERIC = [
    "age_years",
    "admission_month",  # month of year, for seasonality
    "admission_period",  # months elapsed (year * 12 + month): lets a model use the most recent price level
]
FEATURES = CATEGORICAL + NUMERIC


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ALLOWED_INPUTS if c not in df.columns and c not in contract.ADMISSION_TIME_OPTIONAL]
    if missing:
        raise KeyError(f"required admission-time columns missing: {missing}")
    src = df.reindex(columns=ALLOWED_INPUTS)  # optional columns may be absent; nothing outside the allow-list is read
    proc = src["requested_procedure"].astype("string")
    diag = src["diagnosis_icd10"].astype("string")
    admitted = pd.to_datetime(src["admission_date"])
    out = pd.DataFrame(
        {
            "sex": src["sex"],
            "residence_area": src["residence_area"],
            "diagnosis_icd10": diag,
            "diagnosis_category": diag.str[:3],
            "diagnosis_chapter": diag.str[:1],
            "requested_procedure": proc,
            "procedure_prefix2": proc.str[:2],
            "procedure_prefix4": proc.str[:4],
            "procedure_code_system": src["procedure_code_system"],
            "admission_type": src["admission_type"],
            "bed_specialty": src["bed_specialty"],
            "provider_id": src["provider_id"],
            "age_years": pd.to_numeric(src["age_years"], errors="coerce").astype(float),
            "admission_month": admitted.dt.month.astype(float),
            "admission_period": (admitted.dt.year * 12 + admitted.dt.month - 1).astype(float),
        },
        index=src.index,
    )
    for col in CATEGORICAL:
        out[col] = out[col].astype("string").fillna("NA")
    return out[FEATURES]
