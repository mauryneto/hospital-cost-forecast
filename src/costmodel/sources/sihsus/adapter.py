"""Map SIH/SUS RD records to the canonical admissions table.

SIH-specific cleaning rules (each was checked against the real files, see README "Design decisions"):

* Keep only ``IDENT == "1"`` (the regular AIH). ``IDENT == "5"`` rows are monthly continuation bills of very long stays
  (mostly psychiatric / chronic care): the same ``N_AIH`` reappears every month. They are not new admissions.
* Rows are dated by *admission* (``DT_INTER``), not by the billing month of the file. Files are downloaded months past
  the last admission month, because 99.9% of admissions are billed within 5 months.
* One row per ``N_AIH``: if an AIH ever appears twice, keep the most recent bill.
* Drop admissions with ``VAL_TOT <= 0`` (nothing was paid, usually rejected bills).
"""

from __future__ import annotations

import pandas as pd

SEX = {"1": "M", "3": "F"}
ADMISSION_TYPE = {"01": "elective", "02": "urgent"}


def age_in_years(idade: pd.Series, cod_idade: pd.Series) -> pd.Series:
    """SIH stores age as a number plus a unit: 0/1 hours, 2 days, 3 months, 4 years, 5 = 100 + years."""
    idade = pd.to_numeric(idade, errors="coerce")
    unit = cod_idade.astype("string")
    years = pd.Series(float("nan"), index=idade.index)
    years = years.mask(unit.isin(["0", "1", "2", "3"]), 0.0)
    years = years.mask(unit == "4", idade)
    years = years.mask(unit == "5", 100 + idade)
    return years


def to_canonical(raw: pd.DataFrame, first_admission: str, last_admission: str) -> tuple[pd.DataFrame, dict[str, int]]:
    log: dict[str, int] = {"raw_rows": len(raw)}
    df = raw[raw["IDENT"] == "1"]
    log["regular_aih_only"] = len(df)

    admission = pd.to_datetime(df["DT_INTER"], format="%Y%m%d", errors="coerce")
    df = df[admission.between(pd.Timestamp(first_admission), pd.Timestamp(last_admission))]
    log["inside_admission_window"] = len(df)

    df = df.sort_values("SOURCE_FILE").drop_duplicates("N_AIH", keep="last")
    log["one_row_per_aih"] = len(df)

    df = df[df["VAL_TOT"] > 0]
    log["paid_amount_positive"] = len(df)

    out = pd.DataFrame(
        {
            "admission_id": df["N_AIH"],
            "admission_date": pd.to_datetime(df["DT_INTER"], format="%Y%m%d"),
            "age_years": age_in_years(df["IDADE"], df["COD_IDADE"]),
            "sex": df["SEXO"].map(SEX).fillna("U"),
            "diagnosis_icd10": df["DIAG_PRINC"],
            "requested_procedure": df["PROC_SOLIC"],
            "procedure_code_system": "SIGTAP",
            "admission_type": df["CAR_INT"].map(ADMISSION_TYPE).fillna("other"),
            "provider_id": df["CNES"],
            "residence_area": df["MUNIC_RES"],
            "bed_specialty": df["ESPEC"],
            "paid_amount": df["VAL_TOT"].astype(float),
            "discharge_date": pd.to_datetime(df["DT_SAIDA"], format="%Y%m%d", errors="coerce"),
            "length_of_stay_days": df["DIAS_PERM"].astype(float),
            "icu_days": df["UTI_MES_TO"].astype(float),
            "performed_procedure": df["PROC_REA"],
            "death": df["MORTE"].astype(float),
            "src_billing_month": df["ANO_CMPT"] + "-" + df["MES_CMPT"],
            "src_source_file": df["SOURCE_FILE"],
        }
    )
    out = out.sort_values(["admission_date", "admission_id"]).reset_index(drop=True)
    return out, log
