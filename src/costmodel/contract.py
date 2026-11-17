"""The data contract: the canonical ``admissions`` table every source must produce. See DATA_CONTRACT.md."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors
import pandera.pandas as pa
from pandera.pandas import Check, Column

# Known at admission: the only columns features may be built from.
ADMISSION_TIME_REQUIRED = [
    "admission_id",
    "admission_date",
    "age_years",
    "sex",
    "diagnosis_icd10",
    "requested_procedure",
    "procedure_code_system",
    "admission_type",
    "provider_id",
]
ADMISSION_TIME_OPTIONAL = ["residence_area", "bed_specialty"]
ADMISSION_TIME = ADMISSION_TIME_REQUIRED + ADMISSION_TIME_OPTIONAL

TARGET = "paid_amount"

# Known only during or after the stay. Allowed in the table, never as features.
POST_ADMISSION = ["discharge_date", "length_of_stay_days", "icu_days", "performed_procedure", "death"]

SOURCE_PREFIX = "src_"
ALL_COLUMNS = ADMISSION_TIME + [TARGET] + POST_ADMISSION

DEFAULT_MAX_PAID_AMOUNT = 1_000_000.0

TEXT_COLUMNS = [
    "admission_id",
    "sex",
    "diagnosis_icd10",
    "requested_procedure",
    "procedure_code_system",
    "admission_type",
    "provider_id",
    "residence_area",
    "bed_specialty",
    "performed_procedure",
]
DATE_COLUMNS = ["admission_date", "discharge_date"]


def _columns(max_paid_amount: float) -> dict[str, Column]:
    text = {"dtype": "string", "coerce": True}
    return {
        "admission_id": Column(**text, unique=True, checks=Check.str_length(min_value=1)),
        "admission_date": Column("datetime64[ns]", coerce=True),
        "age_years": Column(float, Check.in_range(0, 130), coerce=True),
        "sex": Column(**text, checks=Check.isin(["M", "F", "U"])),
        "diagnosis_icd10": Column(**text, checks=Check.str_matches(r"^[A-Z]\d{2}[0-9A-Z]?$")),
        "requested_procedure": Column(**text, checks=Check.str_length(min_value=1)),
        "procedure_code_system": Column(**text, checks=Check.isin(["SIGTAP", "TUSS", "OTHER"])),
        "admission_type": Column(**text, checks=Check.isin(["elective", "urgent", "other"])),
        "provider_id": Column(**text, checks=Check.str_length(min_value=1)),
        "residence_area": Column(**text, nullable=True, required=False),
        "bed_specialty": Column(**text, nullable=True, required=False),
        TARGET: Column(float, Check.in_range(0, max_paid_amount, include_min=False), coerce=True),
        "discharge_date": Column("datetime64[ns]", nullable=True, required=False, coerce=True),
        "length_of_stay_days": Column(float, Check.ge(0), nullable=True, required=False, coerce=True),
        "icu_days": Column(float, Check.ge(0), nullable=True, required=False, coerce=True),
        "performed_procedure": Column(**text, nullable=True, required=False),
        "death": Column(float, Check.isin([0, 1]), nullable=True, required=False, coerce=True),
        rf"^{SOURCE_PREFIX}.+": Column(regex=True, nullable=True, required=False),
    }


def schema(max_paid_amount: float = DEFAULT_MAX_PAID_AMOUNT) -> pa.DataFrameSchema:
    """The whole table, as ``costmodel data`` saves it: every column checked, no column outside the contract."""
    return pa.DataFrameSchema(_columns(max_paid_amount), strict=True)


def input_schema() -> pa.DataFrameSchema:
    """Only the admission-time columns: what scoring needs. The cost is not known yet; other columns are ignored."""
    columns = _columns(DEFAULT_MAX_PAID_AMOUNT)
    return pa.DataFrameSchema({c: columns[c] for c in ADMISSION_TIME}, strict=False)


class ContractError(ValueError):
    """The table does not follow the contract. ``report`` has one row per problem."""

    def __init__(self, report: pd.DataFrame):
        self.report = report
        super().__init__(f"{len(report)} contract problem(s):\n{report.to_string(index=False)}")


def validate(df: pd.DataFrame, max_paid_amount: float = DEFAULT_MAX_PAID_AMOUNT) -> pd.DataFrame:
    """Return the table with canonical dtypes, or raise ContractError listing every problem found."""
    return _check(schema(max_paid_amount), df)


def validate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Like ``validate``, for admissions to be scored: only the admission-time columns are checked."""
    return _check(input_schema(), df)


def _check(checker: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    try:
        return checker.validate(df, lazy=True)
    except pandera.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        # A missing column has no column name; an empty or unparseable value (e.g. date 2024-02-30) has no value.
        cases["column"] = cases["column"].map(lambda v: "(table)" if pd.isna(v) else str(v))
        cases["failure_case"] = cases["failure_case"].map(lambda v: "<empty or invalid>" if pd.isna(v) else str(v))
        cases["check"] = cases["check"].map(str)
        report = (
            cases.groupby(["column", "check"])
            .agg(rows=("failure_case", "size"), examples=("failure_case", lambda s: ", ".join(s.unique()[:3])))
            .reset_index()
        )
        raise ContractError(report) from None


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet admissions file without guessing types (codes keep their leading zeros)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        from costmodel.paths import HOME

        path = HOME / path  # e.g. "data/my_admissions.csv" inside the work folder
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    df = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""])
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    return df
