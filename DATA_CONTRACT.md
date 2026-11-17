# Data contract: the `admissions` table

One row per hospital admission. Any source that produces this table can use the whole pipeline. Check a file with:

```
uv run costmodel validate path/to/admissions.csv
docker compose run --rm costmodel validate data/admissions.csv     # file inside ./docker-work/data
```

Every problem is listed at once (column, rule broken, number of rows, examples) and the command exits with code 1.
To see a valid file, generate made-up data with `costmodel synthetic`.

An empty template with the header is in [`templates/admissions_template.csv`](templates/admissions_template.csv).
Dates are `YYYY-MM-DD`; money is a decimal number with `.` as the separator, in the dataset's currency.

## Admission-time columns (the model may use these)

| Column | Required | Type | Rule | Meaning |
|---|---|---|---|---|
| `admission_id` | yes | text | unique | Identifier of the admission (claim / AIH number). Never a person's identifier. |
| `admission_date` | yes | date | | Day the patient was admitted. Splits and monitoring use this date. |
| `age_years` | yes | number | 0 to 130 | Age at admission, in whole or fractional years (babies under one year: `0`). |
| `sex` | yes | text | `M`, `F`, `U` | `U` = unknown / not informed. |
| `diagnosis_icd10` | yes | text | ICD-10 without dot, e.g. `I219` | Main diagnosis recorded at admission. |
| `requested_procedure` | yes | text | | Procedure requested when the admission was authorised. |
| `procedure_code_system` | yes | text | `SIGTAP`, `TUSS`, `OTHER` | Coding system of the procedure columns. |
| `admission_type` | yes | text | `elective`, `urgent`, `other` | |
| `provider_id` | yes | text | | Hospital identifier (CNES for SUS, provider code for a plan). |
| `residence_area` | no | text | | Patient's municipality / region code. Blank if unknown. |
| `bed_specialty` | no | text | | Bed or care specialty code. Blank if unknown. |

## Target

| Column | Required | Type | Rule | Meaning |
|---|---|---|---|---|
| `paid_amount` | yes | number | > 0 and below the configured ceiling | Total amount paid for the admission. |

## Post-admission columns (never used as features)

Known only during or after the stay. Optional; when present they enable leakage tests and richer analysis.

| Column | Type | Meaning |
|---|---|---|
| `discharge_date` | date | |
| `length_of_stay_days` | number | |
| `icu_days` | number | |
| `performed_procedure` | text | Procedure actually performed (often differs from the requested one in private care). |
| `death` | 0 / 1 | In-hospital death. |

## Anything else

Extra columns are allowed if their names start with `src_` (for example `src_billing_month`). The pipeline carries them
along and never uses them as features. Any other unknown column is an error, so typos do not pass silently.

## What must never be in the table

Names, document numbers (CPF, card or member number), addresses, phone numbers, full birth dates or anything else that
identifies a person. Age in years and a municipality code are enough.
