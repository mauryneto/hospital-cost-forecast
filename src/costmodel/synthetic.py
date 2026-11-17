"""Synthetic admissions in the canonical format, shaped like a private health plan (TUSS codes, contract prices).

Entirely made up: no real patient, plan or provider. It exists to try the "fill the template and run" path end to end,
and to test it automatically. Cost = procedure price x yearly price increase x age, admission-type and hospital
effects x random noise, so a model has real structure to find.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from costmodel import contract

N_PROCEDURES = 60
N_PROVIDERS = 30
YEARLY_PRICE_INCREASE = 0.06


def generate(n: int, start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    procedures = np.array([f"3{rng.integers(1_000_000, 9_999_999)}" for _ in range(N_PROCEDURES)])
    base_price = rng.lognormal(np.log(3000), 0.9, N_PROCEDURES)
    proc_weight = rng.dirichlet(np.full(N_PROCEDURES, 0.6))
    chapters = list("ACEIJKNOS")
    proc_diagnoses = [
        [f"{rng.choice(chapters)}{rng.integers(0, 99):02d}{rng.integers(0, 9)}" for _ in range(rng.integers(1, 4))]
        for _ in range(N_PROCEDURES)
    ]
    providers = np.array([f"PRV{i:03d}" for i in range(N_PROVIDERS)])
    provider_effect = rng.lognormal(0, 0.15, N_PROVIDERS)

    days = pd.date_range(start, end, freq="D")
    admission_date = pd.to_datetime(rng.choice(days, n))
    p = rng.choice(N_PROCEDURES, n, p=proc_weight)
    prov = rng.integers(0, N_PROVIDERS, n)
    age = np.clip(rng.normal(48, 22, n), 0, 100).round()
    urgent = rng.random(n) < 0.6
    years = (admission_date - pd.Timestamp(start)).days / 365.25

    cost = (
        base_price[p]
        * (1 + YEARLY_PRICE_INCREASE) ** years
        * (1 + 0.008 * np.maximum(age - 40, 0))
        * np.where(urgent, 1.25, 1.0)
        * provider_effect[prov]
        * rng.lognormal(0, 0.5, n)
    ).round(2)
    stay = np.maximum(1, rng.poisson(1 + cost / 1500)).astype(float)

    df = pd.DataFrame(
        {
            "admission_id": [f"SYN{i:08d}" for i in range(n)],
            "admission_date": admission_date.normalize(),
            "age_years": age,
            "sex": rng.choice(["M", "F", "U"], n, p=[0.47, 0.52, 0.01]),
            "diagnosis_icd10": [rng.choice(proc_diagnoses[i]) for i in p],
            "requested_procedure": procedures[p],
            "procedure_code_system": "TUSS",
            "admission_type": np.where(urgent, "urgent", rng.choice(["elective", "other"], n, p=[0.9, 0.1])),
            "provider_id": providers[prov],
            "residence_area": rng.choice([f"REG{i:02d}" for i in range(20)], n),
            "bed_specialty": rng.choice(["clinical", "surgical", "obstetric", "paediatric"], n),
            "paid_amount": cost,
            "discharge_date": admission_date.normalize() + pd.to_timedelta(stay, unit="D"),
            "length_of_stay_days": stay,
            "icu_days": np.where(rng.random(n) < 0.08, rng.integers(1, 10, n), 0).astype(float),
            "performed_procedure": np.where(
                rng.random(n) < 0.9, procedures[p], procedures[rng.integers(0, N_PROCEDURES, n)]
            ),
            "death": (rng.random(n) < 0.02).astype(float),
        }
    )
    return df.sort_values(["admission_date", "admission_id"]).reset_index(drop=True)[contract.ALL_COLUMNS]


def write(path: Path, n: int, start: str, end: str, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(n, start, end, seed).to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"{n:,} synthetic admissions written to {path}")
    return path
