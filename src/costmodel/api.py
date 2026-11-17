"""HTTP API: ``POST /predict`` prices admissions with the production model; ``GET /health`` says which model it uses.

Start it with ``costmodel serve`` (interactive docs at ``/docs``). The model is loaded once, at startup: the API does
not start without a ``production`` model, and it keeps the version it loaded until it is restarted. So a promotion or
a rollback (moving the alias in the registry) takes effect when the API restarts.

Every admission is checked twice, on purpose: the request schema below rejects bad requests early with a clear message
(including any field the contract does not know at admission, such as the cost or the length of stay), and the scorer
then applies the same contract check as the batch job.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from costmodel import contract
from costmodel.config import Config
from costmodel.score import Scorer

MAX_ADMISSIONS = 1_000


class Admission(BaseModel):
    """What is known when the patient is admitted: the contract's admission-time columns, nothing else."""

    model_config = ConfigDict(extra="forbid")  # e.g. paid_amount or length_of_stay_days -> 422, never silently ignored

    admission_id: str = Field(min_length=1, max_length=64, examples=["330000000001"])
    admission_date: date = Field(examples=["2024-03-15"])
    age_years: float = Field(ge=0, le=130, examples=[67])
    sex: Literal["M", "F", "U"]
    diagnosis_icd10: str = Field(pattern=r"^[A-Z]\d{2}[0-9A-Z]?$", examples=["I214"])
    requested_procedure: str = Field(min_length=1, max_length=20, examples=["0303060212"])
    procedure_code_system: Literal["SIGTAP", "TUSS", "OTHER"]
    admission_type: Literal["elective", "urgent", "other"]
    provider_id: str = Field(min_length=1, max_length=20, examples=["2280167"])
    residence_area: str | None = Field(default=None, max_length=20, examples=["330455"])
    bed_specialty: str | None = Field(default=None, max_length=20, examples=["03"])


class PredictRequest(BaseModel):
    admissions: list[Admission] = Field(min_length=1, max_length=MAX_ADMISSIONS)


class Prediction(BaseModel):
    admission_id: str
    predicted_cost: float


class PredictResponse(BaseModel):
    model: str
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    model_version: str
    fitted_on: str | None = None
    test_mae: float | None = None


def create_app(cfg: Config, scorer: Scorer | None = None) -> FastAPI:
    """The app for one dataset. ``scorer`` is for tests; by default the production model is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = scorer or Scorer.production(cfg)  # no production model -> the API refuses to start
        yield

    app = FastAPI(
        title="hospital-cost-forecast",
        summary="Predicted cost of a hospital admission, at the moment of admission.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> Health:
        s: Scorer = app.state.scorer
        test_mae = s.tags.get("test_mae")
        return Health(
            status="ok",
            model=s.model_name,
            model_version=s.version,
            fitted_on=s.tags.get("fitted_on"),
            test_mae=float(test_mae) if test_mae is not None else None,
        )

    @app.post("/predict")
    def predict(request: PredictRequest) -> PredictResponse:
        s: Scorer = app.state.scorer
        frame = pd.DataFrame([a.model_dump() for a in request.admissions])
        frame["admission_date"] = pd.to_datetime(frame["admission_date"])
        try:
            predicted = s.score(frame)
        except contract.ContractError as exc:  # e.g. the same admission_id twice
            raise HTTPException(status_code=422, detail=exc.report.to_dict(orient="records")) from None
        return PredictResponse(
            model=s.model_name,
            model_version=s.version,
            predictions=[
                Prediction(admission_id=a.admission_id, predicted_cost=round(float(p), 2))
                for a, p in zip(request.admissions, predicted, strict=True)
            ],
        )

    return app
