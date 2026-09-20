from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churn_pipeline import EXPECTED_INPUT_COLUMNS, MODEL_PATH, POSITIVE_CLASS, load_artifact


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customerID: str | None = None
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run train.py first to create it."
        )
    return load_artifact(MODEL_PATH)


app = FastAPI(title="Telco Customer Churn API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerInput) -> PredictionResponse:
    try:
        artifact = get_artifact()
        pipeline = artifact["pipeline"]
        frame = pd.DataFrame([payload.model_dump()])
        frame = frame[[column for column in frame.columns if column in EXPECTED_INPUT_COLUMNS]]
        probabilities = pipeline.predict_proba(frame)
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(POSITIVE_CLASS)
        churn_probability = float(probabilities[0, positive_index])
        prediction = POSITIVE_CLASS if churn_probability >= 0.5 else "No"
        return PredictionResponse(prediction=prediction, churn_probability=round(churn_probability, 4))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction failure: {exc}") from exc
