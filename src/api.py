"""FastAPI inference service for the churn model.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
Then POST to /predict, e.g.:
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"tenure_months": 5, "monthly_charges": 79.9, "total_charges": 399.5,
           "support_calls": 3, "data_usage_gb": 210.0, "contract": "month-to-month",
           "payment_method": "electronic-check", "internet_service": "fiber-optic",
           "tech_support": "no"}'
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"

app = FastAPI(title="Churn Prediction API", version="1.0.0")


class CustomerFeatures(BaseModel):
    tenure_months: float = Field(ge=0, le=200)
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0, le=50)
    data_usage_gb: float = Field(ge=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    payment_method: Literal["electronic-check", "mailed-check",
                            "bank-transfer", "credit-card"]
    internet_service: Literal["dsl", "fiber-optic", "none"]
    tech_support: Literal["yes", "no"]


class PredictionOut(BaseModel):
    churn_probability: float
    churn_predicted: bool
    risk_band: Literal["low", "medium", "high"]


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    # The pickled pipeline references the `features`/`data` modules by their
    # top-level names (as used in train.py), so ensure src/ is importable.
    import sys
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionOut)
def predict(features: CustomerFeatures) -> PredictionOut:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    import pandas as pd
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    band = "low" if proba < 0.33 else ("medium" if proba < 0.66 else "high")
    return PredictionOut(
        churn_probability=round(proba, 4),
        churn_predicted=proba >= 0.5,
        risk_band=band,
    )
