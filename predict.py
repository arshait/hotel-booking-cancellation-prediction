"""FastAPI service for hotel booking cancellation probability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hotel_ml import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    required = {"pipeline", "threshold", "features", "model_name"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    return artifact


try:
    ARTIFACT = load_artifact(MODEL_PATH)
except (FileNotFoundError, ValueError) as exc:
    ARTIFACT = None
    STARTUP_ERROR = str(exc)
else:
    STARTUP_ERROR = None

app = FastAPI(
    title="Hotel booking cancellation API",
    version="1.0.0",
    description=(
        "Scores one booking with the saved preprocessing pipeline. "
        "The service does not retrain on startup."
    ),
)


class Booking(BaseModel):
    lead_time: float = Field(ge=0)
    arrival_date_year: int = Field(ge=2010, le=2100)
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: float = Field(ge=0)
    stays_in_week_nights: float = Field(ge=0)
    adults: float = Field(ge=0)
    children: float | None = Field(default=None, ge=0)
    babies: float = Field(ge=0)
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: float = Field(ge=0)
    previous_bookings_not_canceled: float = Field(ge=0)
    days_in_waiting_list: float = Field(ge=0)
    adr: float | None = None
    required_car_parking_spaces: float = Field(ge=0)
    total_of_special_requests: float = Field(ge=0)
    hotel: str
    arrival_date_month: str
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    reserved_room_type: str
    deposit_type: str
    customer_type: str


class Prediction(BaseModel):
    cancellation_probability: float
    will_cancel: bool


@app.get("/health")
def health() -> dict[str, Any]:
    if ARTIFACT is None:
        raise HTTPException(status_code=503, detail=STARTUP_ERROR or "model not loaded")
    return {
        "status": "ok",
        "model_name": ARTIFACT["model_name"],
        "threshold": ARTIFACT["threshold"],
    }


@app.post("/predict", response_model=Prediction)
def predict(booking: Booking) -> Prediction:
    if ARTIFACT is None:
        raise HTTPException(status_code=503, detail=STARTUP_ERROR or "model not loaded")
    record = booking.model_dump()
    if record.get("adr") is not None and record["adr"] < 0:
        record["adr"] = None
    frame = pd.DataFrame([record], columns=FEATURES)
    try:
        probability = float(ARTIFACT["pipeline"].predict_proba(frame)[:, 1][0])
    except Exception as exc:  # noqa: BLE001 - return a client-safe message
        raise HTTPException(status_code=400, detail=f"Could not score booking: {exc}") from exc
    if not 0.0 <= probability <= 1.0:
        raise HTTPException(status_code=500, detail="Model returned a probability outside [0, 1].")
    threshold = float(ARTIFACT["threshold"])
    return Prediction(
        cancellation_probability=round(probability, 6),
        will_cancel=probability >= threshold,
    )


def feature_contract() -> dict[str, list[str]]:
    return {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}
