import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

import predict
from hotel_ml import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"
EXAMPLE = ROOT / "reports" / "example_request.json"


def test_saved_model_roundtrip_and_probability_range():
    artifact = joblib.load(MODEL)
    frame = pd.DataFrame([json.loads(EXAMPLE.read_text(encoding="utf-8"))], columns=FEATURES)
    probability = float(artifact["pipeline"].predict_proba(frame)[:, 1][0])
    assert 0.0 <= probability <= 1.0
    again = float(joblib.load(MODEL)["pipeline"].predict_proba(frame)[:, 1][0])
    assert probability == again
    assert artifact["model_name"] == "hist_gradient_boosting"
    assert artifact["threshold"] == 0.4


def test_health_and_predict_contract():
    client = TestClient(predict.app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert "threshold" in body

    booking = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    response = client.post("/predict", json=booking)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"cancellation_probability", "will_cancel"}
    assert 0.0 <= payload["cancellation_probability"] <= 1.0
    assert isinstance(payload["will_cancel"], bool)
    again = client.post("/predict", json=booking)
    assert again.json() == payload


def test_invalid_request_is_rejected():
    client = TestClient(predict.app)
    response = client.post("/predict", json={"hotel": "City Hotel"})
    assert response.status_code == 422
    booking = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    booking["lead_time"] = -5
    assert client.post("/predict", json=booking).status_code == 422
