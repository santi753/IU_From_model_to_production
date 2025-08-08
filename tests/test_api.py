import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi.testclient import TestClient
import mlops.api_main as api_main

class DummyModel:
    def predict(self, df: pd.DataFrame):
        return pd.DataFrame({"probability": [0.4] * len(df), "label": [1] * len(df)})

class DummyClient:
    def get_latest_versions(self, name, stages=None):
        return [type("mv", (), {"version": "1"})()]

def _setup(monkeypatch):
    monkeypatch.setattr(api_main.mlflow.pyfunc, "load_model", lambda uri: DummyModel())
    monkeypatch.setattr(api_main.mlflow.tracking, "MlflowClient", lambda: DummyClient())


def test_predict_endpoint(monkeypatch):
    _setup(monkeypatch)
    with TestClient(api_main.app) as client:
        payload = {feat: 0.0 for feat in api_main.FEATURES}
        response = client.post("/predict", json=payload, headers={"x-api-key": "dev-key"})
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == api_main.MODEL_NAME
        assert data["predictions"][0]["fraud_probability"] == 0.4
        assert data["predictions"][0]["predicted_label"] == 1

def test_batch_predict_endpoint(monkeypatch):
    _setup(monkeypatch)
    with TestClient(api_main.app) as client:
        # Create batch request with 2 transactions
        payload = {
            "records": [
                {feat: 0.0 for feat in api_main.FEATURES},
                {feat: 1.0 for feat in api_main.FEATURES}
            ]
        }
        response = client.post("/predict/batch", json=payload, headers={"x-api-key": "dev-key"})
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == api_main.MODEL_NAME
        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["fraud_probability"] == 0.4
        assert data["predictions"][0]["predicted_label"] == 1

def test_missing_features_error(monkeypatch):
    _setup(monkeypatch)
    with TestClient(api_main.app) as client:
        # Missing some features - Pydantic will validate first and return 422
        payload = {"Time": 0.0, "Amount": 100.0}  # Missing V1-V28
        response = client.post("/predict", json=payload, headers={"x-api-key": "dev-key"})
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

def test_invalid_api_key():
    with TestClient(api_main.app) as client:
        payload = {feat: 0.0 for feat in api_main.FEATURES}
        response = client.post("/predict", json=payload, headers={"x-api-key": "wrong-key"})
        assert response.status_code == 401
        data = response.json()
        assert "Invalid or missing API key" in data["detail"]
