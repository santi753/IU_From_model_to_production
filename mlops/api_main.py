# mlops/api_main.py
import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Union

API_KEY = os.getenv("API_KEY", "dev-key")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detector")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# ------------- Auth dependency -------------
def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return True

# ------------- Schemas -------------
# 30 original columns (unscaled Time & Amount)
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

class Transaction(BaseModel):
    Time: float
    Amount: float
    # V1..V28
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float;  V6: float;  V7: float
    V8: float;  V9: float;  V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

class BatchRequest(BaseModel):
    records: List[Transaction] = Field(..., description="List of transactions")

# ------------- Load model on startup -------------
@app.on_event("startup")
def load_model():
    global model, model_version
    uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(uri)
    # best-effort: get version from client (optional)
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        model_version = mv.version
    except Exception:
        model_version = "unknown"

@app.get("/health")
def health():
    return {"status": "ok", "model_name": MODEL_NAME, "model_version": model_version}

@app.post("/predict")
def predict(
    payload: Union[Transaction, BatchRequest],
    auth_ok: bool = Depends(verify_api_key)
):
    # Normalize to DataFrame
    if isinstance(payload, Transaction):
        df = pd.DataFrame([payload.dict()])
    else:
        df = pd.DataFrame([r.dict() for r in payload.records])

    # Ensure column order
    df = df[FEATURES]

    preds = model.predict(df)  # returns DataFrame with probability + label
    preds = preds.rename(columns={"label": "predicted_label", "probability": "fraud_probability"})
    results = preds.to_dict(orient="records")
    return {"model_name": MODEL_NAME, "model_version": model_version, "predictions": results}

@app.get("/model-info")
def model_info(auth_ok: bool = Depends(verify_api_key)):
    return {"model_name": MODEL_NAME, "model_version": model_version, "features": FEATURES}
