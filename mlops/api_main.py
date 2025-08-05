# mlops/api_main.py
import os
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

API_KEY = os.getenv("API_KEY", "dev-key")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detector")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Fraud Detection API", version="1.1.0")

# ------------- Prometheus Metrics -------------
request_count = Counter(
    'fraud_api_requests_total',
    'Total number of requests by endpoint and status',
    ['endpoint', 'status']
)

request_latency = Histogram(
    'fraud_api_request_duration_seconds',
    'Request latency by endpoint',
    ['endpoint']
)

prediction_count = Counter(
    'fraud_api_predictions_total',
    'Total number of predictions by label',
    ['label']
)

model_version_gauge = Gauge(
    'fraud_api_model_version',
    'Current MLflow model version'
)

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
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float;  V6: float;  V7: float  # noqa: E702
    V8: float;  V9: float;  V10: float; V11: float; V12: float; V13: float; V14: float  # noqa: E702
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float  # noqa: E702
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float  # noqa: E702

class BatchRequest(BaseModel):
    records: List[Transaction] = Field(..., description="List of transactions")

# ------------- Model Management -------------
model = None
model_version = None

def load_production_model():
    """Load the Production model from MLflow registry"""
    global model, model_version
    uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(uri)
    # best-effort: get version from client (optional)
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        model_version = mv.version
        # Update Prometheus gauge with numeric version
        try:
            model_version_gauge.set(float(model_version))
        except ValueError:
            # If version is not numeric, set to 0
            model_version_gauge.set(0)
    except Exception:
        model_version = "unknown"
        model_version_gauge.set(0)

# ------------- Middleware for metrics -------------
@app.middleware("http")
async def track_requests(request: Request, call_next):
    # Skip metrics endpoint to avoid recursion
    if request.url.path == "/metrics":
        return await call_next(request)
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Extract endpoint path
    endpoint = request.url.path
    
    # Record metrics
    request_count.labels(endpoint=endpoint, status=response.status_code).inc()
    request_latency.labels(endpoint=endpoint).observe(duration)
    
    return response

# ------------- Load model on startup -------------
@app.on_event("startup")
def startup_event():
    load_production_model()
    print(f"Loaded model {MODEL_NAME} version {model_version}")

# ------------- Endpoints -------------
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
    
    # Count predictions by label
    for label in preds["predicted_label"]:
        prediction_count.labels(label=str(label)).inc()
    
    results = preds.to_dict(orient="records")
    return {"model_name": MODEL_NAME, "model_version": model_version, "predictions": results}

@app.get("/model-info")
def model_info(auth_ok: bool = Depends(verify_api_key)):
    return {"model_name": MODEL_NAME, "model_version": model_version, "features": FEATURES}

@app.post("/reload")
def reload_model(auth_ok: bool = Depends(verify_api_key)):
    """Hot-reload the Production model from MLflow"""
    try:
        old_version = model_version
        load_production_model()
        return {
            "status": "success",
            "model_name": MODEL_NAME,
            "old_version": old_version,
            "new_version": model_version,
            "message": f"Model reloaded successfully from {old_version} to {model_version}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics - unauthenticated, rely on Docker port binding"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)