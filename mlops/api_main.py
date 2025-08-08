# mlops/api_main.py
import os
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

API_KEY = os.getenv("API_KEY", "dev-key")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detector")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(TRACKING_URI)

# ------------- Model Management -------------
model = None
model_version = None

def load_production_model():
    """Load the Production model from MLflow registry using alias system"""
    global model, model_version
    try:
        # Use alias-based URI instead of stage-based
        uri = f"models:/{MODEL_NAME}@Production"
        model = mlflow.pyfunc.load_model(uri)
        
        # Get version from alias system
        client = mlflow.tracking.MlflowClient()
        model_version_info = client.get_model_version_by_alias(MODEL_NAME, "Production")
        model_version = model_version_info.version
        
        # Update Prometheus gauge with numeric version
        try:
            model_version_gauge.set(float(model_version))
        except ValueError:
            # If version is not numeric, set to 0
            model_version_gauge.set(0)
            
        print(f"Successfully loaded model {MODEL_NAME} version {model_version} from Production alias")
        return True
        
    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow error loading model: {e}")
        if "not found" in str(e).lower() or "alias" in str(e).lower():
            print(f"No Production model found for {MODEL_NAME}. API will start without a model.")
            model_version = "none"
            model_version_gauge.set(0)
            return False
        else:
            raise
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        model_version = "error"
        model_version_gauge.set(0)
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - try to load model but don't fail if it doesn't exist
    model_loaded = load_production_model()
    if model_loaded:
        print(f"✅ API started with model {MODEL_NAME} version {model_version}")
    else:
        print(f"⚠️ API started without a model. Use /reload endpoint after training.")
    yield
    # Shutdown
    pass

app = FastAPI(title="Fraud Detection API", version="1.1.0", lifespan=lifespan)

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
# 30 total columns: Time, Amount, and V1-V28
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

# ------------- Endpoints -------------
@app.get("/health")
def health():
    """Health check endpoint - works even without a model loaded"""
    status = "ok" if model is not None else "no_model"
    return {
        "status": status, 
        "model_name": MODEL_NAME, 
        "model_version": model_version or "none",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(
    payload: Union[Transaction, BatchRequest],
    auth_ok: bool = Depends(verify_api_key)
):
    """Unified prediction endpoint for single or batch transactions"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="No model is currently loaded. Please train a model first and use the /reload endpoint."
        )
    
    try:
        # Normalize to DataFrame
        if isinstance(payload, Transaction):
            df = pd.DataFrame([payload.dict()])
        else:
            df = pd.DataFrame([r.dict() for r in payload.records])

        # Validate required features
        missing_features = [feat for feat in FEATURES if feat not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}"
            )

        # Ensure column order
        df = df[FEATURES]

        # Make prediction
        preds = model.predict(df)  # returns DataFrame with probability + label
        preds = preds.rename(columns={"label": "predicted_label", "probability": "fraud_probability"})
        
        # Count predictions by label
        for label in preds["predicted_label"]:
            prediction_count.labels(label=str(label)).inc()
        
        results = preds.to_dict(orient="records")
        
        response_data = {
            "model_name": MODEL_NAME, 
            "model_version": model_version, 
            "predictions": results
        }
        
        # Add batch size for batch requests
        if isinstance(payload, BatchRequest):
            response_data["batch_size"] = len(results)
            
        return response_data
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
def model_info(auth_ok: bool = Depends(verify_api_key)):
    return {
        "model_name": MODEL_NAME, 
        "model_version": model_version or "none", 
        "features": FEATURES,
        "model_loaded": model is not None
    }

@app.post("/reload")
def reload_model(auth_ok: bool = Depends(verify_api_key)):
    """Hot-reload the Production model from MLflow"""
    try:
        old_version = model_version
        model_loaded = load_production_model()
        
        if model_loaded:
            return {
                "status": "success",
                "model_name": MODEL_NAME,
                "old_version": old_version or "none",
                "new_version": model_version,
                "message": f"Model reloaded successfully from {old_version or 'none'} to {model_version}"
            }
        else:
            return {
                "status": "warning",
                "model_name": MODEL_NAME,
                "old_version": old_version or "none", 
                "new_version": "none",
                "message": "No Production model found to load"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics - unauthenticated, rely on Docker port binding"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
