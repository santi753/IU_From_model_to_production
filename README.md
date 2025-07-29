# Fraud Detection Model Demo

This repository shows how to train and serve a credit card fraud detection model using **MLflow** and a small FastAPI application.

The instructions below assume you are using **Windows PowerShell**.

## Prerequisites

- Python 3.8 or newer installed and added to your PATH
- [Optional] Git for cloning the repository
- The "Credit Card Fraud Detection" dataset (from Kaggle) saved as `data\data_raw.csv`

## 1. Set up a virtual environment
```powershell
# From the project directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Train and register the model
By default the tracking server uses the local file `mlflow.db` and stores runs under `mlruns/`.
```powershell
python .\mlops\train_and_register.py
```
This step trains a RandomForest model, logs it to MLflow, and registers it as `fraud_detector` in the **Production** stage.

## 3. Inspect runs with the MLflow UI (optional)
```powershell
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser to explore the experiment and model registry.

## 4. Start the API
```powershell
uvicorn mlops.api_main:app --reload
```
The service loads the production model from MLflow. Use the `API_KEY` environment variable (default value `dev-key`) to authorize requests.

## 5. Example request
Here is a minimal example of calling the `/predict` endpoint with PowerShell:
```powershell
$transaction = @{ Time = 0; Amount = 149.62; V1 = -1.36; V2 = -0.07; V3 = 2.54; V4 = 1.38; V5 = -0.34; V6 = 0.46; V7 = 0.24; V8 = 0.10; V9 = 0.36; V10 = 0.09; V11 = -0.55; V12 = -0.62; V13 = -0.99; V14 = -0.31; V15 = 1.47; V16 = -0.47; V17 = 0.21; V18 = 0.03; V19 = 0.40; V20 = 0.25; V21 = -0.02; V22 = 0.28; V23 = -0.11; V24 = 0.07; V25 = 0.13; V26 = -0.19; V27 = 0.13; V28 = -0.02 }
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body ($transaction | ConvertTo-Json) -ContentType 'application/json' -Headers @{ 'x-api-key' = 'dev-key' }
```
The response contains the predicted label and fraud probability.

## Environment variables
- `MLFLOW_TRACKING_URI` – location of the tracking database (default `sqlite:///mlflow.db`)
- `MLFLOW_ARTIFACT_LOCATION` – folder for MLflow artifacts (`./mlruns` by default)
- `MODEL_NAME` – name of the registered model (`fraud_detector`)
- `DECISION_THRESHOLD` – threshold for predicting fraud (0.640 by default)
- `API_KEY` – key required when calling the API (`dev-key` by default)

Set them in PowerShell using `$env:VAR = 'value'` before running the training script or API server.

## Check MLflow

- $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
- $env:API_KEY = "dev-key"
- mlflow ui --backend-store-uri sqlite:///mlflow.db
