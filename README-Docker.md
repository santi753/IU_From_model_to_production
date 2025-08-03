# Docker Setup for Fraud Detection System

This setup containerizes the entire fraud detection system with MLflow tracking server, model training, and API service.

## Architecture

- **MLflow Server**: Centralized tracking server with SQLite backend and local artifact storage
- **Training Container**: Trains the model and registers it to MLflow
- **API Container**: Serves predictions using the production model from MLflow

## Quick Start

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the services:**
   - MLflow UI: http://localhost:5000
   - API: http://localhost:8000
   - API Health: http://localhost:8000/health

3. **Train a new model (if needed):**
   ```bash
   docker-compose run --rm train
   ```

4. **Stop all services:**
   ```bash
   docker-compose down
   ```

## Individual Container Commands

### MLflow Server Only
```bash
# Build
docker build -f Dockerfile.mlflow -t fraud-mlflow .

# Run with volumes
docker run -d \
  --name fraud-mlflow \
  -p 127.0.0.1:5000:5000 \
  -v mlflow-db:/mlflow/mlflow-db \
  -v mlflow-artifacts:/mlflow/mlruns \
  fraud-mlflow
```

### Training Only
```bash
# Build
docker build -f Dockerfile.train -t fraud-train .

# Run (requires MLflow server running)
docker run --rm \
  --network host \
  -v $(pwd)/data:/app/data:ro \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  fraud-train
```

### API Only
```bash
# Build
docker build -f Dockerfile.api -t fraud-api .

# Run (requires MLflow server running)
docker run -d \
  --name fraud-api \
  --network host \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  fraud-api
```

## Environment Variables

### MLflow Server
- `MLFLOW_BACKEND_STORE_URI`: SQLite database location (default: `/mlflow/mlflow-db/mlflow.db`)
- `MLFLOW_ARTIFACTS_DESTINATION`: Artifact storage location (default: `/mlflow/mlruns`)

### Training Container
- `MLFLOW_TRACKING_URI`: MLflow server URL (default: `http://mlflow:5000` in compose)
- `MODEL_NAME`: Model registry name (default: `fraud_detector`)
- `DECISION_THRESHOLD`: Fraud detection threshold (default: `0.640`)

### API Container
- `MLFLOW_TRACKING_URI`: MLflow server URL (default: `http://mlflow:5000` in compose)
- `MODEL_NAME`: Model registry name (default: `fraud_detector`)
- `API_KEY`: API authentication key (default: `dev-key`)

## Data Persistence

The docker-compose setup uses named volumes to persist:
- `mlflow-db`: MLflow metadata (experiments, runs, model registry)
- `mlflow-artifacts`: Model artifacts and other logged files

To remove all data and start fresh:
```bash
docker-compose down -v
```

## Development Workflow

1. **Initial Setup:**
   ```bash
   # Start MLflow server and API
   docker-compose up mlflow api -d
   
   # Train the initial model
   docker-compose run --rm train
   
   # Verify model is registered
   # Open http://localhost:5000 and check Models tab
   ```

2. **Retrain Model:**
   ```bash
   # Train new version
   docker-compose run --rm train
   
   # Reload API to use new model
   curl -X POST http://localhost:8000/reload \
     -H "x-api-key: dev-key"
   ```

3. **Test API:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Make prediction (example)
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -H "x-api-key: dev-key" \
     -d '{"Time":0,"Amount":149.62,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02}'
   ```

## Monitoring

Access Prometheus metrics at http://localhost:8000/metrics

## Troubleshooting

- **MLflow UI not accessible**: Check if container is healthy: `docker-compose ps`
- **Training fails**: Ensure data file exists at `./data/data_raw.csv`
- **API can't find model**: Verify model is in Production stage in MLflow UI
- **Port conflicts**: Modify port bindings in docker-compose.yml if needed