# Credit Card Fraud Detection MLOps System

A comprehensive MLOps pipeline for credit card fraud detection that demonstrates model training, serving, monitoring, and automated retraining using MLflow, FastAPI, Docker, and GitHub Actions.

## 🏗️ System Architecture

This project implements a complete MLOps workflow with the following components:

- **Model Training**: RandomForest classifier with SMOTE balancing and feature engineering
- **Model Registry**: MLflow for experiment tracking and model versioning
- **API Service**: FastAPI-based prediction service with authentication and monitoring
- **Drift Detection**: Automated data drift monitoring using Population Stability Index (PSI)
- **Containerization**: Docker-based deployment with docker-compose orchestration
- **CI/CD Pipeline**: GitHub Actions workflow for automated training and deployment
- **Monitoring**: Prometheus metrics integration for production monitoring

## 📊 Dataset

The system expects a credit card fraud dataset with the following structure:
- `Time`: Transaction timestamp
- `Amount`: Transaction amount
- `V1-V28`: PCA-transformed features
- `Class`: Target variable (0=Normal, 1=Fraud)

You can either:
1. Download the "Credit Card Fraud Detection" dataset from Kaggle and place it at `data/data_raw.csv`
2. Use Git LFS: `git lfs pull` (if the repository is configured with LFS)
3. The system will generate synthetic data if no dataset is found

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Docker** and **Docker Compose**
- **Git** (optional, for cloning)

### Option 1: Local Development Setup

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd iu_fraud
   python -m venv .venv
   
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   
   # Linux/macOS
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Train and Register Model**
   ```bash
   python mlops/train_and_register.py
   ```

3. **Start MLflow UI** (optional)
   ```bash
   mlflow ui
   # Visit http://localhost:5000
   ```

4. **Start API Service**
   ```bash
   uvicorn mlops.api_main:app --reload
   # API available at http://localhost:8000
   ```

### Option 2: Docker Setup

1. **Quick Start with Docker Compose**
   ```bash
   # Start all services
   docker-compose up --build -d
   
   # Train initial model
   docker-compose run --rm train
   
   # View services
   docker-compose ps
   ```

2. **Access Services**
   - **MLflow UI**: http://localhost:5000
   - **API Service**: http://localhost:8000
   - **API Health**: http://localhost:8000/health
   - **Metrics**: http://localhost:8000/metrics
   - **Prometheus**: http://localhost:9090

3. **Stop Services**
   ```bash
   docker-compose down
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL | `sqlite:///mlflow.db` |
| `MLFLOW_ARTIFACT_LOCATION` | Artifact storage location | `./mlruns` |
| `MODEL_NAME` | Registered model name | `fraud_detector` |
| `DECISION_THRESHOLD` | Fraud detection threshold | `0.640` |
| `API_KEY` | API authentication key | `dev-key` |
| `DATA_PATH` | Training data file path | `data/data_raw.csv` |
| `DRIFT_PSI_THRESHOLD` | Drift detection threshold | `0.2` |

### Setting Environment Variables

**Windows PowerShell:**
```powershell
$env:API_KEY = "your-secret-key"
$env:DECISION_THRESHOLD = "0.5"
```

**Linux/macOS:**
```bash
export API_KEY="your-secret-key"
export DECISION_THRESHOLD="0.5"
```

## 🔍 API Usage

### Authentication

All prediction endpoints require an API key in the header:
```bash
-H "x-api-key: dev-key"
```

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{
    "Time": 0,
    "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
  }'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{
    "records": [
      {"Time": 0, "Amount": 149.62, "V1": -1.36, ...},
      {"Time": 100, "Amount": 200.50, "V1": 0.5, ...}
    ]
  }'
```

#### Model Information
```bash
curl http://localhost:8000/model-info \
  -H "x-api-key: dev-key"
```

#### Hot Reload Model
```bash
curl -X POST http://localhost:8000/reload \
  -H "x-api-key: dev-key"
```

#### Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

## 🐳 Docker Deployment

### Architecture

The Docker setup includes:
- **MLflow Container**: Centralized tracking server with SQLite backend
- **Training Container**: On-demand model training
- **API Container**: FastAPI prediction service
- **Prometheus Container**: Metrics collection and monitoring

### Individual Container Commands

#### MLflow Server
```bash
docker build -f Dockerfile.mlflow -t fraud-mlflow .
docker run -d --name fraud-mlflow \
  -p 127.0.0.1:5000:5000 \
  -v mlflow-db:/mlflow/mlflow-db \
  -v mlflow-artifacts:/mlflow/mlruns \
  fraud-mlflow
```

#### Training Job
```bash
docker build -f Dockerfile.train -t fraud-train .
docker run --rm \
  --network host \
  -v $(pwd)/data:/app/data:ro \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  fraud-train
```

#### API Service
```bash
docker build -f Dockerfile.api -t fraud-api .
docker run -d --name fraud-api \
  --network host \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  fraud-api
```

### Docker Compose Workflow

```bash
# Start MLflow and API services
docker-compose up mlflow api -d

# Train initial model
docker-compose run --rm train

# Retrain with new data
docker-compose run --rm train

# View logs
docker-compose logs api

# Clean up
docker-compose down -v
```

## ⚙️ GitHub Actions CI/CD

The repository includes a comprehensive GitHub Actions workflow (`.github/workflows/mlops.yml`) that supports:

### Trigger Types

1. **Scheduled Runs**
   - Daily at 3 AM UTC: Drift monitoring
   - Monthly (1st day): Automatic retraining

2. **Manual Triggers**
   - Standard run: Current month processing
   - Demo mode: 12-month simulation

### Workflow Features

- **Automated Data Generation**: Creates synthetic datasets if none exist
- **Drift Detection**: PSI-based monitoring with configurable thresholds
- **Conditional Training**: Retrains models when drift exceeds thresholds
- **Service Health Checks**: Validates all components before processing
- **Comprehensive Testing**: API endpoints and model validation
- **Artifact Collection**: Logs and metrics for debugging

### Running in GitHub Actions

1. **Manual Trigger**
   ```
   Go to Actions tab → MLOps Pipeline → Run workflow
   ```

2. **Demo Mode**
   ```
   Go to Actions tab → MLOps Pipeline → Run workflow
   Select "Run full 12-month demonstration" → true
   ```

### Environment Variables for CI

Set these secrets in your GitHub repository:

| Secret | Description |
|--------|-------------|
| `API_KEY` | Production API key |
| `DRIFT_PSI_THRESHOLD` | Drift detection threshold |

## 📈 Monitoring and Drift Detection

### Metrics Available

The system exposes Prometheus metrics:
- `fraud_api_requests_total`: Request counts by endpoint and status
- `fraud_api_request_duration_seconds`: Request latency
- `fraud_api_predictions_total`: Prediction counts by label
- `fraud_api_model_version`: Current model version

### Drift Detection

The system includes automated drift detection using Population Stability Index (PSI):

```bash
# Check drift between datasets
python mlops/drift_check.py new_data.csv --ref-data reference.csv

# With environment variables
REF_DATA=data/data_raw.csv DRIFT_PSI_THRESHOLD=0.1 \
  python mlops/drift_check.py data/simulated/month_06.csv
```

**PSI Interpretation:**
- `< 0.1`: No significant change
- `0.1 - 0.2`: Moderate change
- `> 0.2`: Significant change (triggers retraining)

### Data Simulation

Generate monthly datasets with progressive drift:

```bash
# Generate 12 months of simulated data
python mlops/simulate_monthly_data.py

# Custom parameters
python mlops/simulate_monthly_data.py \
  --num-months 6 \
  --sample-size 5000 \
  --output-dir data/custom
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies (included in requirements.txt)
pip install pytest

# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_api.py -v
pytest tests/test_drift_check.py -v
```

### Test Coverage

The test suite covers:
- API endpoint functionality
- Authentication and authorization
- Batch prediction handling
- Error handling and validation
- Drift detection algorithms
- Model loading and prediction

## 🛠️ Development Workflow

### Local Development

1. **Setup Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Start Services**
   ```bash
   # Terminal 1: MLflow
   mlflow ui

   # Terminal 2: API
   uvicorn mlops.api_main:app --reload

   # Terminal 3: Train model
   python mlops/train_and_register.py
   ```

3. **Debug Model Issues**
   ```bash
   python debug_model.py
   ```

### Production Deployment

1. **Update Environment Variables**
   ```bash
   export API_KEY="production-secret-key"
   export MLFLOW_TRACKING_URI="postgresql://user:pass@host:5432/mlflow"
   ```

2. **Deploy with Docker**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Monitor Health**
   ```bash
   curl http://production-host:8000/health
   curl http://production-host:8000/metrics
   ```

## 📂 Project Structure

```
iu_fraud/
├── mlops/                          # Core MLOps components
│   ├── api_main.py                # FastAPI prediction service
│   ├── train_and_register.py      # Model training pipeline
│   ├── drift_check.py             # Data drift detection
│   └── simulate_monthly_data.py   # Data simulation
├── data/                          # Data storage
│   ├── data_raw.csv              # Original dataset
│   └── simulated/                # Generated monthly datasets
├── tests/                         # Test suite
│   ├── test_api.py               # API endpoint tests
│   ├── test_drift_check.py       # Drift detection tests
│   └── test_wrapper.py           # Model wrapper tests
├── artifacts/                     # Model artifacts
├── prometheus/                    # Monitoring configuration
├── .github/workflows/             # CI/CD pipeline
│   └── mlops.yml                 # GitHub Actions workflow
├── docker-compose.yml            # Container orchestration
├── Dockerfile.api                # API service container
├── Dockerfile.train              # Training container
├── Dockerfile.mlflow             # MLflow server container
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -am 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Model Not Found Error**
```bash
# Check model registry
mlflow models list

# Retrain model
python mlops/train_and_register.py

# Reload API
curl -X POST http://localhost:8000/reload -H "x-api-key: dev-key"
```

**Docker Container Issues**
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs api
docker-compose logs mlflow

# Restart services
docker-compose restart
```

**Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Modify ports in docker-compose.yml
ports:
  - "127.0.0.1:8001:8000"  # Change 8000 to 8001
```

**Data Issues**
```bash
# Validate data format
python -c "import pandas as pd; df=pd.read_csv('data/data_raw.csv'); print(df.info())"

# Generate synthetic data
python mlops/simulate_monthly_data.py --sample-size 1000
```

### Support

For additional support:
1. Check the [Issues](../../issues) page for known problems
2. Review the GitHub Actions logs for CI/CD issues
3. Examine container logs with `docker-compose logs`
4. Use the debug script: `python debug_model.py`

---

**Built with ❤️ for MLOps best practices**