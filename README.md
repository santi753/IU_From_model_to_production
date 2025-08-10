# Government Aid Application Fraud Detection MLOps System

A comprehensive MLOps pipeline for detecting fraud in government aid applications that demonstrates model training, serving, monitoring, and automated retraining using MLflow, FastAPI, Docker, and GitHub Actions.

**Note: This system uses a credit card fraud detection dataset and implementation as a demonstration/proxy for government aid application fraud detection. The MLOps architecture and techniques shown here can be adapted for actual government aid fraud detection systems.**

## 🏗️ System Architecture

This project implements a complete MLOps workflow designed for a government agency supporting people in need with financial and consultancy programs. The system demonstrates how to automatically detect fraudulent applications to ensure legitimate beneficiaries receive support while preventing misuse of public resources.

![Fraud Detection MLOps System Architecture](Diagram%20marmaid.png)

**Key Components:**
- **Model Training**: RandomForest classifier with SMOTE balancing and feature engineering for fraud detection
- **Model Registry**: MLflow for experiment tracking and model versioning
- **API Service**: FastAPI-based prediction service with authentication and monitoring
- **Drift Detection**: Automated data drift monitoring using Population Stability Index (PSI) to adapt to changing political and social conditions
- **Containerization**: Docker-based deployment with docker-compose orchestration
- **CI/CD Pipeline**: GitHub Actions workflow for automated training and deployment
- **Monitoring**: Prometheus metrics integration for production monitoring

## 📊 Dataset

The system expects a fraud detection dataset with the following structure:
- `Time`: Transaction/Application timestamp
- `Amount`: Transaction/Application amount
- `V1-V28`: PCA-transformed features (anonymized characteristics)
- `Class`: Target variable (0=Normal, 1=Fraud)

For demonstration purposes, the system uses the Credit Card Fraud Detection dataset structure, which serves as a proxy for government aid application fraud patterns.

You can either:
1. Download the "Credit Card Fraud Detection" dataset from Kaggle: https://www.kaggle.com/datasets/divyaraj2006/credit-card-fraud-detection and place it at `data/data_raw.csv`
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

All prediction endpoints require an API key in the header for security compliance:
```bash
-H "x-api-key: dev-key"
```

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Fraud Prediction
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

#### Batch Fraud Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{
    "records": [
      {
        "Time": 0,
        "Amount": 149.62,
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
        "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
        "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
        "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
      },
      {
        "Time": 100,
        "Amount": 200.50,
        "V1": 0.5, "V2": 1.2, "V3": -0.8, "V4": 0.3,
        "V5": -0.1, "V6": 0.9, "V7": 0.15, "V8": 0.25,
        "V9": 0.45, "V10": 0.12, "V11": -0.33, "V12": -0.41,
        "V13": -0.77, "V14": -0.22, "V15": 1.11, "V16": -0.35,
        "V17": 0.33, "V18": 0.08, "V19": 0.52, "V20": 0.31,
        "V21": -0.05, "V22": 0.35, "V23": -0.14, "V24": 0.09,
        "V25": 0.17, "V26": -0.23, "V27": 0.16, "V28": -0.04
      }
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
- **Training Container**: On-demand model training for fraud detection
- **API Container**: FastAPI prediction service for application assessment
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

# Retrain with new application data
docker-compose run --rm train

# View logs
docker-compose logs api

# Clean up
docker-compose down -v
```

## ⚙️ GitHub Actions CI/CD

The repository includes a comprehensive GitHub Actions workflow (`.github/workflows/mlops.yml`) that supports automated model management as political and social conditions change:

### Trigger Types

1. **Scheduled Runs**
   - Daily at 3 AM UTC: Drift monitoring for changing application patterns
   - Monthly (1st day): Automatic retraining to adapt to policy changes

2. **Manual Triggers**
   - Standard run: Current month processing
   - Demo mode: 12-month simulation for testing system adaptability

### Workflow Features

- **Automated Data Generation**: Creates synthetic aid application datasets if none exist
- **Drift Detection**: PSI-based monitoring to detect changes in application patterns
- **Conditional Training**: Retrains models when application data drift exceeds thresholds
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
| `API_KEY` | Production API key for agency systems |
| `DRIFT_PSI_THRESHOLD` | Drift detection threshold |

## 📈 Monitoring and Drift Detection

### Metrics Available

The system exposes Prometheus metrics for government oversight:
- `fraud_api_requests_total`: Request counts by endpoint and status
- `fraud_api_request_duration_seconds`: Request latency for application processing
- `fraud_api_predictions_total`: Prediction counts by fraud/legitimate classification
- `fraud_api_model_version`: Current model version in production

### Drift Detection

The system includes automated drift detection using Population Stability Index (PSI) to adapt to changing political and social conditions:

```bash
# Check drift between application datasets
python mlops/drift_check.py new_applications.csv --ref-data reference_applications.csv

# With environment variables
REF_DATA=data/data_raw.csv DRIFT_PSI_THRESHOLD=0.1 \
  python mlops/drift_check.py data/simulated/month_06.csv
```

**PSI Interpretation:**
- `< 0.1`: No significant change in application patterns
- `0.1 - 0.2`: Moderate change (monitor closely)
- `> 0.2`: Significant change (triggers automatic retraining)

### Data Simulation

Generate monthly datasets with progressive drift to simulate changing social conditions:

```bash
# Generate 12 months of simulated fraud detection data
python mlops/simulate_monthly_data.py

# Custom parameters for different scenarios
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
- API endpoint functionality for application processing
- Authentication and authorization for government security
- Batch application processing
- Error handling and validation
- Drift detection algorithms for policy changes
- Model loading and prediction accuracy

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
   export API_KEY="production-government-secret-key"
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
├── .github/workflows/             # CI/CD pipeline
│   └── mlops.yml                 # GitHub Actions workflow
├── artifacts/                     # Model artifacts
│   ├── clf.pkl                   # Trained classifier model
│   ├── feature_order.json        # Feature ordering for predictions
│   ├── scaler.pkl                # Data scaler for preprocessing
│   ├── threshold.txt             # Decision threshold configuration
│   └── trained_features.json     # Training feature metadata
├── data/                          # Fraud detection data storage
│   ├── simulated/                # Generated monthly datasets
│   └── data_raw.csv              # Original credit card fraud dataset
├── mlops/                         # Core MLOps components
│   ├── api_main.py               # FastAPI prediction service
│   ├── drift_check.py            # Data drift detection for changing conditions
│   ├── simulate_monthly_data.py  # Monthly data simulation
│   └── train_and_register.py     # Model training pipeline for fraud detection
├── notebook/                      # Jupyter notebooks for development
│   ├── fraud_detection_model.pkl # Model development artifacts
│   └── model_creation.ipynb      # Model creation and experimentation
├── prometheus/                    # Monitoring configuration
│   └── prometheus.yml            # Prometheus monitoring config
├── tests/                         # Test suite
│   ├── test_api.py               # API endpoint tests
│   ├── test_drift_check.py       # Drift detection tests
│   └── test_wrapper.py           # Model wrapper tests
├── .dockerignore                  # Docker ignore file
├── .gitattributes                 # Git attributes configuration
├── .gitignore                     # Git ignore file
├── Dockerfile.api                 # API service container
├── Dockerfile.mlflow              # MLflow server container
├── Dockerfile.train               # Training container
├── LICENSE                        # Project license
├── README-Docker.md               # Docker-specific documentation
├── README.md                      # This file
├── docker-compose.yml             # Container orchestration
├── mlflow.db                      # MLflow SQLite database
└── requirements.txt               # Python dependencies 
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

---

*This system enables government agencies to automatically detect fraud in aid applications while ensuring legitimate beneficiaries receive the support they need. The MLOps pipeline adapts to changing political and social conditions through automated drift detection and retraining.*