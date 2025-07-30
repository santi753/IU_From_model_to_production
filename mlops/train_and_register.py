# mlops/train_and_register.py
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.pyfunc
import cloudpickle

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "data_raw.csv"),
    )
)
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detector")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", 0.640))

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", "./mlruns")

mlflow.set_tracking_uri(TRACKING_URI)

# -----------------------------
# Custom PyFunc Model
# -----------------------------
class FraudModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps a fitted scaler + classifier and performs:
    - scale Amount, Time
    - create interaction features
    - predict_proba -> apply threshold -> output probability + label
    """
    def load_context(self, context):
        import pickle
        with open(context.artifacts["clf"], "rb") as f:
            self.clf = pickle.load(f)
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)
        with open(context.artifacts["feature_order"], "r") as f:
            self.feature_order = json.load(f)  # original input order (30 cols)
        with open(context.artifacts["threshold"], "r") as f:
            self.threshold = float(f.read().strip())
        # Load the exact feature names the model was trained with
        with open(context.artifacts["trained_features"], "r") as f:
            self.trained_features = json.load(f)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure all required columns exist
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Work with a copy of the input columns
        X = df[self.feature_order].copy()

        # Scale Amount and Time (BEFORE adding engineered features)
        # Create a temporary DataFrame for scaling
        scale_df = X[["Amount", "Time"]].copy()
        scaled_values = self.scaler.transform(scale_df)
        X["Amount"] = scaled_values[:, 0]
        X["Time"] = scaled_values[:, 1]

        # Add engineered features (using scaled values)
        X["Amount_Time"] = X["Amount"] * X["Time"]
        X["V1_V2"] = X["V1"] * X["V2"]
        X["V3_V4"] = X["V3"] * X["V4"]
        
        # Ensure columns are in the exact order the model expects
        X = X[self.trained_features]
        
        return X

    def predict(self, context, model_input):
        if isinstance(model_input, dict):  # single row as dict
            model_input = pd.DataFrame([model_input])
        elif isinstance(model_input, list):
            model_input = pd.DataFrame(model_input)
        elif not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X = self._preprocess(model_input)
        proba = self.clf.predict_proba(X)[:, 1]
        label = (proba >= self.threshold).astype(int)
        return pd.DataFrame({"probability": proba, "label": label})


def main():
    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(DATA_PATH)
    X_full = df.drop("Class", axis=1)
    y_full = df["Class"]

    # columns (30 original)
    feature_order: List[str] = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

    # -------------------------
    # Split
    # -------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # -------------------------
    # Fit scaler on train only (BEFORE creating copies)
    # -------------------------
    scaler = StandardScaler().fit(X_train[["Amount", "Time"]])

    # Apply scaling and create engineered features
    def prepare_features(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """Apply scaling and add engineered features"""
        frame = frame.copy()
        # Scale Amount and Time
        frame[["Amount", "Time"]] = scaler.transform(frame[["Amount", "Time"]])
        # Add engineered features
        frame["Amount_Time"] = frame["Amount"] * frame["Time"]
        frame["V1_V2"] = frame["V1"] * frame["V2"]
        frame["V3_V4"] = frame["V3"] * frame["V4"]
        return frame

    X_train = prepare_features(X_train, scaler)
    X_val = prepare_features(X_val, scaler)
    X_test = prepare_features(X_test, scaler)

    # -------------------------
    # Balance with SMOTE (train only)
    # -------------------------
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # -------------------------
    # Train model (RandomForest as in notebook)
    # -------------------------
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_bal, y_train_bal)
    
    # Store the actual feature names used during training
    trained_features = X_train_bal.columns.tolist()
    print(f"Model trained with {len(trained_features)} features: {trained_features}")

    # -------------------------
    # Log to MLflow as pyfunc + register
    # -------------------------
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run() as run:
        # metrics you might want to log
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_param("threshold", DECISION_THRESHOLD)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("num_features", len(trained_features))

        # Save artifacts needed by wrapper
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        import pickle
        with open(artifacts_dir / "clf.pkl", "wb") as f:
            pickle.dump(clf, f)
        with open(artifacts_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(artifacts_dir / "feature_order.json", "w") as f:
            json.dump(feature_order, f)
        with open(artifacts_dir / "threshold.txt", "w") as f:
            f.write(str(DECISION_THRESHOLD))
        # Save the trained feature names IN THE EXACT ORDER
        with open(artifacts_dir / "trained_features.json", "w") as f:
            json.dump(trained_features, f)

        # Input/Output signature
        from mlflow.models.signature import ModelSignature
        from mlflow.types import Schema, ColSpec

        input_schema = Schema([ColSpec("double", c) for c in feature_order])
        output_schema = Schema([ColSpec("double", "probability"), ColSpec("integer", "label")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Example for UI - use only the original features (wrapper will add engineered ones)
        input_example = X_full.head(1)[feature_order].to_dict(orient="records")[0]

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=FraudModelWrapper(),
            artifacts={
                "clf": str(artifacts_dir / "clf.pkl"),
                "scaler": str(artifacts_dir / "scaler.pkl"),
                "feature_order": str(artifacts_dir / "feature_order.json"),
                "threshold": str(artifacts_dir / "threshold.txt"),
                "trained_features": str(artifacts_dir / "trained_features.json"),
            },
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )

        # Get the version number correctly
        if hasattr(model_info, 'registered_model_version'):
            if hasattr(model_info.registered_model_version, 'version'):
                version = model_info.registered_model_version.version
            else:
                version = model_info.registered_model_version
        else:
            # Fallback: query the latest version
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(MODEL_NAME)
            version = latest_versions[-1].version if latest_versions else "1"

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=str(version),
            stage="Production",
            archive_existing_versions=True,
        )

        print(f"Model registered as {MODEL_NAME} v{version} and promoted to Production.")


if __name__ == "__main__":
    main()