import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from mlops.train_and_register import FraudModelWrapper

ARTIFACTS = {
    "clf": "artifacts/clf.pkl",
    "scaler": "artifacts/scaler.pkl",
    "feature_order": "artifacts/feature_order.json",
    "threshold": "artifacts/threshold.txt",
    "trained_features": "artifacts/trained_features.json",
}

def _load_wrapper():
    wrapper = FraudModelWrapper()
    ctx = type("ctx", (), {"artifacts": ARTIFACTS})()
    wrapper.load_context(ctx)
    return wrapper

def test_preprocess_adds_engineered_columns():
    wrapper = _load_wrapper()
    with open(ARTIFACTS["feature_order"]) as f:
        features = json.load(f)
    data = {feat: 0.0 for feat in features}
    df = pd.DataFrame([data])
    processed = wrapper._preprocess(df)
    assert list(processed.columns) == wrapper.trained_features
    assert "Amount_Time" in processed.columns
    assert "V1_V2" in processed.columns
    assert "V3_V4" in processed.columns


def test_wrapper_predict_returns_dataframe():
    wrapper = _load_wrapper()
    with open(ARTIFACTS["feature_order"]) as f:
        features = json.load(f)
    data = {feat: 0.0 for feat in features}
    df = pd.DataFrame([data])
    result = wrapper.predict(None, df)
    assert list(result.columns) == ["probability", "label"]
    assert len(result) == 1
