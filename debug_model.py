import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import pickle

# Set the same environment variables
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
MODEL_NAME = "fraud_detector"

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

try:
    # Try to load the model
    print(f"\nTrying to load model: {MODEL_NAME}/Production...")
    uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(uri)
    print("✅ Model loaded successfully!")
    
    # Try to get model version
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if versions:
        print(f"Model version: {versions[0].version}")
    
    # Check model artifacts
    print("\nChecking model artifacts...")
    model_path = model._model_impl.python_model._model_meta.model_uri
    print(f"Model path: {model_path}")
    
    # Test prediction with sample data
    print("\nTesting prediction...")
    test_data = {
        "Time": 0.0,
        "Amount": 149.62,
        "V1": -1.359807134,
        "V2": -0.072781173,
        "V3": 2.536346738,
        "V4": 1.378155224,
        "V5": -0.338320769,
        "V6": 0.462387778,
        "V7": 0.239598554,
        "V8": 0.098697901,
        "V9": 0.363786969,
        "V10": 0.090794172,
        "V11": -0.551599533,
        "V12": -0.617800856,
        "V13": -0.991389847,
        "V14": -0.311169354,
        "V15": 1.468176972,
        "V16": -0.470400525,
        "V17": 0.207971242,
        "V18": 0.025790619,
        "V19": 0.403992960,
        "V20": 0.251412098,
        "V21": -0.018306778,
        "V22": 0.277837576,
        "V23": -0.110473910,
        "V24": 0.066928075,
        "V25": 0.128539358,
        "V26": -0.189114844,
        "V27": 0.133558377,
        "V28": -0.021053053
    }
    
    df = pd.DataFrame([test_data])
    
    # Let's debug what the model expects
    print("\nInput DataFrame columns:", df.columns.tolist())
    print("Input DataFrame shape:", df.shape)
    
    prediction = model.predict(df)
    print("✅ Prediction successful!")
    print(f"Result: {prediction}")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Try to inspect the model's expected features
    print("\n\nTrying to inspect model's expected features...")
    try:
        # Load the clf.pkl directly to see what features it expects
        import glob
        pkl_files = glob.glob("mlruns/**/artifacts/*/clf.pkl", recursive=True)
        if pkl_files:
            print(f"Found pickle file: {pkl_files[0]}")
            with open(pkl_files[0], 'rb') as f:
                clf = pickle.load(f)
                if hasattr(clf, 'feature_names_in_'):
                    print(f"Model expects these features: {clf.feature_names_in_}")
                    print(f"Number of features: {len(clf.feature_names_in_)}")
    except Exception as e2:
        print(f"Could not inspect pickle file: {e2}")