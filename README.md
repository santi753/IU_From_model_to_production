# IU_From_model_to_production

This project demonstrates a simple MLflow workflow for training and serving a credit card fraud detection model.

## Training

Run `python mlops/train_and_register.py` to train the model and register it in the local MLflow tracking server. The script automatically downloads the dataset if `data/data_raw.csv` is missing.

## API

After training, launch the API with:

```bash
uvicorn mlops.api_main:app --reload
```

Use the `API_KEY` environment variable (defaults to `dev-key`) when calling the endpoints.

