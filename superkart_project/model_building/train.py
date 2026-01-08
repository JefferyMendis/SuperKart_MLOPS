"""
Production training script for SuperKart Sales Forecasting.

Key features:
- Robust XGBoost training using DMatrix (version-safe)
- Early stopping enabled
- sklearn-compatible wrapper for Pipeline integration
- MLflow logging
- Hugging Face Dataset & Model Hub integration

Required environment variables:
- HF_TOKEN
- MLFLOW_TRACKING_URI
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow

import xgboost as xgb
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# ENVIRONMENT CHECKS
# =========================================================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be set.")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI must be set.")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("superkart-sales-prod")

HF_OWNER = "JefferyMendis"
DATASET_REPO = f"{HF_OWNER}/superkart"

REMOTE_FILES = {
    "X_train": "X_train.csv",
    "X_test": "X_test.csv",
    "y_train": "y_train.csv",
    "y_test": "y_test.csv",
}

OUT_DIR = Path("superkart_project/model_building")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_PATH = OUT_DIR / os.getenv("PROD_MODEL_FILENAME", "model_prod.joblib")

# =========================================================
# XGBOOST HYPERPARAMETERS
# =========================================================
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", 300))
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", 6))
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", 0.05))
XGB_N_JOBS = int(os.getenv("XGB_N_JOBS", -1))
EARLY_STOPPING_ROUNDS = int(os.getenv("XGB_EARLY_STOPPING_ROUNDS", 30))
VAL_SIZE = float(os.getenv("PROD_VAL_SIZE", 0.1))
XGB_VERBOSE = int(os.getenv("XGB_VERBOSE", 10))

# =========================================================
# HELPERS
# =========================================================
def download_csv(repo_id: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    return pd.read_csv(path)

# =========================================================
# LOAD DATA
# =========================================================
print("Downloading SuperKart processed dataset splits...")

X_train = download_csv(DATASET_REPO, REMOTE_FILES["X_train"])
y_train = download_csv(DATASET_REPO, REMOTE_FILES["y_train"]).squeeze()
X_test = download_csv(DATASET_REPO, REMOTE_FILES["X_test"])
y_test = download_csv(DATASET_REPO, REMOTE_FILES["y_test"]).squeeze()

y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# =========================================================
# FEATURE SCHEMA (SUPERKART)
# =========================================================
numeric_features = [
    c for c in [
        "Product_Weight",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Establishment_Year",
    ]
    if c in X_train.columns
]

categorical_features = [
    c for c in [
        "Product_Sugar_Content",
        "Product_Type",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
    ]
    if c in X_train.columns
]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# =========================================================
# PREPROCESSOR
# =========================================================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    remainder="drop",
)

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VAL_SIZE,
    random_state=42,
)

preprocessor.fit(X_tr)

X_tr_t = preprocessor.transform(X_tr)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

# =========================================================
# XGBOOST TRAINING (DMATRIX)
# =========================================================
print("Training XGBoost using DMatrix (early stopping enabled)...")

dtrain = xgb.DMatrix(X_tr_t, label=y_tr)
dval = xgb.DMatrix(X_val_t, label=y_val)

xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": XGB_MAX_DEPTH,
    "learning_rate": XGB_LEARNING_RATE,
    "eval_metric": "rmse",
    "seed": 42,
    "nthread": XGB_N_JOBS,
}

evals_result = {}

booster = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=XGB_N_ESTIMATORS,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    evals_result=evals_result,
    verbose_eval=XGB_VERBOSE,
)

print("Best iteration:", booster.best_iteration)

# =========================================================
# SKLEARN-COMPATIBLE WRAPPER
# =========================================================
class XGBBoosterRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, booster):
        self.booster = booster

    def fit(self, X, y=None):
        # Booster already trained
        return self

    def predict(self, X):
        dmatrix = xgb.DMatrix(X)
        return self.booster.predict(dmatrix)

# =========================================================
# FINAL PIPELINE
# =========================================================
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgbregressor", XGBBoosterRegressor(booster)),
])

# =========================================================
# EVALUATION
# =========================================================
print("Evaluating model on test set...")

y_pred = final_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.4f}")

# =========================================================
# MLFLOW LOGGING
# =========================================================
with mlflow.start_run():
    mlflow.log_param("mode", "prod")
    mlflow.log_params({
        "n_estimators": XGB_N_ESTIMATORS,
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE,
        "val_size": VAL_SIZE,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(booster.best_iteration),
    })

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Save training history
    eval_path = OUT_DIR / "evals_result.json"
    with open(eval_path, "w") as f:
        json.dump(evals_result, f)
    mlflow.log_artifact(str(eval_path))

    # Save model
    joblib.dump(final_pipeline, LOCAL_MODEL_PATH)
    mlflow.log_artifact(str(LOCAL_MODEL_PATH))

print("Model logged to MLflow successfully.")

# =========================================================
# UPLOAD TO HUGGING FACE MODEL HUB
# =========================================================
HF_MODEL_REPO = os.getenv("PROD_MODEL_REPO", f"{HF_OWNER}/superkart-sales-model")
api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print("Model repo exists.")
except RepositoryNotFoundError:
    print("Creating model repo:", HF_MODEL_REPO)
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=str(LOCAL_MODEL_PATH),
    path_in_repo=LOCAL_MODEL_PATH.name,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)

print("✅ SuperKart production training completed successfully.")

