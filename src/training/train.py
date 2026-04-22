import os
import hashlib
import json
import time 
from datetime import datetime, timezone
import gcsfs
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

from src.configs.loader import get_path, load_config
from src.constants import PROJECT_ROOT
from src.training.model_factory import build_model, fit_model, log_model_by_type
from src.utils.logger import get_logger

logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")

def normalize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature dtypes for model training."""
    df = df.copy()
    object_columns = df.select_dtypes(include=["object"]).columns
    for col in object_columns:
        df[col] = df[col].astype("category")
    return df

def load_training_data(train_file: str, val_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and validation data."""
    if train_file.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        df_train = pd.read_parquet(train_file, filesystem=fs)
        df_val = pd.read_parquet(val_file, filesystem=fs)
    else:
        df_train = pd.read_parquet(train_file)
        df_val = pd.read_parquet(val_file)
    return df_train, df_val

def build_training_cost_summary(started_at_utc, finished_at_utc, duration_seconds) -> dict:
    cost_cfg = ENV_CFG.get("costs", {}).get("training", {})
    hourly_rate = float(cost_cfg.get("estimated_hourly_rate", 0.0))
    estimated_cost = (duration_seconds / 3600.0) * hourly_rate if cost_cfg.get("enabled") else 0.0
    return {
        "training_duration_seconds": round(duration_seconds, 3),
        "estimated_training_cost": round(estimated_cost, 6),
        "currency": cost_cfg.get("currency", "EUR")
    }

def get_or_create_experiment(project_name: str):
    if not mlflow.get_experiment_by_name(project_name):
        mlflow.create_experiment(project_name)
    mlflow.set_experiment(project_name)

def train(train_file: str | None = None, val_file: str | None = None):
    """Main training task for Churn Classification."""
    data_path = get_path("splits")
    train_file = train_file or f"{data_path}/train.parquet"
    val_file = val_file or f"{data_path}/val.parquet"

    logger.info(f"Loading training data from: {train_file}")
    df_train, df_val = load_training_data(train_file, val_file)

    data_cfg = TRAIN_CFG["data"]
    model_cfg = TRAIN_CFG["model"]
    target_column = data_cfg["target_column"].lower().replace(" ", "_")
    model_type = model_cfg["type"]
    
    # Define features by dropping the target and other unneeded columns
    drop_cols = [target_column] + TRAIN_CFG.get("features", {}).get("drop_columns", [])
    
    X_train = normalize_feature_dtypes(df_train.drop(columns=drop_cols, errors="ignore"))
    X_val = normalize_feature_dtypes(df_val.drop(columns=drop_cols, errors="ignore"))

    # Target Mapping for Classification (Yes=1, No=0)
    def robust_map(series):
        # 1. Convert to strings & force lower case
        # 2. Apply map
        # 3. If NaNs do occur (because a value was neither 'Yes' nor 'No'), fill with 0
        return series.astype(str).str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

    y_train = robust_map(df_train[target_column])
    y_val = robust_map(df_val[target_column])

    # MLflow Setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    get_or_create_experiment(ENV_CFG.get("project_name", "churn-prediction"))

    with mlflow.start_run() as run:
        logger.info(f"Starting model training. Run ID: {run.info.run_id}")
        seed = ENV_CFG.get("random_seed")
        
        # Build and Fit Model
        model = build_model(model_cfg, seed=seed)
        
        start_perf = time.perf_counter()
        start_time = datetime.now(timezone.utc)
        
        fit_model(model, model_type, X_train, y_train, X_val, y_val)
        
        duration = time.perf_counter() - start_perf
        cost_summary = build_training_cost_summary(start_time, datetime.now(timezone.utc), duration)

        # Metrics Calculation
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        
        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
        
        if hasattr(model, "predict_proba"):
            metrics["roc_auc"] = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        # Logging
        mlflow.log_params(model_cfg.get("params", {}))
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("duration_seconds", duration)

        log_model_by_type(
            model=model,
            model_type=model_type,
            input_example=X_val.iloc[:3],
            metadata={"target": target_column}
        )

        logger.info(f"Train complete. Metrics: {metrics}")
        
        # Local export for Dev/Stage
        if ENV_CFG["environment"] != "prod":
            model_path = os.path.join(get_path("models"), "model.joblib")
            import joblib
            joblib.dump(model, model_path)
            logger.info(f"Model saved locally to {model_path}")

    return model, run.info.run_id

if __name__ == "__main__":
    train()