from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import fsspec
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.configs.loader import ensure_dir, get_path, load_config
from src.training.explainability import log_feature_importance, log_shap_summary
from src.training.model_factory import build_model, fit_model, log_model_by_type
from src.utils.logger import get_logger


logger = get_logger(__name__)

ENV_CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def normalize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize feature dtypes for model training.
    """
    df = df.copy()
    object_columns = df.select_dtypes(include=["object"]).columns

    for col in object_columns:
        df[col] = df[col].astype("category")

    return df


def load_training_data(
    train_file: str,
    val_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation data from local storage or GCS.
    """
    df_train = pd.read_parquet(train_file)
    df_val = pd.read_parquet(val_file)

    return df_train, df_val


def build_training_cost_summary(
    started_at_utc,
    finished_at_utc,
    duration_seconds,
) -> dict:
    """
    Build a lightweight estimated cost summary for the training run.
    """
    cost_cfg = ENV_CFG.get("costs", {}).get("training", {})
    hourly_rate = float(cost_cfg.get("estimated_hourly_rate", 0.0))
    estimated_cost = (
        (duration_seconds / 3600.0) * hourly_rate
        if cost_cfg.get("enabled")
        else 0.0
    )

    return {
        "training_duration_seconds": round(duration_seconds, 3),
        "estimated_training_cost": round(estimated_cost, 6),
        "currency": cost_cfg.get("currency", "EUR"),
    }


def get_or_create_experiment(project_name: str) -> None:
    """
    Select an existing MLflow experiment or create it if missing.
    """
    if not mlflow.get_experiment_by_name(project_name):
        mlflow.create_experiment(project_name)

    mlflow.set_experiment(project_name)


def save_feature_schema(
    df: pd.DataFrame,
    path: str = "models/feature_schema.json",
) -> dict:
    """
    Save the final training feature schema used by the model.

    The schema path can point to local storage or GCS. The returned schema is
    used at inference time to align incoming features.
    """
    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    parent = str(PurePosixPath(path).parent)
    ensure_dir(parent)

    with fsspec.open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema


def _log_feature_schema_artifact(feature_schema: dict) -> None:
    """
    Log the feature schema to MLflow from a temporary local file.

    MLflow `log_artifact` expects a local path, even when the persistent schema
    itself is stored on GCS.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "feature_schema.json"

        with local_path.open("w", encoding="utf-8") as f:
            json.dump(feature_schema, f, indent=2)

        mlflow.log_artifact(str(local_path), artifact_path="feature_schema")


def find_best_threshold(
    y_true,
    y_proba,
    *,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Find the best classification threshold on validation data.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted churn probabilities.
        metric: Optimization metric. Currently supports "f1".
        thresholds: Candidate thresholds.

    Returns:
        Best threshold and best metric score.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def train(
    train_file: str | None = None,
    val_file: str | None = None,
):
    """
    Train the churn classification model and log artifacts to MLflow.
    """
    data_path = get_path("splits")
    train_file = train_file or f"{data_path}/train.parquet"
    val_file = val_file or f"{data_path}/val.parquet"

    logger.info(f"Loading training data from: {train_file}")
    df_train, df_val = load_training_data(train_file, val_file)

    data_cfg = TRAIN_CFG["data"]
    model_cfg = TRAIN_CFG["model"]
    target_column = data_cfg["target_column"].lower().replace(" ", "_")
    model_type = model_cfg["type"]

    drop_cols = [target_column] + TRAIN_CFG.get("features", {}).get("drop_columns", [])

    X_train = normalize_feature_dtypes(
        df_train.drop(columns=drop_cols, errors="ignore")
    )
    X_val = normalize_feature_dtypes(
        df_val.drop(columns=drop_cols, errors="ignore")
    )

    models_path = get_path("models")
    feature_schema_path = f"{models_path}/feature_schema.json"
    feature_schema = save_feature_schema(X_train, feature_schema_path)

    def robust_map(series):
        """
        Map churn labels to binary targets.
        """
        return (
            series.astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

    y_train = robust_map(df_train[target_column])
    y_val = robust_map(df_val[target_column])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    get_or_create_experiment(ENV_CFG.get("project_name", "churn-prediction"))

    with mlflow.start_run() as run:
        logger.info(f"Starting model training. Run ID: {run.info.run_id}")
        seed = ENV_CFG.get("random_seed")

        model = build_model(model_cfg, seed=seed)

        start_perf = time.perf_counter()
        start_time = datetime.now(timezone.utc)

        fit_model(model, model_type, X_train, y_train, X_val, y_val)

        duration = time.perf_counter() - start_perf
        cost_summary = build_training_cost_summary(
            start_time,
            datetime.now(timezone.utc),
            duration,
        )

        mlflow.log_metric(
            "training_duration_seconds",
            cost_summary["training_duration_seconds"],
        )
        mlflow.log_metric(
            "estimated_training_cost",
            cost_summary["estimated_training_cost"],
        )
        mlflow.log_param("cost_currency", cost_summary["currency"])

        log_feature_importance(model, list(X_train.columns))

        try:
            sample = X_train.sample(min(500, len(X_train)), random_state=42)
            log_shap_summary(model, sample)
        except Exception as e:
            logger.warning(f"SHAP logging skipped: {e}")

        y_proba = model.predict_proba(X_val)[:, 1]

        best_threshold, best_threshold_score = find_best_threshold(
            y_val,
            y_proba,
            metric="f1",
        )

        y_pred = (y_proba >= best_threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_proba),
            "brier_score": brier_score_loss(y_val, y_proba),
            "decision_threshold": best_threshold,
        }

        mlflow.log_params(model_cfg.get("params", {}))
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)
        mlflow.log_param("decision_threshold", best_threshold)
        mlflow.log_metric("duration_seconds", duration)
        _log_feature_schema_artifact(feature_schema)
        mlflow.log_param("n_features", len(feature_schema["columns"]))

        mlflow.log_params(
            {
                "customer_value": ENV_CFG.get("decision", {}).get("customer_value"),
                "cost_discount": ENV_CFG.get("decision", {}).get("cost_discount"),
                "cost_contact": ENV_CFG.get("decision", {}).get("cost_contact"),
                "discount_uplift": ENV_CFG.get("decision", {}).get("discount_uplift"),
                "contact_uplift": ENV_CFG.get("decision", {}).get("contact_uplift"),
            }
        )

        log_model_by_type(
            model=model,
            model_type=model_type,
            input_example=X_val.iloc[:3],
            metadata={
                "target": target_column,
            },
        )

        logger.info(f"Train complete. Metrics: {metrics}")

        if ENV_CFG["environment"] != "prod":
            model_path = f"{models_path}/model.joblib"
            ensure_dir(models_path)
            joblib.dump(model, model_path)
            logger.info(f"Model saved locally to {model_path}")

    return model, run.info.run_id


if __name__ == "__main__":
    train()