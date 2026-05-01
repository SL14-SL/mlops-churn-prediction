import os
import socket
from typing import Any

import mlflow
from mlflow import MlflowClient

from src.inference.router import load_registry_model
from src.inference.schema import load_feature_schema
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_tracking_uri(cfg: dict) -> str:
    """
    Determine the MLflow tracking URI based on environment.

    Priority:
    1. MLFLOW_TRACKING_URI env var
    2. Docker service hostname
    3. config fallback
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri is not None:
        return tracking_uri

    is_docker = os.path.exists("/.dockerenv")

    if is_docker:
        try:
            mlflow_ip = socket.gethostbyname("mlflow")
            return f"http://{mlflow_ip}:5000"
        except Exception:
            return "http://mlflow:5000"

    return cfg.get("mlflow_tracking_uri", "http://localhost:5000")


def reload_serving_model(
    *,
    model_name: str,
    cfg: dict,
) -> dict[str, Any]:
    """
    Reload the current champion model from the MLflow Model Registry.

    Returns all serving artifacts/state needed by the API.
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg))

    model, model_type, serving_alias, model_uri = load_registry_model(model_name)

    feature_schema = load_feature_schema()

    serving_model_version = None
    serving_model_run_id = None

    if serving_alias and serving_alias != "unknown":
        client = MlflowClient()
        version = client.get_model_version_by_alias(model_name, serving_alias)
        serving_model_version = str(version.version)
        serving_model_run_id = version.run_id

    logger.info(
        "Model reloaded: %s (alias=%s, version=%s)",
        model_name,
        serving_alias,
        serving_model_version,
    )

    return {
        "model": model,
        "model_type": model_type,
        "serving_alias": serving_alias,
        "model_uri": model_uri,
        "serving_model_version": serving_model_version,
        "serving_model_run_id": serving_model_run_id,
        "feature_schema": feature_schema,
        "model_name": model_name,
    }