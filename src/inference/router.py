import os

from mlflow.tracking import MlflowClient

from src.inference.model_loader import load_model_by_type
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_model_alias() -> str:
    """
    Returns which registry alias should be served.

    Default:
        champion

    Optional override:
        MODEL_ALIAS=challenger
    """
    alias = os.getenv("MODEL_ALIAS", "champion").strip().lower()

    if alias not in {"champion", "challenger"}:
        logger.warning(f"Unknown MODEL_ALIAS='{alias}', defaulting to 'champion'")
        return "champion"

    return alias


def resolve_model_uri(model_name: str, alias: str) -> str:
    return f"models:/{model_name}@{alias}"


def load_registry_model(model_name: str):
    """
    Loads model + metadata from MLflow Registry by alias.

    Returns:
        model, model_type, target_transformation, alias, model_uri
    """
    alias = resolve_model_alias()
    model_uri = resolve_model_uri(model_name, alias)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(mv.run_id)
    decision_threshold = float(
        run.data.params.get("decision_threshold", 0.5)
    )

    model_type = (
        run.data.tags.get("model_type")
        or run.data.params.get("model_type")
        or "xgboost"
    )

    model = load_model_by_type(model_uri, model_type)

    logger.info(
        f"Loaded registry model: alias={alias} | "
        f"model_uri={model_uri} | model_type={model_type} | "
    )

    return model, model_type, alias, model_uri, decision_threshold