import os
import socket

from .bundle_loader import load_serving_bundle
from .releases.lifecycle_repository import (
    load_active_release_manifest,
)
from .serving_bundle import ServingBundle


def resolve_tracking_uri(
    cfg: dict,
) -> str:
    """Resolve the MLflow tracking URI."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:
        return tracking_uri

    if os.path.exists("/.dockerenv"):
        try:
            mlflow_ip = socket.gethostbyname(
                "mlflow"
            )
            return f"http://{mlflow_ip}:5000"
        except OSError:
            return "http://mlflow:5000"

    tracking = cfg.get(
        "tracking",
        {},
    )

    if isinstance(tracking, dict):
        configured_uri = tracking.get(
            "mlflow_tracking_uri"
        )

        if configured_uri:
            return str(configured_uri)

    return str(
        cfg.get(
            "mlflow_tracking_uri",
            "http://localhost:5000",
        )
    )


def load_serving_bundle_for_release(
    *,
    release_id: str,
    model_name: str,
    cfg: dict,
    models_path: str,
) -> ServingBundle:
    """Load one concrete serving release."""
    return load_serving_bundle(
        release_id=release_id,
        expected_model_name=model_name,
        models_path=models_path,
        tracking_uri=resolve_tracking_uri(
            cfg
        ),
    )


def reload_serving_model(
    *,
    model_name: str,
    cfg: dict,
) -> ServingBundle:
    """Load the release referenced by the active pointer."""
    paths = cfg.get(
        "paths",
        {},
    )

    if not isinstance(paths, dict):
        raise ValueError(
            "Configuration has no valid paths section."
        )

    models_path = paths.get(
        "models"
    )

    if not models_path:
        raise ValueError(
            "Configuration has no models path."
        )

    manifest, _ = (
        load_active_release_manifest(
            models_path=str(models_path)
        )
    )

    return load_serving_bundle_for_release(
        release_id=manifest.release_id,
        model_name=model_name,
        cfg=cfg,
        models_path=str(models_path),
    )