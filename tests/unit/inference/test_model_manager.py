from unittest.mock import MagicMock, patch

import pytest

from mlops_churn_prediction.inference.model_manager import (
    load_serving_bundle_for_release,
    reload_serving_model,
)
from mlops_churn_prediction.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_churn_prediction.inference.serving_bundle import (
    ServingBundle,
)


def build_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-7",
        created_at_utc=(
            "2026-09-22T12:00:00+00:00"
        ),
        task_type=TaskType.CLASSIFICATION,
        model=ModelReference(
            name="churn-model",
            version="7",
            run_id="run-7",
            uri="models:/churn-model/7",
            model_type="gradient_boosting",
        ),
        artifacts={
            "feature_schema": ArtifactReference(
                path="feature_schema.json",
                sha256="a" * 64,
            ),
        },
        metadata={
            "decision_threshold": 0.42,
        },
    )


def test_load_serving_bundle_for_release() -> None:
    expected_bundle = MagicMock(
        spec=ServingBundle
    )

    with (
        patch(
            "mlops_churn_prediction.inference."
            "model_manager.resolve_tracking_uri",
            return_value="http://mlflow:5000",
        ),
        patch(
            "mlops_churn_prediction.inference."
            "model_manager.load_serving_bundle",
            return_value=expected_bundle,
        ) as load_bundle,
    ):
        result = load_serving_bundle_for_release(
            release_id="release-7",
            model_name="churn-model",
            cfg={},
            models_path="models",
        )

    assert result is expected_bundle

    load_bundle.assert_called_once_with(
        release_id="release-7",
        expected_model_name="churn-model",
        models_path="models",
        tracking_uri="http://mlflow:5000",
    )


def test_reload_uses_active_release() -> None:
    manifest = build_manifest()
    expected_bundle = MagicMock(
        spec=ServingBundle
    )
    config = {
        "paths": {
            "models": "models",
        },
    }

    with (
        patch(
            "mlops_churn_prediction.inference."
            "model_manager.load_active_release_manifest",
            return_value=(
                manifest,
                "models/serving_releases/release-7",
            ),
        ) as load_active,
        patch(
            "mlops_churn_prediction.inference."
            "model_manager.load_serving_bundle_for_release",
            return_value=expected_bundle,
        ) as load_bundle,
    ):
        result = reload_serving_model(
            model_name="churn-model",
            cfg=config,
        )

    assert result is expected_bundle

    load_active.assert_called_once_with(
        models_path="models"
    )
    load_bundle.assert_called_once_with(
        release_id="release-7",
        model_name="churn-model",
        cfg=config,
        models_path="models",
    )


def test_reload_rejects_missing_models_path() -> None:
    with pytest.raises(
        ValueError,
        match="no models path",
    ):
        reload_serving_model(
            model_name="churn-model",
            cfg={
                "paths": {},
            },
        )


def test_reload_rejects_invalid_paths_section() -> None:
    with pytest.raises(
        ValueError,
        match="no valid paths section",
    ):
        reload_serving_model(
            model_name="churn-model",
            cfg={
                "paths": [],
            },
        )
