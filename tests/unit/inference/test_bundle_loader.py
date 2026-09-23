from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from mlops_churn_prediction.inference.bundle_loader import (
    _resolve_artifact_uri,
    load_serving_bundle,
)
from mlops_churn_prediction.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
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
            "prediction_probe": ArtifactReference(
                path="prediction_probe.json",
                sha256="b" * 64,
            ),
        },
        metadata={
            "decision_threshold": 0.42,
        },
    )


def test_loads_complete_serving_bundle() -> None:
    manifest = build_manifest()
    model = MagicMock()
    feature_schema = {
        "columns": [
            "tenure",
            "monthlycharges",
        ],
        "dtypes": {
            "tenure": "float64",
            "monthlycharges": "float64",
        },
    }

    with (
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader.mlflow.set_tracking_uri",
        ) as set_tracking_uri,
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader.load_release_manifest",
            return_value=(
                manifest,
                "models/serving_releases/release-7",
            ),
        ),
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader._resolve_artifact_uri",
            return_value=(
                "models/serving_releases/"
                "release-7/feature_schema.json"
            ),
        ) as resolve_artifact,
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader.load_json",
            return_value=feature_schema,
        ),
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader.load_model_by_type",
            return_value=model,
        ) as load_model,
    ):
        bundle = load_serving_bundle(
            release_id="release-7",
            expected_model_name="churn-model",
            models_path="models",
            tracking_uri="http://mlflow:5000",
        )

    set_tracking_uri.assert_called_once_with(
        "http://mlflow:5000"
    )
    resolve_artifact.assert_called_once_with(
        release_root=(
            "models/serving_releases/release-7"
        ),
        reference=manifest.artifacts[
            "feature_schema"
        ],
    )
    load_model.assert_called_once_with(
        "models:/churn-model/7",
        "gradient_boosting",
    )

    assert bundle.release_id == "release-7"
    assert bundle.model is model
    assert bundle.model_name == "churn-model"
    assert bundle.model_version == "7"
    assert bundle.model_run_id == "run-7"
    assert bundle.decision_threshold == 0.42
    assert bundle.feature_schema == feature_schema


def test_rejects_wrong_model_name() -> None:
    manifest = build_manifest()

    with patch(
        "mlops_churn_prediction.inference."
        "bundle_loader.load_release_manifest",
        return_value=(
            manifest,
            "models/serving_releases/release-7",
        ),
    ):
        with pytest.raises(
            ValueError,
            match="does not match configuration",
        ):
            load_serving_bundle(
                release_id="release-7",
                expected_model_name="wrong-model",
                models_path="models",
                tracking_uri="http://mlflow:5000",
            )


def test_rejects_forecasting_release() -> None:
    manifest = replace(
        build_manifest(),
        task_type=TaskType.FORECASTING,
    )

    with patch(
        "mlops_churn_prediction.inference."
        "bundle_loader.load_release_manifest",
        return_value=(
            manifest,
            "models/serving_releases/release-7",
        ),
    ):
        with pytest.raises(
            ValueError,
            match="not a classification release",
        ):
            load_serving_bundle(
                release_id="release-7",
                expected_model_name="churn-model",
                models_path="models",
                tracking_uri="http://mlflow:5000",
            )


def test_rejects_missing_feature_schema() -> None:
    manifest = replace(
        build_manifest(),
        artifacts={
            "prediction_probe": ArtifactReference(
                path="prediction_probe.json",
                sha256="b" * 64,
            ),
        },
    )

    with patch(
        "mlops_churn_prediction.inference."
        "bundle_loader.load_release_manifest",
        return_value=(
            manifest,
            "models/serving_releases/release-7",
        ),
    ):
        with pytest.raises(
            ValueError,
            match="no feature_schema artifact",
        ):
            load_serving_bundle(
                release_id="release-7",
                expected_model_name="churn-model",
                models_path="models",
                tracking_uri="http://mlflow:5000",
            )


def test_rejects_artifact_checksum_mismatch() -> None:
    reference = ArtifactReference(
        path="feature_schema.json",
        sha256="a" * 64,
    )

    with (
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader.file_exists",
            return_value=True,
        ),
        patch(
            "mlops_churn_prediction.inference."
            "bundle_loader.sha256_uri",
            return_value="b" * 64,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="checksum mismatch",
        ):
            _resolve_artifact_uri(
                release_root=(
                    "models/serving_releases/release-7"
                ),
                reference=reference,
            )