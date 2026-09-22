from dataclasses import replace

import pytest

from mlops_churn_prediction.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_churn_prediction.inference.serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)


def build_manifest() -> ServingReleaseManifest:
    checksum = "a" * 64

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
                sha256=checksum,
            ),
            "prediction_probe": ArtifactReference(
                path="prediction_probe.json",
                sha256=checksum,
            ),
        },
        dataset_version="dataset-1",
        config_hash="config-hash",
        git_commit="abc123",
        metadata={
            "decision_threshold": 0.42,
        },
    )


def build_bundle() -> ServingBundle:
    manifest = build_manifest()

    return ServingBundle(
        release_id=manifest.release_id,
        manifest=manifest,
        model=object(),
        model_name=manifest.model.name,
        model_type=manifest.model.model_type,
        decision_threshold=0.42,
        feature_schema={
            "columns": [
                "tenure",
                "monthlycharges",
            ],
            "dtypes": {
                "tenure": "float64",
                "monthlycharges": "float64",
            },
        },
        serving_alias="champion",
        model_uri=manifest.model.uri,
        model_version=manifest.model.version,
        model_run_id=manifest.model.run_id,
    )


def test_valid_serving_bundle_passes_validation() -> None:
    validate_serving_bundle(
        build_bundle()
    )


def test_rejects_model_name_mismatch() -> None:
    bundle = replace(
        build_bundle(),
        model_name="wrong-model",
    )

    with pytest.raises(
        ValueError,
        match="model name does not match manifest",
    ):
        validate_serving_bundle(bundle)


def test_rejects_threshold_mismatch() -> None:
    bundle = replace(
        build_bundle(),
        decision_threshold=0.75,
    )

    with pytest.raises(
        ValueError,
        match=(
            "decision threshold does not match manifest"
        ),
    ):
        validate_serving_bundle(bundle)


def test_rejects_invalid_feature_schema() -> None:
    bundle = replace(
        build_bundle(),
        feature_schema={
            "columns": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="feature schema has no columns",
    ):
        validate_serving_bundle(bundle)


def test_rejects_missing_threshold_metadata() -> None:
    bundle = build_bundle()
    manifest = replace(
        bundle.manifest,
        metadata={},
    )
    invalid_bundle = replace(
        bundle,
        manifest=manifest,
    )

    with pytest.raises(
        ValueError,
        match="invalid decision threshold",
    ):
        validate_serving_bundle(
            invalid_bundle
        )