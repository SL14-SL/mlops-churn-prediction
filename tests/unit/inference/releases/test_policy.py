from dataclasses import replace

import pytest

from mlops_churn_prediction.inference.releases.contracts import (
    ArtifactReference,
    ModelReference,
    ServingReleaseManifest,
    TaskType,
)
from mlops_churn_prediction.inference.releases.policy import (
    validate_task_manifest,
)

VALID_CHECKSUM = "d" * 64


def artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        path=path,
        sha256=VALID_CHECKSUM,
    )



def build_valid_manifest() -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="classification-release",
        created_at_utc="2026-09-14T08:00:00+00:00",
        task_type=TaskType.CLASSIFICATION,
        model=ModelReference(
            name="classification-model",
            version="1",
            run_id="classification-run",
            uri="models:/classification-model/1",
            model_type="xgboost",
        ),
        artifacts={
            "feature_schema": artifact("feature_schema.json"),
        },
        metadata={
            "decision_threshold": 0.42,
        },
    )


def test_validate_task_manifest() -> None:
    validate_task_manifest(build_valid_manifest())


def test_manifest_requires_feature_schema() -> None:
    manifest = replace(
        build_valid_manifest(),
        artifacts={
            "prediction_probe": artifact(
                "prediction_probe.json"
            )
        },
    )

    with pytest.raises(
        ValueError,
        match="missing required artifacts",
    ):
        validate_task_manifest(manifest)


def test_manifest_requires_decision_threshold() -> None:
    manifest = replace(
        build_valid_manifest(),
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="requires a numeric decision_threshold",
    ):
        validate_task_manifest(manifest)


@pytest.mark.parametrize(
    "decision_threshold",
    [-0.1, 1.1, "0.5", True],
)
def test_manifest_rejects_invalid_threshold(
    decision_threshold: object,
) -> None:
    manifest = replace(
        build_valid_manifest(),
        metadata={
            "decision_threshold": decision_threshold,
        },
    )

    with pytest.raises(ValueError):
        validate_task_manifest(manifest)


def test_manifest_rejects_forecasting_task() -> None:
    manifest = replace(
        build_valid_manifest(),
        task_type=TaskType.FORECASTING,
    )

    with pytest.raises(
        ValueError,
        match="Expected a classification",
    ):
        validate_task_manifest(manifest)
