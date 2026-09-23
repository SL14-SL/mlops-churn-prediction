from typing import Any

import mlflow

from ..configs.paths import join_uri
from ..storage.filesystem import file_exists
from .model_loader import load_model_by_type
from .releases.contracts import (
    ArtifactReference,
    TaskType,
)
from .releases.repository import (
    load_release_manifest,
)
from .releases.storage import (
    load_json,
    sha256_uri,
)
from .serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)


def _resolve_artifact_uri(
    *,
    release_root: str,
    reference: ArtifactReference,
) -> str:
    """Resolve and checksum-validate a release artifact."""
    artifact_uri = join_uri(
        release_root,
        reference.path,
    )

    if not file_exists(artifact_uri):
        raise FileNotFoundError(
            f"Serving artifact not found: {artifact_uri}"
        )

    actual_checksum = sha256_uri(
        artifact_uri
    )

    if actual_checksum != reference.sha256:
        raise ValueError(
            "Serving artifact checksum mismatch: "
            f"{reference.path}"
        )

    return artifact_uri


def _decision_threshold(
    metadata: dict[str, Any] | None,
) -> float:
    """Read the classification decision threshold."""
    if not isinstance(metadata, dict):
        raise ValueError(
            "Serving manifest has no valid metadata."
        )

    value = metadata.get(
        "decision_threshold"
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            "Serving manifest has an invalid "
            "decision threshold."
        )

    threshold = float(value)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Serving manifest decision threshold must be "
            "between 0 and 1."
        )

    return threshold


def load_serving_bundle(
    *,
    release_id: str,
    expected_model_name: str,
    models_path: str,
    tracking_uri: str,
) -> ServingBundle:
    """Load and validate one classification serving release."""
    mlflow.set_tracking_uri(
        tracking_uri
    )

    manifest, release_root = (
        load_release_manifest(
            models_path=models_path,
            release_id=release_id,
        )
    )

    if manifest.task_type is not TaskType.CLASSIFICATION:
        raise ValueError(
            "Serving release is not a classification release."
        )

    if manifest.model.name != expected_model_name:
        raise ValueError(
            "Serving manifest model name does not match "
            "configuration: "
            f"{manifest.model.name} != "
            f"{expected_model_name}"
        )

    try:
        feature_schema_reference = (
            manifest.artifacts[
                "feature_schema"
            ]
        )
    except KeyError as exc:
        raise ValueError(
            "Serving manifest has no feature_schema artifact."
        ) from exc

    feature_schema_uri = (
        _resolve_artifact_uri(
            release_root=release_root,
            reference=feature_schema_reference,
        )
    )

    feature_schema = load_json(
        feature_schema_uri
    )

    model = load_model_by_type(
        manifest.model.uri,
        manifest.model.model_type,
    )

    bundle = ServingBundle(
        release_id=manifest.release_id,
        manifest=manifest,
        model=model,
        model_name=manifest.model.name,
        model_type=manifest.model.model_type,
        decision_threshold=_decision_threshold(
            manifest.metadata
        ),
        feature_schema=feature_schema,
        serving_alias="champion",
        model_uri=manifest.model.uri,
        model_version=manifest.model.version,
        model_run_id=manifest.model.run_id,
    )

    validate_serving_bundle(bundle)
    return bundle