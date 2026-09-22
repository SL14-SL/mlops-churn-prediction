from dataclasses import dataclass
from typing import Any

from .releases.contracts import (
    ServingReleaseManifest,
    validate_serving_manifest,
)


@dataclass(frozen=True)
class ServingBundle:
    """Complete validated state required for churn inference."""

    release_id: str
    manifest: ServingReleaseManifest
    model: Any
    model_name: str
    model_type: str
    decision_threshold: float
    feature_schema: dict[str, Any]
    serving_alias: str
    model_uri: str
    model_version: str
    model_run_id: str


def _manifest_decision_threshold(
    manifest: ServingReleaseManifest,
) -> float:
    """Return the classification threshold stored in metadata."""
    metadata = manifest.metadata

    if not isinstance(metadata, dict):
        raise ValueError(
            "Serving manifest has no valid metadata."
        )

    threshold = metadata.get(
        "decision_threshold"
    )

    if (
        isinstance(threshold, bool)
        or not isinstance(
            threshold,
            (int, float),
        )
    ):
        raise ValueError(
            "Serving manifest has an invalid decision threshold."
        )

    normalized_threshold = float(
        threshold
    )

    if not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError(
            "Serving manifest decision threshold must be "
            "between 0 and 1."
        )

    return normalized_threshold


def validate_serving_bundle(
    bundle: ServingBundle,
) -> None:
    """Raise ValueError when a churn serving bundle is invalid."""
    if not bundle.release_id:
        raise ValueError(
            "Serving bundle has no release ID."
        )

    if not isinstance(
        bundle.manifest,
        ServingReleaseManifest,
    ):
        raise ValueError(
            "Serving bundle has no valid manifest."
        )

    validate_serving_manifest(
        bundle.manifest
    )

    if bundle.model is None:
        raise ValueError(
            "Serving bundle has no model."
        )

    required_strings = {
        "model name": bundle.model_name,
        "model type": bundle.model_type,
        "serving alias": bundle.serving_alias,
        "model URI": bundle.model_uri,
        "model version": bundle.model_version,
        "model run ID": bundle.model_run_id,
    }

    for field_name, value in required_strings.items():
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Serving bundle has no {field_name}."
            )

    if (
        isinstance(
            bundle.decision_threshold,
            bool,
        )
        or not isinstance(
            bundle.decision_threshold,
            (int, float),
        )
    ):
        raise ValueError(
            "Serving bundle has an invalid decision threshold."
        )

    if not 0.0 <= float(
        bundle.decision_threshold
    ) <= 1.0:
        raise ValueError(
            "Serving bundle decision threshold must be "
            "between 0 and 1."
        )

    if not isinstance(
        bundle.feature_schema,
        dict,
    ):
        raise ValueError(
            "Serving bundle has an invalid feature schema."
        )

    columns = bundle.feature_schema.get(
        "columns"
    )

    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "Serving bundle feature schema has no columns."
        )

    if not all(
        isinstance(column, str) and column
        for column in columns
    ):
        raise ValueError(
            "Serving bundle feature schema contains "
            "invalid columns."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Serving bundle feature schema contains "
            "duplicate columns."
        )

    dtypes = bundle.feature_schema.get(
        "dtypes",
        {},
    )

    if not isinstance(dtypes, dict):
        raise ValueError(
            "Serving bundle feature schema has invalid dtypes."
        )

    unknown_dtype_columns = (
        set(dtypes) - set(columns)
    )

    if unknown_dtype_columns:
        raise ValueError(
            "Serving bundle feature schema contains dtypes "
            "for unknown columns: "
            f"{sorted(unknown_dtype_columns)}."
        )

    model_reference = bundle.manifest.model

    expected_values = {
        "release ID": (
            bundle.manifest.release_id,
            bundle.release_id,
        ),
        "model name": (
            model_reference.name,
            bundle.model_name,
        ),
        "model version": (
            model_reference.version,
            bundle.model_version,
        ),
        "model run ID": (
            model_reference.run_id,
            bundle.model_run_id,
        ),
        "model URI": (
            model_reference.uri,
            bundle.model_uri,
        ),
        "model type": (
            model_reference.model_type,
            bundle.model_type,
        ),
    }

    for field_name, (
        manifest_value,
        bundle_value,
    ) in expected_values.items():
        if manifest_value != bundle_value:
            raise ValueError(
                f"Serving bundle {field_name} does not "
                "match manifest."
            )

    if (
        _manifest_decision_threshold(
            bundle.manifest
        )
        != float(bundle.decision_threshold)
    ):
        raise ValueError(
            "Serving bundle decision threshold does not "
            "match manifest."
        )