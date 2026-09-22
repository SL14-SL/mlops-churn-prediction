from .contracts import ServingReleaseManifest, TaskType, validate_serving_manifest

_REQUIRED_ARTIFACTS = {
    "feature_schema",
}


def validate_task_manifest(
    manifest: ServingReleaseManifest,
) -> None:
    """Validate classification-specific release requirements."""
    validate_serving_manifest(manifest)

    if manifest.task_type is not TaskType.CLASSIFICATION:
        raise ValueError(
            "Expected a classification serving manifest."
        )

    missing_artifacts = (
        _REQUIRED_ARTIFACTS - manifest.artifacts.keys()
    )

    if missing_artifacts:
        raise ValueError(
            "Serving manifest is missing required artifacts: "
            f"{sorted(missing_artifacts)}."
        )

    metadata = manifest.metadata or {}
    decision_threshold = metadata.get("decision_threshold")

    if (
        isinstance(decision_threshold, bool)
        or not isinstance(decision_threshold, (int, float))
    ):
        raise ValueError(
            "Classification serving manifest requires a numeric "
            "decision_threshold."
        )

    if not 0.0 <= float(decision_threshold) <= 1.0:
        raise ValueError(
            "Classification decision_threshold must be "
            "between 0 and 1."
        )
