from collections.abc import Mapping
from typing import Any

from mlops_churn_prediction.deployment.prediction_probe import (
    build_prediction_probe,
)
from mlops_churn_prediction.training.contracts import (
    EvaluationResult,
    TrainingResult,
)

from .artifact_publisher import (
    ServingArtifactSource,
)
from .contracts import TaskType
from .input_provider import ServingReleaseInput
from .storage import write_json


def _require_config_path(
    config: Mapping[str, Any],
    name: str,
) -> str:
    """Return one resolved path from the project configuration."""
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError(
            "Config must contain a valid "
            "'paths' section."
        )

    value = paths.get(name)

    if (
        not isinstance(value, str)
        or not value.strip()
        or value.startswith("${")
    ):
        raise ValueError(
            f"Config path '{name}' must be "
            "a resolved non-empty string."
        )

    return value.rstrip("/")


def _require_training_string(
    training_result: TrainingResult,
    name: str,
) -> str:
    """Return a required string from training parameters."""
    value = training_result.parameters.get(
        name
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"Training result must contain '{name}'."
        )

    return value


def _require_artifact_uri(
    training_result: TrainingResult,
    name: str,
) -> str:
    """Return a required artifact URI from a training result."""
    value = training_result.artifacts.get(
        name
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            "Training result must contain "
            f"artifact '{name}'."
        )

    return value


def _optional_training_string(
    training_result: TrainingResult,
    name: str,
) -> str | None:
    """Return an optional string training parameter."""
    value = training_result.parameters.get(
        name
    )

    if value is None:
        return None

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"Training parameter '{name}' "
            "must be a non-empty string."
        )

    return value


class ChurnServingReleaseInputProvider:
    """Build classification serving inputs for a churn candidate."""

    def build_release_input(
        self,
        *,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        config: Mapping[str, Any],
    ) -> ServingReleaseInput:
        """Build immutable artifact sources for a churn release."""
        model_type = _require_training_string(
            training_result,
            "model_type",
        )
        feature_schema_uri = (
            _require_artifact_uri(
                training_result,
                "feature_schema",
            )
        )

        models_path = _require_config_path(
            config,
            "models",
        )
        validated_data_path = (
            _require_config_path(
                config,
                "validated_data",
            )
        )

        prediction_probe = (
            build_prediction_probe(
                validated_data_path=(
                    f"{validated_data_path}"
                    "/train.parquet"
                ),
            )
        )

        prediction_probe_uri = (
            f"{models_path}/training-runs/"
            f"{training_result.run_id}/"
            "prediction_probe.json"
        )

        write_json(
            prediction_probe_uri,
            prediction_probe,
        )

        decision_threshold = (
            training_result.parameters.get(
                "decision_threshold"
            )
        )

        if (
            isinstance(decision_threshold, bool)
            or not isinstance(
                decision_threshold,
                (int, float),
            )
        ):
            raise ValueError(
                "Training result must contain "
                "a numeric decision threshold."
            )

        return ServingReleaseInput(
            task_type=TaskType.CLASSIFICATION,
            model_type=model_type,
            sources={
                "feature_schema": (
                    ServingArtifactSource(
                        source_uri=(
                            feature_schema_uri
                        ),
                        relative_path=(
                            "feature_schema.json"
                        ),
                    )
                ),
                "prediction_probe": (
                    ServingArtifactSource(
                        source_uri=(
                            prediction_probe_uri
                        ),
                        relative_path=(
                            "prediction_probe.json"
                        ),
                    )
                ),
            },
            metadata={
                "decision_threshold": float(
                    decision_threshold
                ),
                "evaluation_approved": (
                    evaluation_result.approved
                ),
                "evaluation_reasons": list(
                    evaluation_result.reasons
                ),
            },
            dataset_version=(
                _optional_training_string(
                    training_result,
                    "dataset_version",
                )
            ),
            config_hash=(
                _optional_training_string(
                    training_result,
                    "config_hash",
                )
            ),
            git_commit=(
                _optional_training_string(
                    training_result,
                    "git_commit",
                )
            ),
        )