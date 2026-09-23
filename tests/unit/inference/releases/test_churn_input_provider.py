from unittest.mock import MagicMock

import pytest

from mlops_churn_prediction.inference.releases import (
    churn_input_provider,
)
from mlops_churn_prediction.inference.releases.contracts import (
    TaskType,
)
from mlops_churn_prediction.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


def build_training_result() -> TrainingResult:
    return TrainingResult(
        model=MagicMock(),
        run_id="run-123",
        metrics={
            "roc_auc": 0.85,
        },
        parameters={
            "model_type": "gradient_boosting",
            "decision_threshold": 0.42,
            "dataset_version": "dataset-v1",
            "config_hash": "config-hash",
            "git_commit": "abc123",
        },
        artifacts={
            "feature_schema": (
                "models/training-runs/run-123/"
                "feature_schema.json"
            ),
        },
    )


def build_evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        metrics={
            "roc_auc": 0.85,
        },
        approved=True,
    )


def build_config() -> dict:
    return {
        "paths": {
            "models": "models",
            "validated_data": (
                "data/validation"
            ),
        },
    }


def test_builds_churn_release_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_probe = {
        "inputs": [
            {
                "customerID": "customer-1",
            }
        ],
    }

    build_probe = MagicMock(
        return_value=prediction_probe
    )
    write_json = MagicMock()

    monkeypatch.setattr(
        churn_input_provider,
        "build_prediction_probe",
        build_probe,
    )
    monkeypatch.setattr(
        churn_input_provider,
        "write_json",
        write_json,
    )

    release_input = (
        churn_input_provider
        .ChurnServingReleaseInputProvider()
        .build_release_input(
            training_result=(
                build_training_result()
            ),
            evaluation_result=(
                build_evaluation_result()
            ),
            config=build_config(),
        )
    )

    assert (
        release_input.task_type
        is TaskType.CLASSIFICATION
    )
    assert (
        release_input.model_type
        == "gradient_boosting"
    )
    assert release_input.dataset_version == (
        "dataset-v1"
    )
    assert release_input.config_hash == (
        "config-hash"
    )
    assert release_input.git_commit == "abc123"

    assert (
        release_input.metadata[
            "decision_threshold"
        ]
        == 0.42
    )

    feature_schema_source = (
        release_input.sources[
            "feature_schema"
        ]
    )
    assert feature_schema_source.source_uri == (
        "models/training-runs/run-123/"
        "feature_schema.json"
    )
    assert (
        feature_schema_source.relative_path
        == "feature_schema.json"
    )

    probe_source = release_input.sources[
        "prediction_probe"
    ]
    assert probe_source.source_uri == (
        "models/training-runs/run-123/"
        "prediction_probe.json"
    )

    build_probe.assert_called_once_with(
        validated_data_path=(
            "data/validation/train.parquet"
        ),
    )
    write_json.assert_called_once_with(
        (
            "models/training-runs/run-123/"
            "prediction_probe.json"
        ),
        prediction_probe,
    )


def test_requires_feature_schema() -> None:
    training_result = (
        build_training_result()
    )
    training_result = TrainingResult(
        model=training_result.model,
        run_id=training_result.run_id,
        metrics=training_result.metrics,
        parameters=(
            training_result.parameters
        ),
        artifacts={},
    )

    with pytest.raises(
        ValueError,
        match="artifact 'feature_schema'",
    ):
        (
            churn_input_provider
            .ChurnServingReleaseInputProvider()
            .build_release_input(
                training_result=training_result,
                evaluation_result=(
                    build_evaluation_result()
                ),
                config=build_config(),
            )
        )


def test_requires_numeric_decision_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_result = (
        build_training_result()
    )
    parameters = dict(
        training_result.parameters
    )
    parameters["decision_threshold"] = (
        "invalid"
    )

    invalid_result = TrainingResult(
        model=training_result.model,
        run_id=training_result.run_id,
        metrics=training_result.metrics,
        parameters=parameters,
        artifacts=training_result.artifacts,
    )

    monkeypatch.setattr(
        churn_input_provider,
        "build_prediction_probe",
        MagicMock(return_value={"inputs": [{}]}),
    )
    monkeypatch.setattr(
        churn_input_provider,
        "write_json",
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="numeric decision threshold",
    ):
        (
            churn_input_provider
            .ChurnServingReleaseInputProvider()
            .build_release_input(
                training_result=invalid_result,
                evaluation_result=(
                    build_evaluation_result()
                ),
                config=build_config(),
            )
        )