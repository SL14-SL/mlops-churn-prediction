from unittest.mock import MagicMock

import pytest

from mlops_churn_prediction.pipeline import (
    adapters,
)
from mlops_churn_prediction.training.contracts import (
    TrainingResult,
)


def build_training_result(
    *,
    model_type: str = "gradient_boosting",
) -> TrainingResult:
    return TrainingResult(
        model=MagicMock(),
        run_id="run-123",
        metrics={
            "roc_auc": 0.85,
        },
        parameters={
            "model_type": model_type,
        },
    )


def test_churn_model_logger_returns_model_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_model = MagicMock(
        return_value="models:/m-123"
    )
    monkeypatch.setattr(
        adapters,
        "log_model_by_type",
        log_model,
    )

    training_result = build_training_result()

    model_uri = (
        adapters.ChurnModelArtifactLogger().log_model(
            training_result,
            artifact_path="trained/model",
            config={},
        )
    )

    assert model_uri == "models:/m-123"

    log_model.assert_called_once_with(
        training_result.model,
        "gradient_boosting",
        artifact_path="trained/model",
        metadata={
            "training_run_id": "run-123",
        },
    )


def test_churn_model_logger_requires_model_type() -> None:
    training_result = TrainingResult(
        model=MagicMock(),
        run_id="run-123",
        metrics={},
        parameters={},
    )

    with pytest.raises(
        ValueError,
        match="must contain a model type",
    ):
        adapters.ChurnModelArtifactLogger().log_model(
            training_result,
            artifact_path="model",
            config={},
        )