from unittest.mock import MagicMock

from mlops_churn_prediction.training import (
    model_factory,
)


def test_log_model_returns_mlflow_model_uri(
    monkeypatch,
) -> None:
    logged_model = MagicMock()
    logged_model.model_uri = "models:/m-123"

    log_model = MagicMock(
        return_value=logged_model
    )
    monkeypatch.setattr(
        model_factory.mlflow.pyfunc,
        "log_model",
        log_model,
    )

    model = MagicMock()

    model_uri = model_factory.log_model_by_type(
        model,
        "gradient_boosting",
        artifact_path="trained/model",
    )

    assert model_uri == "models:/m-123"

    call = log_model.call_args

    assert (
        call.kwargs["artifact_path"]
        == "trained/model"
    )
    assert (
        call.kwargs["python_model"].model
        is model
    )