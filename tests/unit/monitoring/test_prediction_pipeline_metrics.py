from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mlops_churn_prediction.api import services


def build_arguments() -> dict:
    return {
        "payload": SimpleNamespace(
            inputs=[
                {
                    "customerID": "customer-1",
                },
                {
                    "customerID": "customer-2",
                },
            ],
        ),
        "model": MagicMock(),
        "model_type": "gradient_boosting",
        "feature_schema": {},
        "train_cfg": {},
        "dq_reference_categories": {},
        "decision_threshold": 0.23,
    }


def test_pipeline_observes_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = {
        "results": [
            {
                "churn_probability": 0.1,
            },
            {
                "churn_probability": 0.8,
            },
        ],
    }
    internal_pipeline = MagicMock(
        return_value=output
    )
    prediction_observer = MagicMock()
    output_observer = MagicMock()

    monkeypatch.setattr(
        services,
        "_run_prediction_pipeline",
        internal_pipeline,
    )
    monkeypatch.setattr(
        services,
        "observe_prediction",
        prediction_observer,
    )
    monkeypatch.setattr(
        services,
        "observe_model_outputs",
        output_observer,
    )

    result = services.run_prediction_pipeline(
        **build_arguments()
    )

    assert result is output

    observer_arguments = (
        prediction_observer.call_args.kwargs
    )

    assert observer_arguments[
        "task_type"
    ] == "classification"
    assert observer_arguments[
        "status"
    ] == "success"
    assert observer_arguments[
        "observation_count"
    ] == 2
    assert observer_arguments[
        "latency_seconds"
    ] >= 0.0

    output_observer.assert_called_once_with(
        output["results"],
        decision_threshold=0.23,
    )


def test_pipeline_observes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internal_pipeline = MagicMock(
        side_effect=RuntimeError(
            "inference failed"
        )
    )
    prediction_observer = MagicMock()
    output_observer = MagicMock()

    monkeypatch.setattr(
        services,
        "_run_prediction_pipeline",
        internal_pipeline,
    )
    monkeypatch.setattr(
        services,
        "observe_prediction",
        prediction_observer,
    )
    monkeypatch.setattr(
        services,
        "observe_model_outputs",
        output_observer,
    )

    with pytest.raises(
        RuntimeError,
        match="inference failed",
    ):
        services.run_prediction_pipeline(
            **build_arguments()
        )

    observer_arguments = (
        prediction_observer.call_args.kwargs
    )

    assert observer_arguments[
        "status"
    ] == "error"
    assert observer_arguments[
        "observation_count"
    ] == 2
    assert observer_arguments[
        "latency_seconds"
    ] >= 0.0

    output_observer.assert_not_called()