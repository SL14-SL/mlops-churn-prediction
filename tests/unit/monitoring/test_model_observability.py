from unittest.mock import MagicMock

import pytest

from mlops_churn_prediction.monitoring import (
    model_observability,
)
from mlops_churn_prediction.monitoring.model_observability import (
    ClassificationFeedback,
)


def test_observes_churn_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = MagicMock()
    probabilities = MagicMock()

    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_PREDICTIONS",
        predictions,
    )
    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_PROBABILITY",
        probabilities,
    )

    model_observability.observe_model_outputs(
        [
            {
                "churn_probability": 0.25,
            },
            {
                "churn_probability": 0.85,
            },
        ],
        decision_threshold=0.5,
    )

    predictions.labels.assert_any_call(
        predicted_class="0"
    )
    predictions.labels.assert_any_call(
        predicted_class="1"
    )
    probabilities.observe.assert_any_call(
        0.25
    )
    probabilities.observe.assert_any_call(
        0.85
    )


def test_output_observation_uses_active_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = MagicMock()

    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_PREDICTIONS",
        predictions,
    )
    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_PROBABILITY",
        MagicMock(),
    )

    model_observability.observe_model_outputs(
        [
            {
                "churn_probability": 0.3,
            },
        ],
        decision_threshold=0.23,
    )

    predictions.labels.assert_called_once_with(
        predicted_class="1"
    )


def test_observes_classification_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_total = MagicMock()
    correct_total = MagicMock()
    brier_error = MagicMock()

    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_FEEDBACK",
        feedback_total,
    )
    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_CORRECT",
        correct_total,
    )
    monkeypatch.setattr(
        model_observability,
        "CLASSIFICATION_BRIER_ERROR",
        brier_error,
    )

    model_observability.observe_model_feedback(
        ClassificationFeedback(
            request_id="request-1",
            row_index=0,
            probability=0.8,
            predicted_class=1,
            actual_class=1,
        )
    )

    feedback_total.inc.assert_called_once_with()
    correct_total.inc.assert_called_once_with()
    brier_error.inc.assert_called_once_with(
        pytest.approx(0.04)
    )


def test_rejects_invalid_probability() -> None:
    with pytest.raises(
        ValueError,
        match="churn_probability",
    ):
        model_observability.observe_model_outputs(
            [
                {
                    "churn_probability": 1.2,
                },
            ],
            decision_threshold=0.5,
        )