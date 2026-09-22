from mlops_churn_prediction.training.candidate_evaluation import (
    evaluate_model_candidate,
)
from mlops_churn_prediction.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


def build_training_result(
    *,
    metrics: dict[str, float] | None = None,
) -> TrainingResult:
    return TrainingResult(
        model=object(),
        run_id="run-123",
        metrics=(
            metrics
            if metrics is not None
            else {
                "accuracy": 0.82,
                "precision": 0.79,
                "recall": 0.76,
                "f1_score": 0.77,
                "roc_auc": 0.88,
                "brier_score": 0.14,
                "decision_threshold": 0.43,
                "training_duration_seconds": 1.2,
            }
        ),
    )


def test_candidate_with_valid_metrics_is_approved() -> None:
    result = evaluate_model_candidate(
        build_training_result(),
        {},
    )

    assert isinstance(
        result,
        EvaluationResult,
    )
    assert result.approved is True
    assert result.reasons == ()
    assert result.metrics["f1_score"] == 0.77


def test_candidate_with_missing_metrics_is_rejected() -> None:
    result = evaluate_model_candidate(
        build_training_result(
            metrics={
                "f1_score": 0.77,
                "roc_auc": 0.88,
            }
        ),
        {},
    )

    assert result.approved is False
    assert len(result.reasons) == 1
    assert "missing required metrics" in (
        result.reasons[0]
    )
    assert "accuracy" in result.reasons[0]


def test_candidate_with_out_of_range_metric_is_rejected() -> None:
    result = evaluate_model_candidate(
        build_training_result(
            metrics={
                "accuracy": 0.82,
                "precision": 0.79,
                "recall": 0.76,
                "f1_score": 1.2,
                "roc_auc": 0.88,
                "brier_score": 0.14,
            }
        ),
        {},
    )

    assert result.approved is False
    assert len(result.reasons) == 1
    assert "outside the range" in (
        result.reasons[0]
    )
    assert "f1_score" in result.reasons[0]