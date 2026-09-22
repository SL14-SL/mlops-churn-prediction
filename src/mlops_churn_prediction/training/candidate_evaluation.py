from collections.abc import Mapping
from typing import Any

from mlops_churn_prediction.training.contracts import (
    EvaluationResult,
    TrainingResult,
)


_BOUNDED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "brier_score",
)


def evaluate_model_candidate(
    training_result: TrainingResult,
    config: Mapping[str, Any],
) -> EvaluationResult:
    """Validate the technical metrics of a trained candidate."""
    del config

    metrics = {
        name: float(value)
        for name, value
        in training_result.metrics.items()
    }
    reasons: list[str] = []

    missing_metrics = [
        name
        for name in _BOUNDED_METRICS
        if name not in metrics
    ]

    if missing_metrics:
        reasons.append(
            "Candidate is missing required metrics: "
            f"{missing_metrics}."
        )

    invalid_metrics = [
        name
        for name in _BOUNDED_METRICS
        if (
            name in metrics
            and not 0.0 <= metrics[name] <= 1.0
        )
    ]

    if invalid_metrics:
        reasons.append(
            "Candidate contains metrics outside "
            "the range [0, 1]: "
            f"{invalid_metrics}."
        )

    return EvaluationResult(
        metrics=metrics,
        approved=not reasons,
        reasons=tuple(reasons),
    )