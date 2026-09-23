import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

FEATURE_DRIFT_SCORE = Gauge(
    "mlops_feature_drift_score",
    "Latest drift score reported for one monitored feature.",
    ["feature"],
)

FEATURE_DRIFT_DETECTED = Gauge(
    "mlops_feature_drift_detected",
    "Whether the latest feature drift score exceeds its threshold.",
    ["feature"],
)

CLASSIFICATION_PREDICTIONS = Counter(
    "mlops_classification_predictions_total",
    "Total number of classification outputs.",
    ["predicted_class"],
)

CLASSIFICATION_PROBABILITY = Histogram(
    "mlops_classification_probability",
    "Distribution of positive-class probabilities.",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)

CLASSIFICATION_FEEDBACK = Counter(
    "mlops_classification_feedback_total",
    "Total number of classification predictions with ground truth.",
)

CLASSIFICATION_CORRECT = Counter(
    "mlops_classification_correct_total",
    "Total number of correct classification predictions.",
)

CLASSIFICATION_BRIER_ERROR = Counter(
    "mlops_classification_brier_error_total",
    "Accumulated squared probability error.",
)


@dataclass(frozen=True, slots=True)
class ClassificationFeedback:
    """Ground truth matched to one classification prediction."""

    request_id: str
    row_index: int
    probability: float
    predicted_class: int
    actual_class: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError(
                "request_id must not be empty"
            )

        if self.row_index < 0:
            raise ValueError(
                "row_index must not be negative"
            )

        if (
            not math.isfinite(self.probability)
            or not 0.0
            <= self.probability
            <= 1.0
        ):
            raise ValueError(
                "probability must be between zero and one"
            )

        if self.predicted_class not in {
            0,
            1,
        }:
            raise ValueError(
                "predicted_class must be zero or one"
            )

        if self.actual_class not in {
            0,
            1,
        }:
            raise ValueError(
                "actual_class must be zero or one"
            )


def observe_model_outputs(
    results: Iterable[
        Mapping[str, Any]
    ],
    *,
    decision_threshold: float,
) -> None:
    """Record bounded churn output distributions."""
    if (
        not math.isfinite(
            decision_threshold
        )
        or not 0.0
        <= decision_threshold
        <= 1.0
    ):
        raise ValueError(
            "decision_threshold must be "
            "between zero and one"
        )

    for result in results:
        probability = float(
            result["churn_probability"]
        )

        if (
            not math.isfinite(probability)
            or not 0.0
            <= probability
            <= 1.0
        ):
            raise ValueError(
                "churn_probability must be "
                "between zero and one"
            )

        predicted_class = int(
            probability
            >= decision_threshold
        )

        CLASSIFICATION_PREDICTIONS.labels(
            predicted_class=str(
                predicted_class
            ),
        ).inc()

        CLASSIFICATION_PROBABILITY.observe(
            probability
        )


def observe_model_feedback(
    feedback: ClassificationFeedback,
) -> None:
    """Record performance after a delayed label arrives."""
    CLASSIFICATION_FEEDBACK.inc()

    if (
        feedback.predicted_class
        == feedback.actual_class
    ):
        CLASSIFICATION_CORRECT.inc()

    squared_error = (
        feedback.probability
        - feedback.actual_class
    ) ** 2

    CLASSIFICATION_BRIER_ERROR.inc(
        squared_error
    )


def observe_feature_drift(
    *,
    scores: Mapping[str, float],
    threshold: float,
) -> None:
    """Publish project-computed feature drift scores."""
    if (
        not math.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError(
            "Drift threshold must be a finite, "
            "non-negative number."
        )

    for feature, score in scores.items():
        if not feature:
            raise ValueError(
                "Drift feature name must not be empty."
            )

        if (
            not math.isfinite(score)
            or score < 0
        ):
            raise ValueError(
                "Drift scores must be finite, "
                "non-negative numbers."
            )

        FEATURE_DRIFT_SCORE.labels(
            feature=feature,
        ).set(score)

        FEATURE_DRIFT_DETECTED.labels(
            feature=feature,
        ).set(
            float(
                score >= threshold
            )
        )