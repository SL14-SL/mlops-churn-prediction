from collections.abc import Mapping
import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops_churn_prediction.data.contracts import (
    DatasetSplits,
)
from mlops_churn_prediction.training.contracts import (
    TrainingResult,
)
from mlops_churn_prediction.training.feature_schema import (
    save_feature_schema,
)
from mlops_churn_prediction.training.model_factory import (
    build_model,
    fit_model,
)
from mlops_churn_prediction.training.preparation import (
    prepare_training_data,
)


def find_best_threshold(
    y_true: Any,
    y_proba: Any,
    *,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the best classification threshold."""
    if thresholds is None:
        thresholds = np.arange(
            0.10,
            0.91,
            0.01,
        )

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        predictions = (
            y_proba >= threshold
        ).astype(int)

        if metric == "f1":
            score = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        else:
            raise ValueError(
                "Unsupported threshold metric: "
                f"{metric}"
            )

        if score > best_score:
            best_score = float(score)
            best_threshold = float(
                threshold
            )

    return best_threshold, best_score


def _feature_schema_path(
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> str:
    """Return the run-specific feature-schema path."""
    paths = config.get("paths")

    if not isinstance(paths, Mapping):
        raise ValueError(
            "Training config must contain "
            "a valid 'paths' section."
        )

    models_path = paths.get("models")

    if (
        not isinstance(models_path, str)
        or not models_path.strip()
        or models_path.startswith("${")
    ):
        raise ValueError(
            "Config path 'models' must be "
            "a resolved non-empty string."
        )

    return (
        f"{models_path.rstrip('/')}"
        f"/training-runs/{run_id}"
        "/feature_schema.json"
    )


def train_model_candidate(
    datasets: DatasetSplits,
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> TrainingResult:
    """Fit and evaluate one churn model candidate."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(
            "Model candidate training requires a run ID."
        )

    prepared = prepare_training_data(
        datasets,
        config,
    )
    model_config = dict(
        prepared.model_config
    )

    configured_seed = config.get(
        "random_seed"
    )
    seed = (
        int(configured_seed)
        if configured_seed is not None
        else None
    )

    model = build_model(
        model_config,
        seed=seed,
    )

    started_at = time.perf_counter()

    fit_model(
        model,
        prepared.model_type,
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
    )

    duration_seconds = (
        time.perf_counter()
        - started_at
    )

    probabilities = model.predict_proba(
        prepared.x_validation
    )[:, 1]

    decision_threshold, threshold_score = (
        find_best_threshold(
            prepared.y_validation,
            probabilities,
            metric="f1",
        )
    )
    predictions = (
        probabilities
        >= decision_threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                prepared.y_validation,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                prepared.y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                prepared.y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                prepared.y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                prepared.y_validation,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                prepared.y_validation,
                probabilities,
            )
        ),
        "decision_threshold": (
            decision_threshold
        ),
        "threshold_score": threshold_score,
        "training_duration_seconds": float(
            duration_seconds
        ),
    }

    configured_parameters = (
        model_config.get(
            "params",
            {},
        )
    )

    if not isinstance(
        configured_parameters,
        Mapping,
    ):
        raise TypeError(
            "Model parameters must be a mapping."
        )

    parameters = {
        "model_type": prepared.model_type,
        "decision_threshold": (
            decision_threshold
        ),
        **dict(configured_parameters),
    }

    feature_schema_path = _feature_schema_path(
        config,
        run_id=run_id,
    )

    save_feature_schema(
        prepared.x_train,
        feature_schema_path,
    )

    return TrainingResult(
        model=model,
        run_id=run_id,
        metrics=metrics,
        parameters=parameters,
        artifacts={
            "feature_schema": (
                feature_schema_path
            ),
        },
    )