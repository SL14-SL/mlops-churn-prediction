from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mlops_churn_prediction.data.contracts import (
    DatasetSplits,
)
from mlops_churn_prediction.training import (
    candidate,
)
from mlops_churn_prediction.training.contracts import (
    TrainingResult,
)


CONFIG = {
    "random_seed": 42,
    "data": {
        "target_column": "Churn",
    },
    "model": {
        "type": "xgboost",
        "params": {
            "max_depth": 4,
        },
    },
    "features": {
        "drop_columns": [
            "customerid",
        ],
    },
    "paths": {
        "models": "models",
    },
}


def build_splits() -> DatasetSplits:
    return DatasetSplits(
        train=pd.DataFrame(
            {
                "customerid": [
                    "customer-1",
                    "customer-2",
                    "customer-3",
                    "customer-4",
                ],
                "tenure": [
                    1,
                    12,
                    24,
                    36,
                ],
                "churn": [
                    "No",
                    "Yes",
                    "No",
                    "Yes",
                ],
            }
        ),
        validation=pd.DataFrame(
            {
                "customerid": [
                    "customer-5",
                    "customer-6",
                    "customer-7",
                    "customer-8",
                ],
                "tenure": [
                    3,
                    18,
                    30,
                    48,
                ],
                "churn": [
                    "No",
                    "Yes",
                    "No",
                    "Yes",
                ],
            }
        ),
    )


def test_train_model_candidate_returns_training_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MagicMock()
    model.predict_proba.return_value = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )

    build_model = MagicMock(
        return_value=model
    )
    fit_model = MagicMock()

    save_feature_schema = MagicMock(
        return_value={
            "columns": ["tenure"],
            "dtypes": {
                "tenure": "int64",
            },
        }
    )

    monkeypatch.setattr(
        candidate,
        "save_feature_schema",
        save_feature_schema,
    )

    monkeypatch.setattr(
        candidate,
        "build_model",
        build_model,
    )
    monkeypatch.setattr(
        candidate,
        "fit_model",
        fit_model,
    )

    result = candidate.train_model_candidate(
        build_splits(),
        CONFIG,
        run_id="run-123",
    )

    assert isinstance(
        result,
        TrainingResult,
    )
    assert result.model is model
    assert result.run_id == "run-123"

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["f1_score"] == 1.0
    assert result.metrics["roc_auc"] == 1.0
    assert (
        result.metrics[
            "training_duration_seconds"
        ]
        >= 0.0
    )

    assert result.parameters["model_type"] == (
        "xgboost"
    )
    assert result.parameters["max_depth"] == 4

    build_model.assert_called_once_with(
        CONFIG["model"],
        seed=42,
    )
    assert result.artifacts == {
        "feature_schema": (
            "models/training-runs/run-123/"
            "feature_schema.json"
        ),
    }

    schema_call = (
        save_feature_schema.call_args
    )
    assert list(
        schema_call.args[0].columns
    ) == ["tenure"]
    assert schema_call.args[1] == (
        "models/training-runs/run-123/"
        "feature_schema.json"
    )
    fit_model.assert_called_once()

    fit_call = fit_model.call_args
    assert fit_call.args[0] is model
    assert fit_call.args[1] == "xgboost"


def test_train_model_candidate_does_not_start_mlflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MagicMock()
    model.predict_proba.return_value = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )

    monkeypatch.setattr(
        candidate,
        "build_model",
        MagicMock(return_value=model),
    )
    monkeypatch.setattr(
        candidate,
        "fit_model",
        MagicMock(),
    )
    monkeypatch.setattr(
        candidate,
        "save_feature_schema",
        MagicMock(),
    )

    result = candidate.train_model_candidate(
        build_splits(),
        CONFIG,
        run_id="run-456",
    )

    assert result.run_id == "run-456"


def test_train_model_candidate_requires_run_id() -> None:
    with pytest.raises(
        ValueError,
        match="requires a run ID",
    ):
        candidate.train_model_candidate(
            build_splits(),
            CONFIG,
            run_id="",
        )


def test_find_best_threshold_rejects_unknown_metric() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported threshold metric",
    ):
        candidate.find_best_threshold(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            metric="unsupported",
        )