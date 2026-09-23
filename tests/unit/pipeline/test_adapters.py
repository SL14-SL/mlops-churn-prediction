from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_churn_prediction.data.contracts import (
    DataIngestor,
    DatasetCollection,
    DatasetSplits,
    DatasetSplitter,
    FeatureBuilder,
)
from mlops_churn_prediction.training.contracts import (
    EvaluationResult,
    ModelEvaluator,
    ModelTrainer,
    TrainingResult,
)
from mlops_churn_prediction.pipeline import (
    adapters,
)


def build_datasets() -> DatasetCollection:
    return DatasetCollection(
        datasets={
            "train": pd.DataFrame(
                {
                    "customerID": [
                        "customer-1",
                        "customer-2",
                    ],
                    "Churn": [
                        "No",
                        "Yes",
                    ],
                }
            ),
        }
    )


def build_splits() -> DatasetSplits:
    return DatasetSplits(
        train=pd.DataFrame(
            {
                "feature": [
                    1,
                    2,
                ],
                "churn": [
                    0,
                    1,
                ],
            }
        ),
        validation=pd.DataFrame(
            {
                "feature": [
                    3,
                    4,
                ],
                "churn": [
                    0,
                    1,
                ],
            }
        ),
    )


def test_ingestor_satisfies_contract() -> None:
    assert isinstance(
        adapters.ChurnDataIngestor(),
        DataIngestor,
    )


def test_feature_builder_satisfies_contract() -> None:
    assert isinstance(
        adapters.ChurnFeatureBuilder(),
        FeatureBuilder,
    )


def test_splitter_satisfies_contract() -> None:
    assert isinstance(
        adapters.ChurnDatasetSplitter(),
        DatasetSplitter,
    )


def test_ingestor_delegates_to_churn_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = build_datasets()
    ingest = MagicMock(
        return_value=expected
    )
    monkeypatch.setattr(
        adapters,
        "ingest_churn_data",
        ingest,
    )

    result = adapters.ChurnDataIngestor().ingest(
        {
            "environment": "test",
        }
    )

    assert result is expected
    ingest.assert_called_once_with()


def test_feature_builder_delegates_to_feature_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = build_datasets()
    config = {
        "features": {
            "enabled_steps": [
                "clean_names",
            ],
        }
    }
    expected = pd.DataFrame(
        {
            "customerid": [
                "customer-1",
                "customer-2",
            ],
            "churn": [
                0,
                1,
            ],
        }
    )
    build = MagicMock(
        return_value=expected
    )
    monkeypatch.setattr(
        adapters,
        "build_feature_table",
        build,
    )

    result = (
        adapters.ChurnFeatureBuilder()
        .build_features(
            datasets,
            config,
        )
    )

    assert result is expected
    build.assert_called_once_with(
        datasets,
        config,
    )


def test_splitter_delegates_to_split_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
            ],
            "churn": [
                0,
                1,
                0,
                1,
            ],
        }
    )
    config = {
        "data": {
            "target_column": "churn",
        },
        "training": {
            "test_size": 0.5,
            "random_state": 42,
        },
    }
    expected = build_splits()
    split = MagicMock(
        return_value=expected
    )
    monkeypatch.setattr(
        adapters,
        "split_features",
        split,
    )

    result = adapters.ChurnDatasetSplitter().split(
        features,
        config,
    )

    assert result is expected
    split.assert_called_once_with(
        features,
        config,
    )

def test_model_trainer_satisfies_contract() -> None:
    assert isinstance(
        adapters.ChurnModelTrainer(),
        ModelTrainer,
    )


def test_model_trainer_uses_active_mlflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = build_splits()
    config = {
        "data": {
            "target_column": "churn",
        },
        "model": {
            "type": "xgboost",
            "params": {},
        },
    }
    expected = TrainingResult(
        model=object(),
        run_id="mlflow-run-123",
        metrics={
            "f1_score": 0.85,
        },
    )

    get_run_id = MagicMock(
        return_value="mlflow-run-123"
    )
    train_candidate = MagicMock(
        return_value=expected
    )

    monkeypatch.setattr(
        adapters,
        "get_active_training_run_id",
        get_run_id,
    )
    monkeypatch.setattr(
        adapters,
        "train_model_candidate",
        train_candidate,
    )

    result = adapters.ChurnModelTrainer().train(
        datasets,
        config,
    )

    assert result is expected
    get_run_id.assert_called_once_with()
    train_candidate.assert_called_once_with(
        datasets,
        config,
        run_id="mlflow-run-123",
    )

def test_model_evaluator_satisfies_contract() -> None:
    assert isinstance(
        adapters.ChurnModelEvaluator(),
        ModelEvaluator,
    )


def test_model_evaluator_delegates_candidate_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = build_splits()
    training_result = TrainingResult(
        model=object(),
        run_id="run-123",
        metrics={
            "f1_score": 0.8,
        },
    )
    expected = EvaluationResult(
        metrics={
            "f1_score": 0.8,
        },
        approved=True,
    )
    config = {
        "environment": "test",
    }

    evaluate = MagicMock(
        return_value=expected
    )
    monkeypatch.setattr(
        adapters,
        "evaluate_model_candidate",
        evaluate,
    )

    result = adapters.ChurnModelEvaluator().evaluate(
        training_result,
        datasets,
        config,
    )

    assert result is expected
    evaluate.assert_called_once_with(
        training_result,
        config,
    )