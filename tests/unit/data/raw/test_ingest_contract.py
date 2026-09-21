from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_churn_prediction.data.contracts import (
    DatasetCollection,
)
from mlops_churn_prediction.data.raw import (
    ingest as ingest_module,
)


def test_ingest_returns_validated_dataset_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_dataset = pd.DataFrame(
        {
            "customerID": [
                "customer-1",
                "customer-2",
                "customer-3",
                "customer-4",
            ],
            "Churn": [
                "No",
                "Yes",
                "No",
                "Yes",
            ],
        }
    )
    train_base = full_dataset.iloc[:2].copy()
    simulation_truth = full_dataset.iloc[2:].copy()

    incremental_batch = pd.DataFrame(
        {
            "customerID": [
                "customer-5",
            ],
            "Churn": [
                "No",
            ],
        }
    )
    final_train = pd.concat(
        [
            train_base,
            incremental_batch,
        ],
        ignore_index=True,
    )

    training_config = {
        "training": {
            "test_size": 0.2,
        },
        "data": {
            "target_column": "Churn",
        },
    }

    load_config = MagicMock(
        return_value=training_config
    )
    monkeypatch.setattr(
        ingest_module,
        "load_config",
        load_config,
    )

    configured_paths = {
        "raw_data": "data/raw",
        "validated_data": "data/validation",
    }
    get_path = MagicMock(
        side_effect=configured_paths.__getitem__
    )
    monkeypatch.setattr(
        ingest_module,
        "get_path",
        get_path,
    )

    load_base_dataset = MagicMock(
        return_value=full_dataset
    )
    monkeypatch.setattr(
        ingest_module,
        "load_base_dataset",
        load_base_dataset,
    )

    create_simulation_split = MagicMock(
        return_value=(
            train_base,
            simulation_truth,
        )
    )
    monkeypatch.setattr(
        ingest_module,
        "create_simulation_split",
        create_simulation_split,
    )

    persist_simulation_source = MagicMock()
    monkeypatch.setattr(
        ingest_module,
        "persist_simulation_source_if_missing",
        persist_simulation_source,
    )

    collect_batches = MagicMock(
        return_value=[
            incremental_batch,
        ]
    )
    monkeypatch.setattr(
        ingest_module,
        "collect_incremental_batches",
        collect_batches,
    )

    merge_batches = MagicMock(
        return_value=final_train
    )
    monkeypatch.setattr(
        ingest_module,
        "merge_training_batches",
        merge_batches,
    )

    persist_dataset = MagicMock()
    monkeypatch.setattr(
        ingest_module,
        "persist_validated_dataset",
        persist_dataset,
    )

    monkeypatch.setenv(
        "APP_ENV",
        "test",
    )

    result = ingest_module.ingest()

    assert isinstance(
        result,
        DatasetCollection,
    )
    assert (
        result.require("train")
        is final_train
    )
    assert result.metadata == {
        "source": "data/raw",
        "validated_path": "data/validation",
        "environment": "test",
        "incremental_batch_count": 1,
    }

    load_config.assert_called_once_with(
        "training.yaml"
    )
    load_base_dataset.assert_called_once_with(
        raw_path="data/raw",
        training_config=training_config,
    )
    persist_simulation_source.assert_called_once_with(
        simulation_truth,
        raw_path="data/raw",
    )
    collect_batches.assert_called_once_with(
        raw_path="data/raw",
    )
    merge_batches.assert_called_once_with(
        train_base,
        [
            incremental_batch,
        ],
    )
    persist_dataset.assert_called_once_with(
        final_train,
        validated_path="data/validation",
    )