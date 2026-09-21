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