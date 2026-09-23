from unittest.mock import MagicMock

import pandas as pd
import pytest

from mlops_churn_prediction.data.contracts import (
    DatasetCollection,
)
from mlops_churn_prediction.data.features import (
    pipeline,
)


def test_build_feature_table_uses_training_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_dataset = pd.DataFrame(
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
    )
    datasets = DatasetCollection(
        datasets={
            "train": training_dataset,
        }
    )
    config = {
        "features": {
            "enabled_steps": [
                "clean_names",
            ],
        }
    }

    expected_features = pd.DataFrame(
        {
            "customerid": [
                "customer-1",
                "customer-2",
            ],
            "churn": [
                "No",
                "Yes",
            ],
        }
    )
    feature_builder = MagicMock(
        return_value=expected_features
    )
    monkeypatch.setattr(
        pipeline,
        "build_features",
        feature_builder,
    )

    result = pipeline.build_feature_table(
        datasets,
        config,
    )

    assert result is expected_features
    feature_builder.assert_called_once_with(
        training_dataset,
        config=config,
    )


def test_build_feature_table_requires_training_dataset() -> None:
    datasets = DatasetCollection(
        datasets={
            "reference": pd.DataFrame(
                {
                    "customerID": [
                        "customer-1",
                    ],
                }
            ),
        }
    )

    with pytest.raises(
        KeyError,
        match="Required dataset is missing: train",
    ):
        pipeline.build_feature_table(
            datasets,
            {},
        )