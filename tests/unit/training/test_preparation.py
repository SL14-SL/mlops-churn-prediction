import pandas as pd
import pytest

from mlops_churn_prediction.data.contracts import (
    DatasetSplits,
)
from mlops_churn_prediction.training.preparation import (
    PreparedTrainingData,
    map_churn_labels,
    prepare_training_data,
)


CONFIG = {
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
}


def build_splits() -> DatasetSplits:
    return DatasetSplits(
        train=pd.DataFrame(
            {
                "customerid": [
                    "customer-1",
                    "customer-2",
                ],
                "contract": [
                    "Monthly",
                    "Annual",
                ],
                "tenure": [
                    1,
                    24,
                ],
                "churn": [
                    "No",
                    "Yes",
                ],
            }
        ),
        validation=pd.DataFrame(
            {
                "customerid": [
                    "customer-3",
                    "customer-4",
                ],
                "contract": [
                    "Monthly",
                    "Annual",
                ],
                "tenure": [
                    3,
                    36,
                ],
                "churn": [
                    "Yes",
                    "No",
                ],
            }
        ),
    )


def test_prepare_training_data_returns_features_and_targets() -> None:
    result = prepare_training_data(
        build_splits(),
        CONFIG,
    )

    assert isinstance(
        result,
        PreparedTrainingData,
    )
    assert result.target_column == "churn"
    assert result.model_type == "xgboost"

    assert list(result.x_train.columns) == [
        "contract",
        "tenure",
    ]
    assert list(
        result.x_validation.columns
    ) == [
        "contract",
        "tenure",
    ]

    assert str(
        result.x_train["contract"].dtype
    ) == "category"
    assert result.y_train.tolist() == [
        0,
        1,
    ]
    assert result.y_validation.tolist() == [
        1,
        0,
    ]


def test_map_churn_labels_uses_binary_integers() -> None:
    labels = pd.Series(
        [
            "Yes",
            "No",
            "yes",
            "no",
        ]
    )

    result = map_churn_labels(labels)

    assert result.tolist() == [
        1,
        0,
        1,
        0,
    ]
    assert str(result.dtype) == "int64"


def test_prepare_training_data_requires_target_column() -> None:
    splits = build_splits()

    with pytest.raises(
        ValueError,
        match="Target column 'missing'",
    ):
        prepare_training_data(
            splits,
            {
                **CONFIG,
                "data": {
                    "target_column": "missing",
                },
            },
        )


def test_prepare_training_data_requires_model_config() -> None:
    with pytest.raises(
        ValueError,
        match="'model' section",
    ):
        prepare_training_data(
            build_splits(),
            {
                "data": {
                    "target_column": "churn",
                },
            },
        )