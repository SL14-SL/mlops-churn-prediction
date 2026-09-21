import pandas as pd
import pytest

from mlops_churn_prediction.data.contracts import (
    DatasetSplits,
)
from mlops_churn_prediction.data.splits.split import (
    split_features,
)


CONFIG = {
    "data": {
        "target_column": "Churn",
    },
    "training": {
        "test_size": 0.25,
        "random_state": 42,
    },
}


def build_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": range(8),
            "feature_b": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
            ],
            "churn": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )


def test_split_features_returns_dataset_splits() -> None:
    result = split_features(
        build_features(),
        CONFIG,
    )

    assert isinstance(
        result,
        DatasetSplits,
    )
    assert len(result.train) == 6
    assert len(result.validation) == 2
    assert result.test is None

    assert set(result.train["churn"]) == {
        0,
        1,
    }
    assert set(result.validation["churn"]) == {
        0,
        1,
    }


def test_split_features_rejects_missing_target() -> None:
    features = build_features().drop(
        columns=["churn"]
    )

    with pytest.raises(
        ValueError,
        match="Target column 'churn'",
    ):
        split_features(
            features,
            CONFIG,
        )


def test_split_features_rejects_empty_features() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        split_features(
            pd.DataFrame(),
            CONFIG,
        )


def test_split_features_rejects_invalid_config() -> None:
    with pytest.raises(
        TypeError,
        match="section 'training'",
    ):
        split_features(
            build_features(),
            {
                "data": {
                    "target_column": "churn",
                },
                "training": "invalid",
            },
        )