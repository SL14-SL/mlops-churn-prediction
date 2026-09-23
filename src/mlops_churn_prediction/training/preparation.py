from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlops_churn_prediction.data.contracts import (
    DatasetSplits,
)


@dataclass(frozen=True)
class PreparedTrainingData:
    """Features, targets and settings prepared for model fitting."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    target_column: str
    model_type: str
    model_config: Mapping[str, Any]


def normalize_feature_dtypes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert object feature columns to categorical columns."""
    normalized = dataframe.copy()
    object_columns = normalized.select_dtypes(
        include=["object"]
    ).columns

    for column in object_columns:
        normalized[column] = normalized[
            column
        ].astype("category")

    return normalized


def map_churn_labels(
    labels: pd.Series,
) -> pd.Series:
    """Map Yes/No churn labels to binary integer targets."""
    return (
        labels.astype(str)
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
            }
        )
        .fillna(0)
        .astype(int)
    )


def prepare_training_data(
    datasets: DatasetSplits,
    config: Mapping[str, Any],
) -> PreparedTrainingData:
    """Prepare churn features and targets for model fitting."""
    data_config = config.get("data")
    model_config = config.get("model")
    feature_config = config.get(
        "features",
        {},
    )

    if not isinstance(data_config, Mapping):
        raise ValueError(
            "Training config must contain a valid "
            "'data' section."
        )

    if not isinstance(model_config, Mapping):
        raise ValueError(
            "Training config must contain a valid "
            "'model' section."
        )

    if not isinstance(feature_config, Mapping):
        raise ValueError(
            "Training config section 'features' "
            "must be a mapping."
        )

    configured_target = data_config.get(
        "target_column"
    )

    if (
        not isinstance(configured_target, str)
        or not configured_target.strip()
    ):
        raise ValueError(
            "Training target column must be configured."
        )

    target_column = (
        configured_target.lower()
        .replace(" ", "_")
    )

    for name, dataframe in (
        ("train", datasets.train),
        ("validation", datasets.validation),
    ):
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                f"is missing from {name} dataset."
            )

    configured_drop_columns = (
        feature_config.get(
            "drop_columns",
            [],
        )
    )

    if not isinstance(
        configured_drop_columns,
        (list, tuple),
    ):
        raise TypeError(
            "Feature drop columns must be a list "
            "or tuple."
        )

    drop_columns = [
        target_column,
        *(
            str(column)
            for column
            in configured_drop_columns
        ),
    ]

    x_train = normalize_feature_dtypes(
        datasets.train.drop(
            columns=drop_columns,
            errors="ignore",
        )
    )
    x_validation = normalize_feature_dtypes(
        datasets.validation.drop(
            columns=drop_columns,
            errors="ignore",
        )
    )

    return PreparedTrainingData(
        x_train=x_train,
        y_train=map_churn_labels(
            datasets.train[target_column]
        ),
        x_validation=x_validation,
        y_validation=map_churn_labels(
            datasets.validation[
                target_column
            ]
        ),
        target_column=target_column,
        model_type=str(
            model_config.get(
                "type",
                "xgboost",
            )
        ),
        model_config=model_config,
    )