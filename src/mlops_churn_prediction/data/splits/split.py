from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_churn_prediction.configs.loader import (
    get_path,
    load_config,
)
from mlops_churn_prediction.data.contracts import (
    DatasetSplits,
)
from mlops_churn_prediction.storage.filesystem import (
    ensure_dir,
    file_exists,
)
from mlops_churn_prediction.utils.logger import (
    get_logger,
)

logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")
FEATURES = get_path("features")
SPLITS = get_path("splits")


def split_features(
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> DatasetSplits:
    """Create stratified train and validation datasets."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "Features must be provided as a pandas DataFrame."
        )

    if features.empty:
        raise ValueError(
            "Features must not be empty."
        )

    data_config = config.get("data", {})
    training_config = config.get("training", {})

    if not isinstance(data_config, Mapping):
        raise TypeError(
            "Config section 'data' must be a mapping."
        )

    if not isinstance(training_config, Mapping):
        raise TypeError(
            "Config section 'training' must be a mapping."
        )

    configured_target = data_config.get(
        "target_column",
        "churn",
    )
    target_column = (
        str(configured_target)
        .lower()
        .replace(" ", "_")
    )

    if target_column not in features.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in features."
        )

    test_size = float(
        training_config.get(
            "test_size",
            0.2,
        )
    )
    random_state = int(
        training_config.get(
            "random_state",
            42,
        )
    )

    train, validation = train_test_split(
        features,
        test_size=test_size,
        random_state=random_state,
        stratify=features[target_column],
    )

    return DatasetSplits(
        train=train,
        validation=validation,
    )


def split() -> None:
    """Create and persist configured train and validation splits."""
    input_file = f"{FEATURES}/features.parquet"

    if not file_exists(input_file):
        logger.error(
            "Feature file not found: %s",
            input_file,
        )
        return

    logger.info(
        "Loading features for splitting from: %s",
        input_file,
    )

    features = pd.read_parquet(input_file)
    splits = split_features(
        features,
        TRAIN_CFG,
    )

    ensure_dir(SPLITS)

    train_file = f"{SPLITS}/train.parquet"
    validation_file = f"{SPLITS}/val.parquet"

    splits.train.to_parquet(
        train_file,
        index=False,
    )
    splits.validation.to_parquet(
        validation_file,
        index=False,
    )

    target_column = str(
        TRAIN_CFG.get(
            "data",
            {},
        ).get(
            "target_column",
            "churn",
        )
    ).lower().replace(" ", "_")

    logger.info(
        "Data split complete | train_rows=%s | "
        "validation_rows=%s",
        len(splits.train),
        len(splits.validation),
    )
    logger.info(
        "Class distribution (train):\n%s",
        splits.train[target_column].value_counts(
            normalize=True
        ),
    )
    logger.info(
        "Class distribution (validation):\n%s",
        splits.validation[target_column].value_counts(
            normalize=True
        ),
    )


if __name__ == "__main__":
    split()