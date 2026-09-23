from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_churn_prediction.configs.loader import (
    get_path,
    load_config,
)
from mlops_churn_prediction.data.contracts import (
    DatasetCollection,
)
from mlops_churn_prediction.data.features.build_features import (
    build_features,
)
from mlops_churn_prediction.storage.filesystem import (
    file_exists,
)
from mlops_churn_prediction.utils.logger import (
    get_logger,
)

logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")
FEATURES_PATH = get_path("features")
VALIDATED_PATH = get_path("validated_data")

TRAIN_DATASET_NAME = "train"


def build_feature_table(
    datasets: DatasetCollection,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the churn feature table from ingested datasets."""
    training_dataset = datasets.require(
        TRAIN_DATASET_NAME
    )

    return build_features(
        training_dataset,
        config=dict(config),
    )


def run_feature_pipeline(
    config: Mapping[str, Any] | None = None,
) -> None:
    """Load validated data, build features and persist the result."""
    resolved_config = (
        TRAIN_CFG
        if config is None
        else config
    )

    logger.info(
        "Starting feature pipeline | source=%s",
        VALIDATED_PATH,
    )

    try:
        train_path = (
            f"{VALIDATED_PATH}/train.parquet"
        )

        if not file_exists(train_path):
            raise FileNotFoundError(
                "Validated data not found at "
                f"{train_path}"
            )

        training_dataset = pd.read_parquet(
            train_path
        )
        datasets = DatasetCollection(
            datasets={
                TRAIN_DATASET_NAME: (
                    training_dataset
                ),
            }
        )

        features = build_feature_table(
            datasets,
            resolved_config,
        )

        if not FEATURES_PATH.startswith(
            "gs://"
        ):
            os.makedirs(
                FEATURES_PATH,
                exist_ok=True,
            )

        output_file = (
            f"{FEATURES_PATH}/features.parquet"
        )
        features.to_parquet(
            output_file,
            index=False,
        )

        logger.info(
            "Feature engineering successful | "
            "rows=%s | columns=%s | output=%s",
            len(features),
            len(features.columns),
            output_file,
        )

    except Exception:
        logger.exception(
            "Feature pipeline failed."
        )
        raise


if __name__ == "__main__":
    run_feature_pipeline()