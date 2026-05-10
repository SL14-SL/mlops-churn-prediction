from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.loader import ensure_dir, file_exists, get_path, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAIN_CFG = load_config("training.yaml")
FEATURES = get_path("features")
SPLITS = get_path("splits")


def split() -> None:
    """
    Create stratified train and validation splits.

    The configured paths can point to local storage or GCS, so paths are kept
    as strings and directory creation is handled by the project helper.
    """
    input_file = f"{FEATURES}/features.parquet"

    if not file_exists(input_file):
        logger.error(f"Feature file not found: {input_file}")
        return

    logger.info(f"Loading features for splitting from: {input_file}")
    df = pd.read_parquet(input_file)

    target_column = TRAIN_CFG.get("data", {}).get("target_column", "churn")
    target_column = target_column.lower().replace(" ", "_")

    test_size = TRAIN_CFG.get("training", {}).get("test_size", 0.2)
    random_state = TRAIN_CFG.get("training", {}).get("random_state", 42)

    if target_column not in df.columns:
        logger.error(
            f"Target column '{target_column}' not found in features. "
            f"Available: {df.columns.tolist()}"
        )
        return

    logger.info(
        f"Performing stratified split "
        f"(test_size={test_size}, random_state={random_state})"
    )

    train, val = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    ensure_dir(SPLITS)

    train.to_parquet(f"{SPLITS}/train.parquet", index=False)
    val.to_parquet(f"{SPLITS}/val.parquet", index=False)

    logger.info(f"Data split complete. Train rows: {len(train)} | Val rows: {len(val)}")
    logger.info(
        f"Class distribution (Train):\n"
        f"{train[target_column].value_counts(normalize=True)}"
    )
    logger.info(
        f"Class distribution (Val):\n"
        f"{val[target_column].value_counts(normalize=True)}"
    )


if __name__ == "__main__":
    split()