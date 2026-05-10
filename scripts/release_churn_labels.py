from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import uuid4

import fsspec
import pandas as pd

from src.configs.loader import ensure_dir, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = get_path("raw_data")
MONITORING_PATH = get_path("monitoring")

PENDING_LABEL_DIR = f"{MONITORING_PATH}/pending_labels"
GROUND_TRUTH_BATCH_DIR = f"{MONITORING_PATH}/ground_truth_batches"
TRAINING_BATCH_DIR = f"{RAW_DATA_PATH}/new_batches"
RELEASED_LABEL_DIR = f"{MONITORING_PATH}/released_labels"


RAW_TELCO_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def normalize_gcs_path(path: str) -> str:
    if path.startswith("gs://"):
        return path
    return f"gs://{path}"


def list_files(directory: str, pattern: str) -> list[str]:
    glob_pattern = f"{directory}/{pattern}"
    fs, fs_pattern = fsspec.core.url_to_fs(glob_pattern)
    files = fs.glob(fs_pattern)

    if is_gcs_path(directory):
        return sorted(normalize_gcs_path(path) for path in files)

    return sorted(str(path) for path in files)


def archive_file(src: str, dst: str) -> None:
    fs_src, src_path = fsspec.core.url_to_fs(src)
    fs_dst, dst_path = fsspec.core.url_to_fs(dst)

    if fs_src.protocol == fs_dst.protocol:
        fs_src.copy(src_path, dst_path)
        fs_src.rm(src_path)
        return

    with fs_src.open(src_path, "rb") as fsrc:
        with fs_dst.open(dst_path, "wb") as fdst:
            fdst.write(fsrc.read())

    fs_src.rm(src_path)


def normalize_customer_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize customer id column names to the raw Telco schema."""
    df = df.copy()

    if "customerid" in df.columns and "customerID" not in df.columns:
        df = df.rename(columns={"customerid": "customerID"})

    return df


def list_releasable_pending_files(simulation_day: int) -> list[str]:
    """Return pending label files whose labels are available by this simulation day."""
    files = []

    for path in list_files(PENDING_LABEL_DIR, "pending_churn_day_*.csv"):
        df = pd.read_csv(path, nrows=1)

        if "label_available_day" not in df.columns:
            logger.warning("Skipping pending file without label_available_day: %s", path)
            continue

        available_day = int(df["label_available_day"].iloc[0])

        if available_day <= simulation_day:
            files.append(path)

    return files


def release_pending_file(path: str, simulation_day: int) -> tuple[str, str]:
    """Release one pending label file into monitoring and training batch locations."""
    df = pd.read_csv(path)
    df = normalize_customer_id_column(df)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    batch_id = str(uuid4())

    ensure_dir(GROUND_TRUTH_BATCH_DIR)
    ensure_dir(TRAINING_BATCH_DIR)
    ensure_dir(RELEASED_LABEL_DIR)

    monitoring_df = df.copy()

    if "Churn" in monitoring_df.columns and "churn" not in monitoring_df.columns:
        monitoring_df["churn"] = (
            monitoring_df["Churn"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )

    monitoring_df["label_available_at"] = now.isoformat()
    monitoring_df["label_batch_id"] = batch_id
    monitoring_df["released_simulation_day"] = simulation_day

    monitoring_file = (
        f"{GROUND_TRUTH_BATCH_DIR}/"
        f"ground_truth_churn_day_{simulation_day:03d}_{timestamp}.csv"
    )
    monitoring_df.to_csv(monitoring_file, index=False)

    missing_cols = [col for col in RAW_TELCO_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Pending file is missing raw training columns: {missing_cols}")

    training_df = df[RAW_TELCO_COLUMNS].copy()

    if "TotalCharges" in training_df.columns:
        training_df["TotalCharges"] = training_df["TotalCharges"].astype(str)

    training_file = (
        f"{TRAINING_BATCH_DIR}/"
        f"train_batch_churn_day_{simulation_day:03d}_{timestamp}.csv"
    )
    training_df.to_csv(training_file, index=False)

    released_file = f"{RELEASED_LABEL_DIR}/{PurePosixPath(path).name}"
    archive_file(path, released_file)

    logger.info("✅ Released monitoring labels: %s", monitoring_file)
    logger.info("✅ Released training batch: %s", training_file)
    logger.info("📦 Archived pending file: %s", released_file)

    return monitoring_file, training_file


def main(simulation_day: int) -> None:
    """Release all labels available for the given simulation day."""
    files = list_releasable_pending_files(simulation_day)

    if not files:
        logger.info("No labels available for release on simulation day %s.", simulation_day)
        return

    logger.info("Releasing %s pending label batch(es).", len(files))

    for path in files:
        release_pending_file(path, simulation_day)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-day", type=int, required=True)
    args = parser.parse_args()

    main(simulation_day=args.simulation_day)