import os
import shutil

import gcsfs
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.loader import get_path, load_config
from src.data.validation.validate import validate_train
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_raw_churn_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw Telco churn data before validation, merging, and parquet export.

    CSV inference can read `TotalCharges` as either string/object or float depending
    on the specific batch contents. Keeping it as string at ingestion time avoids
    mixed parquet dtypes after concatenating base data with incremental batches.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str)

    if "customerid" in df.columns and "customerID" not in df.columns:
        df = df.rename(columns={"customerid": "customerID"})

    return df


def ingest() -> None:
    """
    Run the churn ingestion step.

    The task loads the base Telco dataset, creates or reuses a simulation holdout,
    validates incremental training batches, quarantines invalid local batches,
    and writes the final validated training set to parquet.
    """
    gcp_cfg = load_config("gcp.yaml")
    training_cfg = load_config("training.yaml")

    raw_path = get_path("raw_data")
    validated_path = get_path("validated_data")

    env = os.getenv("APP_ENV", "dev")
    logger.info("Starting Churn ingestion. Source: %s | Env: %s", raw_path, env)

    try:
        data_file = training_cfg["data"]["feature_sources"]["train"]["path"]
        df_full = pd.read_csv(f"{raw_path}/{data_file}")
        df_full = normalize_raw_churn_schema(df_full)
        logger.info("Base file %s loaded successfully.", data_file)
    except Exception as e:
        logger.error("Failed to load base source file: %s", e)
        return

    validate_train(df_full)
    logger.info("Initial validation passed.")

    test_size = training_cfg["training"].get("test_size", 0.2)
    target_column = training_cfg["data"]["target_column"]

    train_base, sim_truth = train_test_split(
        df_full,
        test_size=test_size,
        random_state=42,
        stratify=df_full[target_column],
    )

    logger.info(
        "Random split completed | train_rows=%s | sim_rows=%s",
        len(train_base),
        len(sim_truth),
    )

    sim_source_path = f"{raw_path}/simulation_ground_truth.csv"

    if raw_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        simulation_file_exists = fs.exists(sim_source_path)
    else:
        simulation_file_exists = os.path.exists(sim_source_path)

    if not simulation_file_exists:
        sim_truth.to_csv(sim_source_path, index=False)
        logger.info("Created simulation ground truth: %s", sim_source_path)

    new_batches_found = []
    batch_dir = f"{raw_path}/new_batches"
    quarantine_dir = f"{raw_path}/quarantine"

    if os.path.exists(batch_dir) or raw_path.startswith("gs://"):
        if env == "prod" or raw_path.startswith("gs://"):
            fs = gcsfs.GCSFileSystem()
            bucket_name = gcp_cfg["gcp"]["gcs"]["bucket_name"]
            batch_pattern = f"gs://{bucket_name}/data/raw/new_batches/*.csv"
            files = fs.glob(batch_pattern)
        else:
            files = [
                os.path.join(batch_dir, f)
                for f in os.listdir(batch_dir)
                if f.endswith(".csv")
            ]

        for file_path in files:
            try:
                batch_df = pd.read_csv(file_path)
                batch_df = normalize_raw_churn_schema(batch_df)

                validate_train(batch_df)
                new_batches_found.append(batch_df)

                logger.info("Batch '%s' validated.", file_path)
            except Exception as e:
                logger.warning("Batch '%s' rejected: %s", file_path, e)

                if not raw_path.startswith("gs://"):
                    os.makedirs(quarantine_dir, exist_ok=True)
                    shutil.move(
                        file_path,
                        os.path.join(quarantine_dir, os.path.basename(file_path)),
                    )

    if new_batches_found:
        final_train = pd.concat([train_base] + new_batches_found, ignore_index=True)
        logger.info("Integrated %s new batches.", len(new_batches_found))
    else:
        final_train = train_base.copy()

    final_train = normalize_raw_churn_schema(final_train)

    if not validated_path.startswith("gs://"):
        os.makedirs(validated_path, exist_ok=True)

    final_train.to_parquet(f"{validated_path}/train.parquet", index=False)
    logger.info(
        "Ingestion complete. Total rows: %s | Output: %s",
        len(final_train),
        validated_path,
    )


if __name__ == "__main__":
    ingest()