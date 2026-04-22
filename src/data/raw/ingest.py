import pandas as pd
import os
import shutil
import gcsfs
from sklearn.model_selection import train_test_split
from src.configs.loader import load_config, get_path
from src.data.validation.validate import validate_train
from src.utils.logger import get_logger

# Initialize project-specific logger
logger = get_logger(__name__)

def ingest(): 
    """
    Main ingestion task for Churn Prediction: 
    - Loads raw Telco data.
    - Performs a random 80/20 split 
    - Handles new batches with quarantine logic.
    """
    gcp_cfg = load_config("gcp.yaml")
    training_cfg = load_config("training.yaml")
    
    raw_path = get_path("raw_data")
    validated_path = get_path("validated_data")
    
    env = os.getenv("APP_ENV", "dev")
    logger.info(f"Starting Churn ingestion. Source: {raw_path} | Env: {env}")
    
    # 1. Load original source file
    try:
        data_file = training_cfg["data"]["feature_sources"]["train"]["path"]
        df_full = pd.read_csv(f"{raw_path}/{data_file}")
        logger.info(f"Base file {data_file} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base source file: {e}")
        return

    # 2. Initial Validation (Checks for required columns like customerID, Churn)
    validate_train(df_full)
    logger.info("Initial validation passed.")

    # 3. Random Split 
    test_size = training_cfg["training"].get("test_size", 0.2)
    train_base, sim_truth = train_test_split(
        df_full, 
        test_size=test_size, 
        random_state=42,
        stratify=df_full[training_cfg["data"]["target_column"]]
    )

    logger.info(
        f"Random split completed | train_rows={len(train_base)} | sim_rows={len(sim_truth)}"
    )
    
    # Save simulation source (Ground Truth for later Drift/Performance tests)
    sim_source_path = f"{raw_path}/simulation_ground_truth.csv"
    
    # Check if file exists (works for local and GCS)
    file_exists = False
    if raw_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        file_exists = fs.exists(sim_source_path)
    else:
        file_exists = os.path.exists(sim_source_path)

    if not file_exists:
        sim_truth.to_csv(sim_source_path, index=False)
        logger.info(f"Created simulation ground truth: {sim_source_path}")

    # 4. Collect Incremental Batches 
    new_batches_found = []
    batch_dir = f"{raw_path}/new_batches"
    quarantine_dir = f"{raw_path}/quarantine"

    # Process local or cloud batches
    if os.path.exists(batch_dir) or raw_path.startswith("gs://"):
        # Cloud Logic (GCS)
        if env == "prod" or raw_path.startswith("gs://"):
            fs = gcsfs.GCSFileSystem()
            bucket_name = gcp_cfg["gcp"]["gcs"]["bucket_name"]
            batch_pattern = f"gs://{bucket_name}/data/raw/new_batches/*.csv"
            files = fs.glob(batch_pattern)
        else:
            # Local Logic
            files = [os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.endswith(".csv")]

        for f in files:
            try:
                # No parse_dates=["Date"] anymore
                batch_df = pd.read_csv(f)
                validate_train(batch_df)
                new_batches_found.append(batch_df)
                logger.info(f"Batch '{f}' validated.")
            except Exception as e:
                logger.warning(f"Batch '{f}' rejected: {e}")
                if not raw_path.startswith("gs://"):
                    if not os.path.exists(quarantine_dir): os.makedirs(quarantine_dir)
                    shutil.move(f, os.path.join(quarantine_dir, os.path.basename(f)))

    # 5. Final Merge
    if new_batches_found:
        final_train = pd.concat([train_base] + new_batches_found, ignore_index=True)
        logger.info(f"Integrated {len(new_batches_found)} new batches.")
    else:
        final_train = train_base

    # 6. Export to Parquet
    if not validated_path.startswith("gs://"):
        os.makedirs(validated_path, exist_ok=True)

    final_train.to_parquet(f"{validated_path}/train.parquet", index=False)
    logger.info(f"Ingestion complete. Total rows: {len(final_train)} | Output: {validated_path}")

if __name__ == "__main__":
    ingest()