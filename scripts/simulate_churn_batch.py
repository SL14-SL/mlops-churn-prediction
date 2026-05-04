from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests

from src.configs.loader import get_path, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()

RAW_DATA_PATH = Path(get_path("raw_data"))
PREDICTIONS_PATH = Path(get_path("predictions"))

SIMULATION_FILE = RAW_DATA_PATH / "simulation_ground_truth.csv"
BATCH_DIR = RAW_DATA_PATH / "new_batches"
INFERENCE_LOG_FILE = PREDICTIONS_PATH / "inference_log.parquet"

API_URL = CFG["api"]["url"]
API_KEY = os.getenv("API_KEY")


def normalize_customer_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize customer id column names to `customerid`.

    The raw Telco dataset uses `customerID`, while feature engineering and
    prediction logs may use `customerid`. Normalizing avoids fragile joins.
    """
    df = df.copy()

    if "customerID" in df.columns and "customerid" not in df.columns:
        df = df.rename(columns={"customerID": "customerid"})

    return df


def get_next_batch(df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
    """
    Select the next holdout batch for simulation.

    This implementation takes the first rows from the holdout file. For a more
    advanced simulation, this can be extended to consume batches incrementally.
    """
    return df.head(batch_size).copy()


def call_prediction_api(batch: pd.DataFrame, request_id: str) -> None:
    """
    Send an unlabeled customer batch to the prediction API.

    The API writes prediction metadata to `inference_log.parquet`. This function
    only triggers prediction generation; prediction IDs are read back from the
    inference log afterwards.
    """
    if not API_KEY:
        raise RuntimeError("API_KEY environment variable is not set.")

    payload = {
        "inputs": batch.to_dict(orient="records"),
        "context": {
            "request_id": request_id,
        },
    }

    response = requests.post(
        API_URL,
        json=payload,
        headers={"X-API-KEY": API_KEY},
        timeout=60,
    )

    if response.status_code >= 400:
        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

    response.raise_for_status()


def load_logged_predictions(request_id: str, expected_rows: int) -> pd.DataFrame:
    """
    Load predictions created by the API call from the inference log.

    Args:
        request_id: Unique request id used for the API call.
        expected_rows: Number of rows sent to the API.

    Returns:
        Logged prediction rows for the request.

    Raises:
        FileNotFoundError: If the inference log was not created.
        ValueError: If no matching predictions are found.
    """
    if not INFERENCE_LOG_FILE.exists():
        raise FileNotFoundError(f"Inference log not found: {INFERENCE_LOG_FILE}")

    log_df = pd.read_parquet(INFERENCE_LOG_FILE)

    if "request_id" not in log_df.columns:
        raise KeyError("Inference log must contain `request_id`.")

    latest_preds = log_df[log_df["request_id"] == request_id].copy()

    if latest_preds.empty:
        raise ValueError(f"No predictions found in inference log for request_id={request_id}")

    if len(latest_preds) != expected_rows:
        logger.warning(
            "Expected %s logged predictions, found %s for request_id=%s",
            expected_rows,
            len(latest_preds),
            request_id,
        )

    latest_preds = normalize_customer_id_column(latest_preds)

    return latest_preds


def simulate_labeled_batch(batch_size: int = 50) -> Path:
    """
    Generate a labeled churn batch from the holdout simulation dataset.

    Flow:
    1. Load `simulation_ground_truth.csv`, which was held out during ingestion.
    2. Split labels from features.
    3. Send unlabeled features to the prediction API.
    4. Read the API-generated predictions from the inference log.
    5. Join logged predictions with the real holdout labels.
    6. Persist a labeled ground-truth batch for performance monitoring.

    Returns:
        Path to the generated ground-truth batch CSV file.
    """
    if not SIMULATION_FILE.exists():
        raise FileNotFoundError(f"Simulation file not found: {SIMULATION_FILE}")

    df = pd.read_csv(SIMULATION_FILE)

    if df.empty:
        raise ValueError("Simulation ground truth file is empty.")

    if "Churn" not in df.columns:
        raise KeyError("Simulation file must contain `Churn` column.")

    if "customerID" not in df.columns and "customerid" not in df.columns:
        raise KeyError("Simulation file must contain `customerID` or `customerid`.")

    batch = get_next_batch(df, batch_size=batch_size)

    ground_truth = batch.copy()
    ground_truth = normalize_customer_id_column(ground_truth)
    ground_truth = ground_truth[["customerid", "Churn"]].copy()
    ground_truth = ground_truth.rename(columns={"Churn": "churn"})

    ground_truth["churn"] = (
        ground_truth["churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )

    if ground_truth["churn"].isna().any():
        raise ValueError("Ground truth contains invalid churn labels.")

    features = batch.drop(columns=["Churn"])
    request_id = f"simulate-churn-{uuid4()}"

    logger.info("📡 Sending batch of size %s to prediction API...", len(features))

    call_prediction_api(features, request_id=request_id)

    latest_preds = load_logged_predictions(
        request_id=request_id,
        expected_rows=len(features),
    )

    merged = latest_preds.merge(
        ground_truth,
        on="customerid",
        how="left",
    )

    if merged["churn"].isna().any():
        missing = int(merged["churn"].isna().sum())
        raise ValueError(f"Failed to join ground truth for {missing} prediction rows.")

    now = datetime.now(timezone.utc)

    merged["label_available_at"] = now.isoformat()
    merged["label_batch_id"] = str(uuid4())

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_file = BATCH_DIR / f"ground_truth_churn_{timestamp}.csv"

    merged.to_csv(output_file, index=False)

    logger.info("✅ Labeled batch saved: %s", output_file)
    logger.info("Rows: %s", len(merged))
    logger.info("Churn rate: %.2f%%", merged["churn"].mean() * 100)

    print(f"Ground truth batch written: {output_file}")
    print(f"Rows: {len(merged)}")
    print(f"Churn rate: {merged['churn'].mean():.2%}")

    return output_file


if __name__ == "__main__":
    simulate_labeled_batch()