from __future__ import annotations

import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests

from src.configs.loader import ensure_dir, file_exists, get_path, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()
RAW_DATA_PATH = get_path("raw_data")
PREDICTIONS_PATH = get_path("predictions")
MONITORING_PATH = get_path("monitoring")

SIMULATION_FILE = f"{RAW_DATA_PATH}/simulation_ground_truth.csv"
INFERENCE_LOG_FILE = f"{PREDICTIONS_PATH}/inference_log.parquet"
PENDING_LABEL_DIR = f"{MONITORING_PATH}/pending_labels"

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
    if not file_exists(INFERENCE_LOG_FILE):
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


def simulate_labeled_batch(
    batch_size: int = 50,
    simulation_day: int = 1,
    label_delay_days: int = 1,
) -> Path:
    """
    Score a holdout batch and store its labels as pending.

    The batch is removed from the simulation pool immediately, but the true
    labels are only released later by `release_churn_labels.py`.
    """
    if not file_exists(SIMULATION_FILE):
        raise FileNotFoundError(f"Simulation file not found: {SIMULATION_FILE}")        

    df = pd.read_csv(SIMULATION_FILE)

    if df.empty:
        raise ValueError("Simulation ground truth file is empty.")

    if "Churn" not in df.columns:
        raise KeyError("Simulation file must contain `Churn` column.")

    if "customerID" not in df.columns and "customerid" not in df.columns:
        raise KeyError("Simulation file must contain `customerID` or `customerid`.")

    batch = get_next_batch(df, batch_size=batch_size)
    features = batch.drop(columns=["Churn"])

    request_id = f"simulate-churn-day-{simulation_day:03d}-{uuid4()}"

    logger.info("📡 Sending batch of size %s to prediction API...", len(features))
    call_prediction_api(features, request_id=request_id)

    latest_preds = load_logged_predictions(
        request_id=request_id,
        expected_rows=len(features),
    )

    raw_batch = normalize_customer_id_column(batch)
    latest_preds = normalize_customer_id_column(latest_preds)

    pending_df = latest_preds.merge(
        raw_batch,
        on="customerid",
        how="left",
        suffixes=("", "_raw"),
    )

    if "Churn" not in pending_df.columns:
        raise ValueError("Pending batch does not contain raw `Churn` labels.")

    pending_df["churn"] = (
        pending_df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )

    if pending_df["churn"].isna().any():
        raise ValueError("Ground truth contains invalid churn labels.")

    label_available_day = simulation_day + label_delay_days
    now = datetime.now(timezone.utc)

    pending_df["simulation_day"] = simulation_day
    pending_df["label_available_day"] = label_available_day
    pending_df["label_created_at"] = now.isoformat()
    pending_df["pending_batch_id"] = str(uuid4())

    ensure_dir(PENDING_LABEL_DIR)

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    pending_file = (
        f"{PENDING_LABEL_DIR}/"
        f"pending_churn_day_{simulation_day:03d}_"
        f"available_day_{label_available_day:03d}_{timestamp}.csv"
    )

    pending_df.to_csv(pending_file, index=False)
    
    remaining = df.iloc[len(batch):].copy()
    remaining.to_csv(SIMULATION_FILE, index=False)

    logger.info("✅ Pending churn batch saved: %s", pending_file)
    logger.info("Rows: %s", len(pending_df))
    logger.info("Churn rate: %.2f%%", pending_df["churn"].mean() * 100)
    logger.info("Label available on simulation day: %s", label_available_day)
    logger.info("Remaining simulation rows: %s", len(remaining))

    print(f"Pending churn batch written: {pending_file}")
    print(f"Rows: {len(pending_df)}")
    print(f"Churn rate: {pending_df['churn'].mean():.2%}")
    print(f"Label available day: {label_available_day}")
    print(f"Remaining simulation rows: {len(remaining)}")

    return pending_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--simulation-day", type=int, default=1)
    parser.add_argument("--label-delay-days", type=int, default=1)
    args = parser.parse_args()

    simulate_labeled_batch(
        batch_size=args.batch_size,
        simulation_day=args.simulation_day,
        label_delay_days=args.label_delay_days,
    )