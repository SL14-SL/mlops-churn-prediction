import glob
import json
import os
import gcsfs
import pandas as pd
import requests
from src.configs.loader import get_path, load_config

# Load configuration
CFG = load_config()
GCP_CFG = load_config("gcp.yaml")

# Extract API URL from config
API_URL = CFG.get("api", {}).get("url", "http://127.0.0.1:8000/predict")

# Load API key for security
api_key = os.getenv("API_KEY", "secret-token-123") # Default for dev
headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json"
}

# Updated for Churn Dataset
REQUEST_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def _load_latest_batch() -> pd.DataFrame:
    """
    Load the latest available batch from local storage or GCS.
    """
    raw_dir = get_path("raw_data")
    is_gcs = raw_dir.startswith("gs://")
    batch_files = []

    if is_gcs:
        fs = gcsfs.GCSFileSystem()
        bucket_name = GCP_CFG["gcp"]["gcs"]["bucket_name"]
        found_files = fs.ls(f"gs://{bucket_name}/data/raw/new_batches/", detail=True)
        batch_files = [f"gs://{f['name']}" for f in found_files if ".csv" in f["name"]]
        batch_files.sort(key=lambda x: next(f["updated"] for f in found_files if f"gs://{f['name']}" == x), reverse=True)
    else:
        batch_pattern = os.path.join(raw_dir, "new_batches", "*.csv")
        batch_files = glob.glob(batch_pattern)
        batch_files.sort(key=os.path.getctime, reverse=True)

    if batch_files:
        latest_batch = batch_files[0]
        print(f"🔥 Using latest batch data from: {latest_batch}")
        return pd.read_csv(latest_batch)

    # Fallback to ground_truth.csv
    ground_truth_dir = get_path("raw_data")
    ground_truth_file = os.path.join(ground_truth_dir, "simulation_ground_truth.csv")
    print(f"⚠️ No new batches found. Falling back to: {ground_truth_file}")
    return pd.read_csv(ground_truth_file)

def _prepare_request_dataframe(
    df: pd.DataFrame,
    *,
    use_full_batch: bool = True,
    n_requests: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Prepare the request dataframe by filtering for required Churn columns.
    """
    # Remove target column if present
    request_df = df.drop(columns=["Churn", "churn"], errors="ignore").copy()
    
    # Ensure all required columns exist
    missing_cols = [c for c in REQUEST_COLUMNS if c not in request_df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Missing columns in source data: {missing_cols}")
    
    # Keep only defined request columns
    available_cols = [c for c in REQUEST_COLUMNS if c in request_df.columns]
    request_df = request_df[available_cols].copy()

    if not use_full_batch:
        sample_size = min(n_requests, len(request_df))
        request_df = request_df.sample(sample_size, random_state=random_state)
    
    return request_df

def run_stress_test(
    n_requests: int = 100,
    *,
    use_full_batch: bool = True,
    random_state: int = 42,
) -> None:
    """
    Send requests to the Churn prediction API.
    """
    df = _load_latest_batch()

    request_df = _prepare_request_dataframe(
        df,
        use_full_batch=use_full_batch,
        n_requests=n_requests,
        random_state=random_state,
    )

    # Convert to JSON records
    payloads = json.loads(request_df.to_json(orient="records"))

    mode = "full batch" if use_full_batch else f"sample ({len(payloads)} rows)"
    print(f"🚀 Sending {len(payloads)} rows to {API_URL} using {mode} mode...")

    try:
        request_body = {
            "inputs": payloads,
            "context": {"request_id": f"stress-test-{os.getpid()}"}
        }
        
        response = requests.post(
            API_URL,
            json=request_body,
            headers=headers,
            timeout=300,
        )

        if response.status_code == 200:
            body = response.json()
            metadata = body.get('metadata', {})
            print(f"✅ Batch request successful.")
            print(f"   Rows processed: {metadata.get('rows')}")
            print(f"   Model used:     {metadata.get('model_name')}")
            print(f"   Total time:     {metadata.get('timing_ms', {}).get('total')}ms")
        else:
            print(f"❌ Batch request failed with status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Batch request raised exception: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run stress test against Churn prediction API")
    parser.add_argument("--use-full-batch", action="store_true", help="Use full batch")
    parser.add_argument("--n-requests", type=int, default=100, help="Number of sampled rows")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_stress_test(
        n_requests=args.n_requests,
        use_full_batch=args.use_full_batch,
        random_state=args.random_state,
    )