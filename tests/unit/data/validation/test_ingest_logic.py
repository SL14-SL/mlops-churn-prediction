import os
import shutil
from unittest.mock import patch

import pandas as pd

from src.data.raw.ingest import ingest


def make_valid_churn_df() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "customerID": ["1234-ABCDE", "5678-FGHIJ", "9012-KLMNO", "3456-PQRST"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [12, 3, 24, 6],
            "PhoneService": ["Yes", "Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No", "No phone service"],
            "InternetService": ["Fiber optic", "DSL", "No", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "No"],
            "OnlineBackup": ["Yes", "No", "No internet service", "Yes"],
            "DeviceProtection": ["No", "No", "No internet service", "No"],
            "TechSupport": ["No", "Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "No"],
            "StreamingMovies": ["No", "No", "No internet service", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [70.35, 45.0, 20.0, 55.5],
            "TotalCharges": ["845.50", "135.0", "480.0", "333.0"],
            "Churn": ["Yes", "No", "No", "Yes"],
        }
    )

    df = pd.concat([base] * 5, ignore_index=True)
    df["customerID"] = [f"customer-{i:03d}" for i in range(len(df))]
    return df


def test_ingest_quarantine_logic():
    with patch.dict(os.environ, {"APP_ENV": "dev"}):
        base_test_dir = "data/test_tmp"
        raw_dir = os.path.join(base_test_dir, "raw")
        batch_dir = os.path.join(raw_dir, "new_batches")
        quarantine_dir = os.path.join(raw_dir, "quarantine")
        validated_dir = os.path.join(base_test_dir, "validation")

        if os.path.exists(base_test_dir):
            shutil.rmtree(base_test_dir)

        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(quarantine_dir, exist_ok=True)

        base_df = make_valid_churn_df()
        base_df.to_csv(os.path.join(raw_dir, "Telco-Customer-Churn.csv"), index=False)

        invalid_batch = base_df.iloc[[0]].copy()
        invalid_batch["tenure"] = -1

        bad_batch_path = os.path.join(batch_dir, "bad_batch.csv")
        invalid_batch.to_csv(bad_batch_path, index=False)

        with patch("src.data.raw.ingest.get_path") as mock_get:
            mock_get.side_effect = (
                lambda x: raw_dir
                if x == "raw_data"
                else validated_dir
                if x == "validated_data"
                else os.path.join(base_test_dir, x)
            )

            ingest()

        quarantined_files = os.listdir(quarantine_dir)

        assert any("bad_batch" in filename for filename in quarantined_files)
        assert not os.path.exists(bad_batch_path)
        assert os.path.exists(os.path.join(validated_dir, "train.parquet"))
        assert os.path.exists(os.path.join(raw_dir, "simulation_ground_truth.csv"))

        shutil.rmtree(base_test_dir)