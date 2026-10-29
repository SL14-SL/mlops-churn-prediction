from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from mlops_churn_prediction.configs.loader import get_path
from mlops_churn_prediction.monitoring.prediction_store import (
    persist_prediction_batch,
)


PREDICTIONS = get_path("predictions")

# Raw fields needed for customer matching, drift and subsequent evaluation.
ALLOWED_INPUT_COLUMNS = frozenset({
    "customerid",
    "customer_id",
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
})

STRING_LOG_COLUMNS = {
    "customerID",
    "customerid",
    "customer_id",
    "TotalCharges",
    "totalcharges",
    "prediction_id",
    "prediction_timestamp",
    "prediction_date",
    "environment",
    "model_name",
    "model_alias",
    "model_version",
    "model_run_id",
    "request_id",
    "release_id",
    "action",
}


def normalize_prediction_log_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers, metadata and string-valued features."""
    result = df.copy()

    for column in STRING_LOG_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("string")

    return result


def log_prediction_batch(
    *,
    inputs: list[dict],
    results: list[dict],
    request_id: str,
    release_id: str,
    decision_threshold: float,
    model_name: str | None = None,
    model_alias: str | None = None,
    model_version: str | None = None,
    model_run_id: str | None = None,
    environment: str | None = None,
    predictions_path: str | None = None,
    fail_on_error: bool = False,
) -> str | None:
    """Persist one request batch and emit technical metadata only."""
    if not inputs or len(inputs) != len(results):
        raise ValueError("Prediction inputs and results must have equal non-zero lengths.")

    for name, value in (("request_id", request_id), ("release_id", release_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    threshold = float(decision_threshold)

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be between zero and one.")

    metadata = {
        "request_id": request_id,
        "release_id": release_id,
        "environment": environment or os.getenv("APP_ENV", "dev"),
        "model_name": model_name,
        "model_alias": model_alias,
        "model_version": model_version,
        "model_run_id": model_run_id,
    }

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        records = []

        for row_index, (features, result) in enumerate(zip(inputs, results, strict=True)):
            probability = float(result["churn_probability"])

            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Prediction probability must be between zero and one.")

            selected_features = {
                key: value
                for key, value in features.items()
                if key.lower() in ALLOWED_INPUT_COLUMNS
            }

            records.append({
                **selected_features,
                **metadata,
                "prediction_id": str(uuid4()),
                "prediction_timestamp": timestamp,
                "row_index": row_index,
                "prediction": probability,
                "decision_threshold": threshold,
                "action": result.get("action"),
                "expected_value": result.get("expected_value"),
                "customer_value": result.get("customer_value"),
            })

        frame = normalize_prediction_log_schema(pd.DataFrame(records))

        output_path = persist_prediction_batch(
            frame,
            predictions_path=(
                predictions_path if predictions_path is not None else PREDICTIONS
            ),
        )
    except Exception as exc:
        # Exception text can contain record values; emit its type only.
        print(
            json.dumps({
                "event": "prediction_logging_failed",
                "service": "prediction_api",
                "rows": len(inputs),
                "error_type": type(exc).__name__,
                **metadata,
            }),
            file=sys.stdout,
            flush=True,
        )

        if fail_on_error:
            raise

        return None

    print(
        json.dumps({
            "event": "prediction_completed",
            "service": "prediction_api",
            "rows": len(inputs),
            **metadata,
        }),
        file=sys.stdout,
        flush=True,
    )

    return output_path