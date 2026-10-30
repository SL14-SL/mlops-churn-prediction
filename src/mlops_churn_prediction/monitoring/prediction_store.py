from pathlib import Path
from uuid import uuid4

import pandas as pd

from mlops_churn_prediction.configs.paths import join_uri
from mlops_churn_prediction.storage.filesystem import file_exists, list_files

REQUIRED_COLUMNS = {
    "prediction_id",
    "prediction_timestamp",
    "prediction",
    "request_id",
    "release_id",
}


def persist_prediction_batch(
    records: pd.DataFrame,
    *,
    predictions_path: str,
) -> str:
    """Persist one prediction batch without modifying previous batches."""
    if records.empty:
        raise ValueError("Cannot persist an empty prediction batch.")

    missing = REQUIRED_COLUMNS - set(records.columns)

    if missing:
        raise ValueError(f"Prediction batch is missing columns: {sorted(missing)}.")

    if not predictions_path.strip():
        raise ValueError("Predictions path must not be empty.")

    frame = records.copy()

    for column in ("prediction_id", "request_id", "release_id"):
        valid = frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )

        if not valid.all():
            raise ValueError(f"Prediction batch contains invalid {column} values.")

    if frame["prediction_id"].duplicated().any():
        raise ValueError("Prediction IDs must be unique within a batch.")

    frame["prediction_timestamp"] = pd.to_datetime(
        frame["prediction_timestamp"],
        errors="raise",
        utc=True,
    )

    if frame["prediction_timestamp"].isna().any():
        raise ValueError("Prediction timestamps must not be missing.")

    prediction_dates = frame["prediction_timestamp"].dt.strftime("%Y-%m-%d")

    if prediction_dates.nunique() != 1:
        raise ValueError("A prediction batch must belong to one UTC date.")

    prediction_date = prediction_dates.iloc[0]
    frame["prediction_date"] = prediction_date

    output_path = join_uri(
        predictions_path,
        "history",
        f"prediction_date={prediction_date}",
        f"batch-{uuid4()}.parquet",
    )

    if output_path.startswith("gs://"):
        frame.to_parquet(output_path, index=False)
    else:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")

        try:
            frame.to_parquet(temporary, index=False)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    return output_path


def read_prediction_history(
    *,
    predictions_path: str,
) -> pd.DataFrame:
    """Read legacy history and completed batch files without duplicate IDs."""
    legacy_path = join_uri(predictions_path, "inference_log.parquet")
    batch_pattern = join_uri(
        predictions_path,
        "history",
        "prediction_date=*",
        "batch-*.parquet",
    )

    paths = []

    if file_exists(legacy_path):
        paths.append(legacy_path)

    # Read batches after legacy data so batch records take precedence.
    paths.extend(sorted(list_files(batch_pattern)))

    frames = []

    for path in paths:
        frame = pd.read_parquet(path)

        if frame.empty:
            continue

        for column in (
            "prediction_id",
            "request_id",
            "release_id",
            "customerID",
            "customerid",
            "customer_id",
            "TotalCharges",
            "totalcharges",
        ):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if "prediction_id" in combined.columns:
        identifiers = combined["prediction_id"].astype("string")
        valid_identifiers = (
            identifiers.notna()
            & identifiers.str.strip().ne("")
        )
        duplicates = (
            valid_identifiers
            & identifiers.duplicated(keep="last")
        )
        combined = combined.loc[~duplicates]

    return combined.reset_index(drop=True)