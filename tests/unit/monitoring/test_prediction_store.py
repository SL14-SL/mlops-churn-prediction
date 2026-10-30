from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from mlops_churn_prediction.monitoring.prediction_store import (
    persist_prediction_batch,
)


def prediction_records(request_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_id": [
                f"{request_id}-row-1",
                f"{request_id}-row-2",
            ],
            "prediction_timestamp": [
                "2026-10-29T12:00:00+00:00",
                "2026-10-29T12:00:00+00:00",
            ],
            "prediction": [0.2, 0.8],
            "request_id": [request_id, request_id],
            "release_id": ["release-1", "release-1"],
        }
    )


def test_parallel_batches_do_not_overwrite_each_other(tmp_path) -> None:
    def write_batch(request_id: str) -> str:
        return persist_prediction_batch(
            prediction_records(request_id),
            predictions_path=str(tmp_path),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(write_batch, ["request-1", "request-2"]))

    assert len(set(paths)) == 2

    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)

    assert len(combined) == 4
    assert combined["prediction_id"].nunique() == 4
    assert set(combined["request_id"]) == {"request-1", "request-2"}
    assert not list(tmp_path.rglob("*.tmp"))


def test_batch_preserves_lineage_and_source_frame(tmp_path) -> None:
    records = prediction_records("request-1")

    path = persist_prediction_batch(
        records,
        predictions_path=str(tmp_path),
    )
    loaded = pd.read_parquet(path)

    assert Path(path).parent.name == "prediction_date=2026-10-29"
    assert loaded["release_id"].tolist() == ["release-1", "release-1"]
    assert loaded["prediction"].tolist() == [0.2, 0.8]
    assert loaded["prediction_date"].tolist() == ["2026-10-29", "2026-10-29"]
    assert "prediction_date" not in records.columns


def test_rejects_empty_batch(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty prediction batch"):
        persist_prediction_batch(
            pd.DataFrame(),
            predictions_path=str(tmp_path),
        )


@pytest.mark.parametrize(
    "column",
    ["prediction_id", "prediction_timestamp", "request_id", "release_id"],
)
def test_rejects_missing_metadata(tmp_path, column: str) -> None:
    records = prediction_records("request-1").drop(columns=column)

    with pytest.raises(ValueError, match="missing columns"):
        persist_prediction_batch(
            records,
            predictions_path=str(tmp_path),
        )


def test_rejects_duplicate_prediction_ids(tmp_path) -> None:
    records = prediction_records("request-1")
    records["prediction_id"] = "duplicate-id"

    with pytest.raises(ValueError, match="must be unique"):
        persist_prediction_batch(
            records,
            predictions_path=str(tmp_path),
        )


def test_rejects_batches_spanning_multiple_utc_dates(tmp_path) -> None:
    records = prediction_records("request-1")
    records.loc[1, "prediction_timestamp"] = "2026-10-30T12:00:00+00:00"

    with pytest.raises(ValueError, match="one UTC date"):
        persist_prediction_batch(
            records,
            predictions_path=str(tmp_path),
        )