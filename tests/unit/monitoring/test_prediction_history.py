from pathlib import Path

import pandas as pd

from mlops_churn_prediction.monitoring.prediction_store import (
    persist_prediction_batch,
    read_prediction_history,
)


def records(prediction_id: str, probability: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_id": [prediction_id],
            "prediction_timestamp": ["2026-10-29T12:00:00+00:00"],
            "prediction": [probability],
            "request_id": ["request-1"],
            "release_id": ["release-1"],
        }
    )


def test_reads_batch_files_without_legacy_history(tmp_path) -> None:
    persist_prediction_batch(
        records("prediction-1", 0.2),
        predictions_path=str(tmp_path),
    )
    persist_prediction_batch(
        records("prediction-2", 0.8),
        predictions_path=str(tmp_path),
    )

    result = read_prediction_history(predictions_path=str(tmp_path))

    assert len(result) == 2
    assert set(result["prediction_id"]) == {"prediction-1", "prediction-2"}


def test_reads_legacy_history(tmp_path) -> None:
    records("legacy-1", 0.3).to_parquet(
        tmp_path / "inference_log.parquet",
        index=False,
    )

    result = read_prediction_history(predictions_path=str(tmp_path))

    assert result["prediction_id"].tolist() == ["legacy-1"]


def test_batch_record_takes_precedence_over_legacy_duplicate(tmp_path) -> None:
    records("shared-1", 0.2).to_parquet(
        tmp_path / "inference_log.parquet",
        index=False,
    )
    persist_prediction_batch(
        records("shared-1", 0.9),
        predictions_path=str(tmp_path),
    )

    result = read_prediction_history(predictions_path=str(tmp_path))

    assert len(result) == 1
    assert result["prediction"].tolist() == [0.9]


def test_preserves_legacy_rows_without_prediction_ids(tmp_path) -> None:
    pd.DataFrame({"prediction": [0.1, 0.2]}).to_parquet(
        tmp_path / "inference_log.parquet",
        index=False,
    )
    persist_prediction_batch(
        records("new-1", 0.8),
        predictions_path=str(tmp_path),
    )

    result = read_prediction_history(predictions_path=str(tmp_path))

    assert len(result) == 3


def test_ignores_unfinished_batch_files(tmp_path) -> None:
    path = (
        tmp_path
        / "history"
        / "prediction_date=2026-10-29"
        / "batch-incomplete.parquet.tmp"
    )
    path.parent.mkdir(parents=True)
    path.write_text("incomplete", encoding="utf-8")

    result = read_prediction_history(predictions_path=str(tmp_path))

    assert result.empty
    assert Path(path).exists()


def test_returns_empty_history_when_no_files_exist(tmp_path) -> None:
    result = read_prediction_history(predictions_path=str(tmp_path))

    assert result.empty