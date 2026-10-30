import json

import pandas as pd
import pytest

from mlops_churn_prediction.monitoring import prediction_logger
from mlops_churn_prediction.monitoring.prediction_logger import (
    log_prediction_batch,
)


@pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
def test_batch_is_persisted_without_customer_data_in_stdout(
    tmp_path,
    capsys,
    environment: str,
) -> None:
    output_path = log_prediction_batch(
        inputs=[
            {
                "customerID": "private-customer-1",
                "TotalCharges": "29.85",
                "private_note": "confidential-note",
                "Churn": "Yes",
            },
            {
                "customerID": "private-customer-2",
                "TotalCharges": 5067.45,
            },
        ],
        results=[
            {
                "churn_probability": 0.2,
                "action": "no_action",
                "expected_value": 0.0,
            },
            {
                "churn_probability": 0.8,
                "action": "offer_discount",
                "expected_value": 12.34,
            },
        ],
        request_id="request-1",
        release_id="release-1",
        decision_threshold=0.38,
        environment=environment,
        predictions_path=str(tmp_path),
    )

    assert output_path is not None
    assert len(list(tmp_path.rglob("batch-*.parquet"))) == 1

    stored = pd.read_parquet(output_path)

    assert len(stored) == 2
    assert stored["prediction_id"].nunique() == 2
    assert stored["row_index"].tolist() == [0, 1]
    assert stored["prediction"].tolist() == [0.2, 0.8]
    assert stored["decision_threshold"].tolist() == [0.38, 0.38]
    assert stored["release_id"].tolist() == ["release-1", "release-1"]
    assert stored["environment"].tolist() == [environment, environment]
    assert stored["TotalCharges"].tolist() == ["29.85", "5067.45"]
    assert "private_note" not in stored.columns
    assert "Churn" not in stored.columns

    captured = capsys.readouterr().out
    entry = json.loads(captured.strip())

    assert entry["event"] == "prediction_completed"
    assert entry["rows"] == 2
    assert entry["release_id"] == "release-1"
    assert "private-customer" not in captured
    assert "confidential-note" not in captured
    assert "prediction" not in entry
    assert "action" not in entry


def test_storage_failure_does_not_expose_exception_content(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    def fail_to_persist(*args, **kwargs):
        raise RuntimeError("confidential-customer-value")

    monkeypatch.setattr(
        prediction_logger,
        "persist_prediction_batch",
        fail_to_persist,
    )

    result = log_prediction_batch(
        inputs=[{"customerID": "private-customer"}],
        results=[{"churn_probability": 0.5}],
        request_id="request-1",
        release_id="release-1",
        decision_threshold=0.38,
        predictions_path=str(tmp_path),
    )

    captured = capsys.readouterr().out
    entry = json.loads(captured.strip())

    assert result is None
    assert entry["event"] == "prediction_logging_failed"
    assert entry["error_type"] == "RuntimeError"
    assert "confidential-customer-value" not in captured
    assert "private-customer" not in captured


def test_rejects_mismatched_batch_lengths() -> None:
    with pytest.raises(ValueError, match="equal non-zero lengths"):
        log_prediction_batch(
            inputs=[{"customerID": "customer-1"}],
            results=[],
            request_id="request-1",
            release_id="release-1",
            decision_threshold=0.38,
        )