import json

from mlops_churn_prediction.monitoring.prediction_logger import log_prediction


def test_prediction_stdout_contains_only_technical_metadata(capsys) -> None:
    log_prediction(
        {
            "customerID": "private-customer-123",
            "Contract": "Month-to-month",
            "MonthlyCharges": 79.95,
            "private_note": "confidential-customer-note",
        },
        0.75,
        environment="prod",
        model_alias="champion",
        model_version="1",
        model_run_id="run-1",
        request_id="request-1",
        action="offer_discount",
        expected_value=12.34,
        customer_value=456.78,
    )

    captured = capsys.readouterr().out
    entries = [
        json.loads(line)
        for line in captured.splitlines()
        if line.startswith("{")
    ]

    assert len(entries) == 1
    entry = entries[0]

    assert entry["event"] == "prediction_completed"
    assert entry["service"] == "prediction_api"
    assert entry["rows"] == 1
    assert entry["request_id"] == "request-1"
    assert entry["model_version"] == "1"

    assert not {
        "input",
        "prediction",
        "action",
        "expected_value",
        "customer_value",
    }.intersection(entry)

    assert "private-customer-123" not in captured
    assert "confidential-customer-note" not in captured