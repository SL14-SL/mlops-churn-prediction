import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_api_dependencies(monkeypatch, mock_xgb_model):
    """
    Mock API dependencies so integration tests do not depend on MLflow,
    startup model loading, GCS, or the full feature pipeline.
    """
    monkeypatch.setenv("API_KEY", "test-secret-key")
    monkeypatch.setenv("APP_ENV", "dev")

    mocked_pipeline_output = {
        "environment": "dev",
        "results": [
            {
                "prediction_id": "test-prediction-id",
                "customer_id": "1234-ABCDE",
                "churn_probability": 0.82,
                "churn_prediction": 1,
                "action": "offer_discount",
                "expected_value": 12.3,
                "customer_value": 100.0,
            }
        ],
        "request_id": "test-request",
        "timings": {"total_ms": 1.0},
        "dq_summary": {"quality_status": "ok", "row_count": 1},
    }

    with (
        patch("src.api.app.model", mock_xgb_model),
        patch("src.api.app.model_type", "xgboost"),
        patch("src.api.app.serving_alias", "champion"),
        patch("src.api.app.serving_model_version", "test-version"),
        patch("src.api.app.serving_model_run_id", "test-run-id"),
        patch("src.api.app.feature_schema", None),
        patch("src.api.app.decision_threshold", 0.5),
        patch("src.api.app.run_prediction_pipeline", return_value=mocked_pipeline_output),
        patch("src.api.app.log_prediction"),
    ):
        yield


def test_api_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "online"
    assert "model_name" in body
    assert body["serving_alias"] == "champion"
    assert body["model_version"] == "test-version"


def test_predict_endpoint_validation_error(api_client, api_headers):
    bad_payload = {"inputs": []}
    response = api_client.post("/predict", json=bad_payload, headers=api_headers)

    assert response.status_code == 422


def test_predict_endpoint_success(api_client, api_headers, sample_prediction_payload):
    response = api_client.post("/predict", json=sample_prediction_payload, headers=api_headers)

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1

    prediction = body["predictions"][0]
    assert prediction["churn_probability"] == 0.82
    assert prediction["action"] == "offer_discount"
    assert prediction["expected_value"] == 12.3

    assert "metadata" in body
    assert body["metadata"]["rows"] == 1
    assert body["metadata"]["request_id"] == "test-request"


def test_predict_endpoint_requires_api_key(api_client, sample_prediction_payload):
    response = api_client.post("/predict", json=sample_prediction_payload)

    assert response.status_code == 403


def test_metrics_endpoint_exposes_custom_metrics(api_client):
    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert "api_request_count_total" in response.text
    assert "api_request_latency_seconds" in response.text