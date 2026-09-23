import pytest

from mlops_churn_prediction.monitoring.serving import (
    REQUEST_COUNT,
    normalize_path,
    observe_request,
    should_ignore_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "/metrics",
        "/livez",
        "/readyz",
        "/docs",
        "/openapi.json",
    ],
)
def test_should_ignore_operational_paths(path: str) -> None:
    assert should_ignore_path(path) is True


def test_business_path_is_observed() -> None:
    assert should_ignore_path("/predict") is False


def test_normalize_path_uses_route_template() -> None:
    result = normalize_path(
        "/customers/123",
        "/customers/{customer_id}",
    )

    assert result == "/customers/{customer_id}"


def test_normalize_path_bounds_unknown_route() -> None:
    result = normalize_path(
        "/unknown/dynamic/value",
        None,
    )

    assert result == "/unmatched"


def test_observe_request_increments_counter() -> None:
    labels = {
        "method": "POST",
        "path": "/test-observe",
        "status_code": "200",
    }
    counter = REQUEST_COUNT.labels(**labels)
    before = counter._value.get()

    observe_request(
        method="post",
        path="/test-observe",
        status_code=200,
        latency_seconds=0.01,
    )

    assert counter._value.get() == before + 1