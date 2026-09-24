from __future__ import annotations

import threading
from collections import Counter, deque
from time import time

from prometheus_client import Counter as PromCounter
from prometheus_client import Gauge, Histogram

DEFAULT_LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
    3.000,
    5.000,
)

IGNORED_PATHS = {
    "/docs",
    "/docs/oauth2-redirect",
    "/livez",
    "/metrics",
    "/openapi.json",
    "/readyz",
    "/redoc",
}

SERVING_READY = Gauge(
    "mlops_serving_ready",
    (
        "Whether a complete serving bundle "
        "is currently active."
    ),
)

REQUEST_COUNT = PromCounter(
    "mlops_api_requests_total",
    "Total number of observed API requests.",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "mlops_api_request_latency_seconds",
    "API request latency in seconds.",
    ["method", "path"],
    buckets=DEFAULT_LATENCY_BUCKETS_SECONDS,
)

_RECENT_EVENTS = deque(maxlen=5000)
_LOCK = threading.Lock()


def should_ignore_path(path: str) -> bool:
    """Return whether a path should be excluded from serving metrics."""
    return path in IGNORED_PATHS


def normalize_path(
    raw_path: str,
    route_path: str | None,
) -> str:
    """Return a bounded route label for Prometheus."""
    if route_path:
        return route_path

    if raw_path in IGNORED_PATHS:
        return raw_path

    return "/unmatched"


def observe_request(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_seconds: float,
) -> None:
    """Record request metrics and the local operational summary."""
    normalized_method = method.upper()
    normalized_status = str(status_code)

    REQUEST_COUNT.labels(
        method=normalized_method,
        path=path,
        status_code=normalized_status,
    ).inc()

    REQUEST_LATENCY.labels(
        method=normalized_method,
        path=path,
    ).observe(latency_seconds)

    with _LOCK:
        _RECENT_EVENTS.append(
            {
                "ts": time(),
                "method": normalized_method,
                "path": path,
                "status_code": status_code,
                "latency_ms": round(
                    latency_seconds * 1000,
                    2,
                ),
            }
        )


def get_summary(
    window_seconds: int = 900,
) -> dict:
    """Return an in-memory summary for recent API requests."""
    now = time()
    cutoff = now - window_seconds

    with _LOCK:
        events = [
            event
            for event in _RECENT_EVENTS
            if event["ts"] >= cutoff
        ]

    total = len(events)
    successes = sum(
        1
        for event in events
        if 200 <= event["status_code"] < 300
    )
    errors = total - successes

    status_counts = Counter(
        str(event["status_code"])
        for event in events
    )
    path_counts = Counter(
        event["path"]
        for event in events
    )
    latencies = sorted(
        event["latency_ms"]
        for event in events
    )

    def percentile(
        values: list[float],
        percentile_value: float,
    ) -> float | None:
        if not values:
            return None

        index = int(
            round(
                (len(values) - 1)
                * percentile_value
            )
        )
        return round(values[index], 2)

    return {
        "window_seconds": window_seconds,
        "requests_total": total,
        "success_total": successes,
        "error_total": errors,
        "success_rate": (
            round(successes / total, 4)
            if total
            else None
        ),
        "error_rate": (
            round(errors / total, 4)
            if total
            else None
        ),
        "latency_ms": {
            "p50": percentile(
                latencies,
                0.50,
            ),
            "p95": percentile(
                latencies,
                0.95,
            ),
            "p99": percentile(
                latencies,
                0.99,
            ),
            "avg": (
                round(
                    sum(latencies)
                    / len(latencies),
                    2,
                )
                if latencies
                else None
            ),
            "max": (
                round(max(latencies), 2)
                if latencies
                else None
            ),
        },
        "status_codes": dict(
            status_counts
        ),
        "paths": dict(
            path_counts
        ),
    }


def set_serving_readiness(
    is_ready: bool,
) -> None:
    """Update the current serving-readiness state."""
    SERVING_READY.set(
        1 if is_ready else 0
    )
