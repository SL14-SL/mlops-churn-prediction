import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from mlops_churn_prediction.monitoring.config import (
    get_serving_settings,
)
from mlops_churn_prediction.monitoring.serving import (
    normalize_path,
    observe_request,
    should_ignore_path,
)

SERVING_CFG = get_serving_settings()


async def serving_monitoring_middleware(
    request: Request,
    call_next: Callable[
        [Request],
        Awaitable[Response],
    ],
) -> Response:
    """
    Record bounded request and latency metrics.

    Monitoring, documentation and configured internal endpoints are
    excluded to prevent self-observation from distorting API metrics.
    """
    if not SERVING_CFG.get(
        "enabled",
        True,
    ):
        return await call_next(
            request
        )

    raw_path = request.url.path
    configured_ignored_paths = set(
        SERVING_CFG.get(
            "ignored_paths"
        )
        or []
    )

    if (
        should_ignore_path(raw_path)
        or raw_path
        in configured_ignored_paths
    ):
        return await call_next(
            request
        )

    started_at = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(
            request
        )
        status_code = response.status_code
        return response

    finally:
        route = request.scope.get(
            "route"
        )
        route_path = getattr(
            route,
            "path",
            None,
        )

        observe_request(
            method=request.method,
            path=normalize_path(
                raw_path,
                route_path,
            ),
            status_code=status_code,
            latency_seconds=(
                time.perf_counter()
                - started_at
            ),
        )