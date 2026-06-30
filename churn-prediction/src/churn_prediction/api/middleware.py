"""Middleware de observabilidade: mede e loga a latência de cada requisição."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from churn_prediction.logging_config import get_logger

logger = get_logger("churn_prediction.api.access")


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Loga método, path, status e latência (ms) de cada requisição."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        return response
