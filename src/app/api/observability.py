from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import get_logger


log = get_logger("api.http")

REQ_COUNTER = Counter(
    "antifraud_http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status"],
)

REQ_LATENCY = Histogram(
    "antifraud_http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path", "status"],
    buckets=(
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ),
)


@dataclass(frozen=True)
class HttpLogEvent:
    request_id: str
    method: str
    path: str
    status: int
    latency_ms: float


def _get_or_create_request_id(request: Request) -> str:
    header_val = request.headers.get("x-request-id")
    if header_val:
        return header_val.strip()[:128]
    return str(uuid.uuid4())


def metrics_payload() -> tuple[bytes, str]:
    body = generate_latest()
    return body, CONTENT_TYPE_LATEST


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    - Ensures X-Request-ID exists (generated if missing)
    - Measures request latency
    - Emits Prometheus metrics
    - Logs a structured event for each request
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = _get_or_create_request_id(request)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_s = time.perf_counter() - start
            status = 500
            path = request.url.path
            method = request.method.upper()

            REQ_COUNTER.labels(method=method, path=path, status=str(status)).inc()
            REQ_LATENCY.labels(method=method, path=path, status=str(status)).observe(latency_s)

            evt = HttpLogEvent(
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                latency_ms=latency_s * 1000.0,
            )
            log.error(
                "http_request",
                extra={
                    "request_id": evt.request_id,
                    "method": evt.method,
                    "path": evt.path,
                    "status": evt.status,
                    "latency_ms": evt.latency_ms,
                },
            )
            raise

        latency_s = time.perf_counter() - start
        status = int(response.status_code)
        path = request.url.path
        method = request.method.upper()

        response.headers["X-Request-ID"] = request_id

        REQ_COUNTER.labels(method=method, path=path, status=str(status)).inc()
        REQ_LATENCY.labels(method=method, path=path, status=str(status)).observe(latency_s)

        evt = HttpLogEvent(
            request_id=request_id,
            method=method,
            path=path,
            status=status,
            latency_ms=latency_s * 1000.0,
        )
        log.info(
            "http_request",
            extra={
                "request_id": evt.request_id,
                "method": evt.method,
                "path": evt.path,
                "status": evt.status,
                "latency_ms": evt.latency_ms,
            },
        )
        return response
