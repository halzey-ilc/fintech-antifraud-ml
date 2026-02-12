from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from app.api.schemas import ScoreItem, ScoreRequest, ScoreResponse
from app.api.service import ModelLoadResult, load_model_from_artifacts
from app.core.config import AppConfig, resolve_paths
from app.core.logging import configure_logging, get_logger

log = get_logger("api")

HTTP_REQUESTS_TOTAL = Counter(
    "antifraud_http_requests_total",
    "Total number of HTTP requests processed by the antifraud API.",
    labelnames=("method", "path", "status"),
)


def _resolve_config_path() -> Path:
    env_path = os.getenv("ANTIFRAUD_CONFIG")
    if env_path:
        p = Path(env_path)
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / "configs" / "dev.yaml"
    if candidate.exists():
        return candidate.resolve()
    return (Path.cwd() / "configs" / "dev.yaml").resolve()


def _get_or_create_request_id(request: Request) -> str:
    incoming = request.headers.get("X-Request-ID")
    if incoming and incoming.strip():
        return incoming.strip()
    return str(uuid.uuid4())


def _normalized_path_for_metrics(request: Request) -> str:
    return request.url.path


def _extract_feature_names(model: Any) -> list[str]:
    names: Any | None = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in list(names)]

    steps: Any | None = getattr(model, "steps", None)
    if steps:
        for _, step in steps:
            step_names: Any | None = getattr(step, "feature_names_in_", None)
            if step_names is not None:
                return [str(x) for x in list(step_names)]

    return []


def _align_dataframe_to_model(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    fitted = _extract_feature_names(model)
    if not fitted:
        return df
    return df.reindex(columns=fitted, fill_value=0)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="fintech-antifraud-ml", version="0.1.0")

    state: dict[str, Any] = {
        "config_path": None,
        "config": None,
        "paths": None,
        "model_result": None,
    }

    @app.middleware("http")
    async def request_context_middleware(  # type: ignore[no-untyped-def]
        request: Request,
        call_next: Callable[..., Any],
    ):
        request_id = _get_or_create_request_id(request)
        start = time.perf_counter()

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        method = request.method
        path = _normalized_path_for_metrics(request)
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "http.request method=%s path=%s status=%s ms=%.2f rid=%s",
            method,
            path,
            status,
            elapsed_ms,
            request_id,
        )

        return response

    try:
        config_path = _resolve_config_path()
        cfg = AppConfig.from_yaml(config_path)
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        paths = resolve_paths(cfg, project_root)

        model_result = load_model_from_artifacts(paths.artifacts_dir)

        state["config_path"] = str(config_path)
        state["config"] = cfg
        state["paths"] = paths
        state["model_result"] = model_result

        if model_result.error:
            log.warning("model.load.failed error=%s", model_result.error)
        else:
            log.info("model.load.ok path=%s", str(model_result.model_path))

    except Exception as exc:  # noqa: BLE001
        state["model_result"] = ModelLoadResult(
            model=None,
            model_path=None,
            selection=None,
            threshold=None,
            error=f"API init failed: {exc}",
        )
        log.exception("api.init.failed")

    @app.get("/health")
    def health() -> JSONResponse:
        mr: ModelLoadResult = state["model_result"]
        return JSONResponse(
            {
                "status": "ok",
                "model_loaded": mr.model is not None,
                "model_path": str(mr.model_path) if mr.model_path else None,
                "error": mr.error,
                "config_path": state["config_path"],
            }
        )

    @app.get("/metadata")
    def metadata() -> JSONResponse:
        mr: ModelLoadResult = state["model_result"]
        return JSONResponse(
            {
                "model_loaded": mr.model is not None,
                "model_path": str(mr.model_path) if mr.model_path else None,
                "selection": mr.selection,
                "threshold": mr.threshold,
            }
        )

    @app.get("/metrics")
    def metrics() -> Response:
        payload = generate_latest()
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        mr: ModelLoadResult = state["model_result"]
        if mr.model is None:
            return ScoreResponse(items=[], threshold=None, model_loaded=False, error=mr.error or "model not loaded")

        df = _align_dataframe_to_model(mr.model, req.to_dataframe())

        try:
            proba = mr.model.predict_proba(df)  # type: ignore[call-arg]
            threshold = float(mr.threshold) if mr.threshold is not None else 0.5

            items: list[ScoreItem] = []
            for row in proba:
                p_fraud = float(row[1])
                is_fraud = p_fraud >= threshold
                items.append(
                    ScoreItem(
                        proba_fraud=p_fraud,
                        score=p_fraud,
                        is_fraud=is_fraud,
                        decision="fraud" if is_fraud else "legit",
                    )
                )

            return ScoreResponse(items=items, threshold=threshold, model_loaded=True, error=None)

        except Exception as exc:  # noqa: BLE001
            return ScoreResponse(
                items=[],
                threshold=float(mr.threshold) if mr.threshold is not None else None,
                model_loaded=True,
                error=str(exc),
            )

    return app
