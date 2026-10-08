"""Application factory. Model reconstruction is not run inside HTTP handlers."""

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response

from .schemas import Capabilities, ErrorDetail, ErrorResponse, Health, Readiness
from .settings import Settings

logger = logging.getLogger("model_anatomy.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    app = FastAPI(
        title="Model Anatomy API",
        version="0.1.0",
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
        responses={404: {"model": ErrorResponse}, 405: {"model": ErrorResponse}},
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=config.allowed_hosts, www_redirect=False
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def observe_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex
        request.state.request_id = request_id
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # Log operational metadata only, never model references, bodies, or tokens.
        logger.info(
            "request_id=%s method=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        error = ErrorResponse(
            error=ErrorDetail(
                code=f"http_{exc.status_code}",
                message=str(exc.detail),
                request_id=request.state.request_id,
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error.model_dump(),
            headers=exc.headers,
        )

    @app.get("/healthz", response_model=Health, tags=["operations"])
    async def health() -> Health:
        return Health()

    @app.get(
        "/readyz", response_model=Readiness, status_code=503, tags=["operations"]
    )
    async def readiness() -> Readiness:
        return Readiness()

    @app.get("/api/v1/capabilities", response_model=Capabilities, tags=["v1"])
    async def capabilities() -> Capabilities:
        return Capabilities()

    return app
