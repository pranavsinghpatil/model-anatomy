"""HTTP boundary for metadata jobs; never import or run model extraction here."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException

from model_anatomy.service.ai import ExplanationService
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, job_detail
from model_anatomy.service.views import EXAMPLES, examples, export, graph_view, load_ir, node_view, prepared

from .schemas import (
    AnalysesResponse,
    AnalysisDetail,
    Capabilities,
    CreateAnalysis,
    ErrorResponse,
    ExamplesResponse,
    GraphResponse,
    Health,
    NodeInspection,
    Question,
    QuestionAnswer,
    Readiness,
)
from .settings import Settings

logger = logging.getLogger("model_anatomy.api")
COOKIE = "model_anatomy_visitor"


def create_app(settings: Settings | None = None, *, service=None, explanation=None) -> FastAPI:
    config = settings if settings is not None else Settings()
    jobs = service or JobService(config)
    ai = explanation or ExplanationService(config)

    @asynccontextmanager
    async def lifespan(app):
        try:
            jobs.initialize()
            verify_migrations()
            app.state.service_ready = True
        except (ServiceError, SQLAlchemyError, OSError):
            app.state.service_ready = False
            logger.error("service initialization unavailable")
        yield
        jobs.db.engine.dispose()
        if explanation is None:
            ai.client.close()
        if hasattr(jobs.storage, "client"):
            jobs.storage.client.close()

    app = FastAPI(
        title="Model Anatomy API",
        version="1.1",
        lifespan=lifespan,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
        responses={status: {"model": ErrorResponse} for status in (404, 410, 422, 429, 503)},
    )
    app.state.service_ready = False
    app.state.jobs = jobs
    app.state.explanation = ai
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.allowed_hosts, www_redirect=False)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    def error(request, status: int, code: str, message: str, headers=None):
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message, "request_id": request.state.request_id}},
            headers=headers,
        )

    def verify_migrations():
        if config.environment == "production":
            with jobs.db.read() as session:
                revisions = session.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
                if revisions != ["0001_durable_jobs"]:
                    raise ServiceError(503, "migrations_required", "Run Alembic upgrade head before serving")

    def require_service():
        if not app.state.service_ready:
            raise ServiceError(503, "service_unavailable", "Job database or private storage is unavailable")

    def guest(request: Request) -> str:
        require_service()
        if not hasattr(request.state, "guest_id"):
            request.state.guest_id, request.state.new_cookie = jobs.guest(request.cookies.get(COOKIE))
        return request.state.guest_id

    def detail(request: Request, analysis_id: str) -> dict:
        if analysis_id in EXAMPLES:
            return prepared(config, analysis_id)[0]
        return job_detail(jobs.get(analysis_id, guest(request)))

    def artifact(request: Request, analysis_id: str):
        if analysis_id in EXAMPLES:
            return prepared(config, analysis_id)[1]
        job = jobs.get(analysis_id, guest(request))
        if job.status not in {"completed", "partial"} or not job.blob_digest:
            raise ServiceError(503, "analysis_not_ready", "Analysis metadata is not available yet")
        return load_ir(jobs.storage.get(job.blob_digest))

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request.state.request_id = uuid4().hex
        origin = request.headers.get("origin")
        if request.method in {"POST", "DELETE"} and origin and origin not in config.cors_origins:
            response = error(request, 403, "origin_rejected", "Request origin is not allowed")
        else:
            try:
                if request.method == "POST":
                    length = request.headers.get("content-length")
                    if length and (not length.isdecimal() or int(length) > 16384):
                        raise ServiceError(413, "request_too_large", "Request body exceeds limit")
                    body = bytearray()
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > 16384:
                            raise ServiceError(413, "request_too_large", "Request body exceeds limit")
                        body.extend(chunk)
                    # Starlette's middleware replays this bounded body to downstream routing.
                    request._body = bytes(body)
                response = await call_next(request)
            except ServiceError as exc:
                response = error(request, exc.status, exc.code, exc.message)
            except (SQLAlchemyError, OSError):
                response = error(request, 503, "service_unavailable", "Service dependency unavailable")
            except Exception:  # noqa: BLE001 - never expose unexpected dependency internals
                logger.error("request failed unexpectedly request_id=%s", request.state.request_id)
                response = error(request, 503, "service_unavailable", "Service dependency unavailable")
        if response.status_code >= 400 and "application/json" not in response.headers.get("content-type", ""):
            headers = {k: v for k, v in response.headers.items() if k.startswith("access-control")}
            response = error(
                request, response.status_code, f"http_{response.status_code}", "Request rejected", headers
            )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        if getattr(request.state, "new_cookie", None):
            response.set_cookie(
                COOKIE,
                request.state.new_cookie,
                max_age=365 * 86400,
                httponly=True,
                secure=config.environment == "production",
                samesite="lax",
                path="/",
            )
        logger.info(
            "request_id=%s method=%s status=%s",
            request.state.request_id,
            request.method,
            response.status_code,
        )
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(request, exc.status_code, f"http_{exc.status_code}", str(exc.detail), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error(request, 422, "validation_error", "Request fields failed validation")

    @app.get("/healthz", response_model=Health)
    def health():
        return Health()

    @app.get("/readyz", response_model=Readiness, responses={503: {"model": ErrorResponse}})
    def readiness(request: Request):
        try:
            require_service()
            verify_migrations()
            jobs.storage.check()
            if not jobs.worker_available():
                raise ServiceError(503, "worker_unavailable", "Reconstruction worker is offline")
        except (SQLAlchemyError, OSError):
            raise ServiceError(503, "service_unavailable", "Service dependency unavailable") from None
        return {"status": "ready", "reason": None}

    @app.get("/api/v1/capabilities", response_model=Capabilities)
    def capabilities():
        available = False
        if app.state.service_ready:
            try:
                available = jobs.worker_available()
            except SQLAlchemyError:
                pass
        return Capabilities(
            reconstruction_available=available,
            ai_available=ai.available,
            storage_backend=config.storage_backend,
            supported_models=["gpt2", "llama"],
        )

    @app.get("/api/v1/examples", response_model=ExamplesResponse)
    def list_examples():
        return {"examples": examples(config)}

    @app.get("/api/v1/analyses", response_model=AnalysesResponse)
    def list_analyses(request: Request):
        return {"analyses": jobs.recent(guest(request))}

    @app.post("/api/v1/analyses", status_code=202, response_model=AnalysisDetail)
    def create_analysis(body: CreateAnalysis, request: Request):
        return jobs.create(guest(request), body.model)

    @app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisDetail)
    def get_analysis(analysis_id: str, request: Request):
        return detail(request, analysis_id)

    @app.delete("/api/v1/analyses/{analysis_id}", response_model=AnalysisDetail)
    def cancel_analysis(analysis_id: str, request: Request):
        if analysis_id in EXAMPLES:
            raise ServiceError(422, "prepared_immutable", "Prepared analyses cannot be cancelled")
        return jobs.cancel(analysis_id, guest(request))

    @app.get("/api/v1/analyses/{analysis_id}/graphs/{graph_id}", response_model=GraphResponse)
    def get_graph(analysis_id: str, graph_id: str, request: Request):
        return graph_view(artifact(request, analysis_id), graph_id)

    @app.get("/api/v1/analyses/{analysis_id}/nodes/{graph_id}/{node_id}", response_model=NodeInspection)
    def get_node(
        analysis_id: str,
        graph_id: str,
        node_id: str,
        request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
    ):
        return node_view(artifact(request, analysis_id), graph_id, node_id, offset, limit)

    @app.get("/api/v1/analyses/{analysis_id}/exports/{format}")
    def get_export(analysis_id: str, format: str, request: Request):
        data, media, filename = export(artifact(request, analysis_id), format)
        return Response(
            data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    @app.post("/api/v1/analyses/{analysis_id}/questions", response_model=QuestionAnswer)
    def ask(analysis_id: str, body: Question, request: Request):
        ir = artifact(request, analysis_id)
        if not ai.available:
            raise ServiceError(
                503, "ai_unavailable", "Set NEBIUS_API_KEY and NEBIUS_MODEL to enable explanations"
            )
        if body.node_id and not body.graph_id:
            raise ServiceError(422, "graph_required", "A node question requires graph_id")
        jobs.question_quota(guest(request))
        return ai.answer(ir, body.question, body.graph_id, body.node_id)

    return app
