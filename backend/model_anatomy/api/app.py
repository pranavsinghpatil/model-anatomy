"""HTTP boundary for metadata jobs; never import or run model extraction here."""

import logging
import time
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
from model_anatomy.service.diagnostics import configure_logging, log_event
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService
from model_anatomy.service.views import artifact_view, export, graph_view, load_ir, node_view

from .schemas import (
    AnalysesResponse,
    AnalysisDetail,
    ArtifactInspectionResponse,
    CreateAnalysis,
    ErrorResponse,
    GraphResponse,
    Health,
    HealthChecks,
    NodeInspection,
    Question,
    QuestionAnswer,
)
from .settings import Settings

logger = logging.getLogger("model_anatomy.api")
COOKIE = "model_anatomy_visitor"


def create_app(settings: Settings | None = None, *, service=None, explanation=None) -> FastAPI:
    config = settings if settings is not None else Settings()
    configure_logging(config.log_level)
    jobs = service or JobService(config)
    ai = explanation or ExplanationService(config)

    @asynccontextmanager
    async def lifespan(app):
        try:
            jobs.initialize()
            verify_migrations()
            app.state.service_ready = True
        except (ServiceError, SQLAlchemyError, OSError) as exc:
            app.state.service_ready = False
            log_event(logger, "service_initialization_failed", exception=exc, level=logging.ERROR)
        yield
        jobs.db.engine.dispose()
        if explanation is None:
            ai.client.close()
        if hasattr(jobs.storage, "client"):
            jobs.storage.client.close()

    app = FastAPI(
        title="Model Anatomy API",
        description=(
            "Analyze a Hugging Face model from its actual configuration, native structure and "
            "checkpoint metadata. Submit a model, poll its persisted analysis, then inspect graphs, "
            "tensors and exports. Every report belongs to a submitted analysis. "
            "Retain the visitor cookie to access your analyses."
        ),
        version="1.1",
        lifespan=lifespan,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
        responses={status: {"model": ErrorResponse} for status in (404, 409, 410, 422, 429, 503)},
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
        log_event(
            logger,
            "request_rejected",
            request_id=request.state.request_id,
            status=status,
            code=code,
            method=request.method,
            level=logging.WARNING,
        )
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
        return jobs.present(jobs.get(analysis_id, guest(request)))

    def artifact(request: Request, analysis_id: str):
        job = jobs.get(analysis_id, guest(request))
        if job.status not in {"completed", "partial", "inspected"} or not job.blob_digest:
            raise ServiceError(503, "analysis_not_ready", "Analysis metadata is not available yet")
        return load_ir(jobs.storage.get(job.blob_digest))

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.monotonic()
        request.state.request_id = uuid4().hex
        origin = request.headers.get("origin")
        same_origin = str(request.url.replace(path="", query="")).rstrip("/")
        if (
            request.method in {"POST", "DELETE"}
            and origin
            and origin != same_origin
            and origin not in config.cors_origins
        ):
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
            except (SQLAlchemyError, OSError) as exc:
                log_event(
                    logger,
                    "request_dependency_failed",
                    request_id=request.state.request_id,
                    exception=exc,
                    level=logging.ERROR,
                )
                response = error(request, 503, "service_unavailable", "Service dependency unavailable")
            except Exception as exc:  # noqa: BLE001 - never expose unexpected dependency internals
                log_event(
                    logger,
                    "request_failed",
                    request_id=request.state.request_id,
                    exception=exc,
                    level=logging.ERROR,
                )
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
        log_event(
            logger,
            "request_finished",
            request_id=request.state.request_id,
            method=request.method,
            status=response.status_code,
            duration_ms=(time.monotonic() - started) * 1000,
            level=logging.DEBUG,
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

    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    def health():
        checks = HealthChecks(database=False, storage=False, worker=False)
        reason = None
        try:
            require_service()
            with jobs.db.read() as session:
                session.execute(text("SELECT 1"))
            verify_migrations()
            checks.database = True
        except (ServiceError, SQLAlchemyError, OSError):
            reason = "database_unavailable"
        try:
            jobs.storage.check()
            checks.storage = True
        except (ServiceError, OSError):
            reason = reason or "storage_unavailable"
        if checks.database:
            try:
                checks.worker = jobs.worker_available()
            except SQLAlchemyError:
                checks.database = False
                reason = "database_unavailable"
        if not checks.worker:
            reason = reason or "worker_unavailable"
        available = all((checks.database, checks.storage, checks.worker))
        result = Health(
            status="ok" if available else "degraded",
            checks=checks,
            analysis_available=available,
            ai_available=ai.available,
            storage_backend=config.storage_backend,
            reason=reason,
        )
        return JSONResponse(status_code=200 if available else 503, content=result.model_dump())

    @app.get("/api/v1/analyses", response_model=AnalysesResponse)
    def list_analyses(request: Request):
        return {"analyses": jobs.recent(guest(request))}

    @app.post(
        "/api/v1/analyses",
        status_code=202,
        response_model=AnalysisDetail,
        summary="Analyze a Hugging Face model",
        description=(
            "Accepts a repository ID or Hugging Face HTTPS URL, persists a new job and returns 202. "
            "The worker resolves an immutable revision and inspects real artifacts. No weights or "
            "remote model code are executed. Poll the returned ID using the same visitor cookie. "
            "An exact-revision cache hit reuses a real inspected report and is marked cached."
        ),
    )
    def create_analysis(body: CreateAnalysis, request: Request):
        created = jobs.create(guest(request), body.model)
        log_event(
            logger,
            "analysis_submitted",
            request_id=request.state.request_id,
            analysis_id=created["id"],
            model_id=created["model_id"],
            engine_version=created["engine_version"],
            stage=created["stage"],
        )
        return created

    @app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisDetail)
    def get_analysis(analysis_id: str, request: Request):
        return detail(request, analysis_id)

    @app.get("/api/v1/analyses/{analysis_id}/artifacts", response_model=ArtifactInspectionResponse)
    def get_artifacts(
        analysis_id: str,
        request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
    ):
        return artifact_view(artifact(request, analysis_id), offset, limit)

    @app.delete("/api/v1/analyses/{analysis_id}", response_model=AnalysisDetail)
    def cancel_analysis(analysis_id: str, request: Request):
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
        if ir.metadata.get("inspection", {}).get("kind") == "artifact_inspection":
            raise ServiceError(
                409,
                "architecture_unavailable",
                "Architecture explanations require a reconstructed architecture; verified artifacts remain available",
            )
        if not ai.available:
            raise ServiceError(
                503, "ai_unavailable", "Set NEBIUS_API_KEY and NEBIUS_MODEL to enable explanations"
            )
        if body.node_id and not body.graph_id:
            raise ServiceError(422, "graph_required", "A node question requires graph_id")
        jobs.question_quota(guest(request))
        return ai.answer(ir, body.question, body.graph_id, body.node_id)

    return app
