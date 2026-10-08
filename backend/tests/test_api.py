import asyncio
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from model_anatomy.api.app import COOKIE, create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        yield client


def test_readiness_requires_initialized_services_and_recent_worker(client):
    response = client.get("/health")
    assert response.status_code == 503
    health = response.json()
    assert health["status"] == "degraded"
    assert health["checks"] == {"database": True, "storage": True, "worker": False}
    assert health["analysis_available"] is False
    assert health["ai_available"] is False
    assert health["storage_backend"] == "local"
    assert health["reason"] == "worker_unavailable"
    client.app.state.jobs.acquire_worker("worker")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["analysis_available"] is True
    assert response.json()["reason"] is None
    client.app.state.jobs.release_worker("worker")
    assert client.get("/health").status_code == 503


def test_only_one_health_endpoint_without_fixed_supported_model_list(client):
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    for path in ("/healthz", "/readyz", "/api/v1/capabilities"):
        assert path not in schema["paths"]
        assert client.get(path).status_code == 404
    assert "supported_models" not in client.get("/health").json()


@pytest.mark.parametrize("dependency", ["database", "storage"])
def test_health_reports_real_dependency_failure_without_leaking_details(client, monkeypatch, dependency):
    from sqlalchemy.exc import SQLAlchemyError

    client.app.state.jobs.acquire_worker("worker")
    target = client.app.state.jobs.db if dependency == "database" else client.app.state.jobs.storage

    def unavailable():
        error = SQLAlchemyError("SECRET") if dependency == "database" else OSError("SECRET")
        raise error

    monkeypatch.setattr(target, "read" if dependency == "database" else "check", unavailable)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["checks"][dependency] is False
    assert response.json()["analysis_available"] is False
    assert response.json()["reason"] == f"{dependency}_unavailable"
    assert "SECRET" not in response.text


def test_request_ids_are_generated_and_errors_correlated(client):
    response = client.get("/missing", headers={"X-Request-ID": "untrusted-value"})
    request_id = response.headers["X-Request-ID"]
    assert UUID(request_id).version == 4
    assert response.json()["error"] == {"code": "http_404", "message": "Not Found", "request_id": request_id}
    assert client.get("/health").headers["X-Request-ID"] != request_id


def test_hosts_cors_credentials_and_mutation_origin(client):
    host = client.get("/health", headers={"Host": "attacker.example"})
    assert host.status_code == 400
    assert host.json()["error"]["request_id"] == host.headers["X-Request-ID"]
    for origin, expected in [("http://localhost:5173", 200), ("https://attacker.example", 400)]:
        response = client.options(
            "/api/v1/analyses", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
        )
        assert response.status_code == expected
        if expected == 200:
            assert response.headers["access-control-allow-credentials"] == "true"
    assert (
        client.post(
            "/api/v1/analyses", json={"model": "gpt2"}, headers={"Origin": "https://attacker.example"}
        ).status_code
        == 403
    )


def test_swagger_same_origin_can_submit_and_cancel_without_loosening_cross_origin_guard(client):
    client.app.state.jobs.acquire_worker("worker")
    response = client.post(
        "/api/v1/analyses", json={"model": "HuggingFaceTB/SmolLM2-360M"},
        headers={"Origin": "http://127.0.0.1"},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["model_id"] == "HuggingFaceTB/SmolLM2-360M"
    assert job["prepared"] is False
    path = f"/api/v1/analyses/{job['id']}"
    assert client.delete(path, headers={"Origin": "http://127.0.0.1:8001"}).status_code == 403
    assert client.delete(path, headers={"Origin": "https://127.0.0.1"}).status_code == 403
    assert client.delete(path, headers={"Origin": "http://127.0.0.1"}).json()["status"] == "cancelled"


def test_same_origin_does_not_bypass_host_validation(client):
    client.app.state.jobs.acquire_worker("worker")
    response = client.post(
        "/api/v1/analyses", json={"model": "gpt2"},
        headers={"Host": "attacker.example", "Origin": "http://attacker.example"},
    )
    assert response.status_code == 400
    assert client.get("/api/v1/analyses").json()["analyses"] == []


def test_create_cookie_recent_isolation_and_cancel(client):
    client.app.state.jobs.acquire_worker("worker")
    response = client.post("/api/v1/analyses", json={"model": "gpt2"})
    assert response.status_code == 202
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert COOKIE in client.cookies
    detail = response.json()
    assert detail["status"] == "queued"
    assert detail["root_graph_id"] is None
    assert detail["prepared"] is False
    assert len(client.get("/api/v1/analyses").json()["analyses"]) == 1
    saved = client.cookies.get(COOKIE)
    client.cookies.clear()
    assert client.get(f"/api/v1/analyses/{detail['id']}").status_code == 404
    assert client.get("/api/v1/analyses").json()["analyses"] == []
    client.cookies.set(COOKIE, saved)
    assert client.delete(f"/api/v1/analyses/{detail['id']}").json()["status"] == "cancelled"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"model": ""},
        {"model": 12},
        {"model": "a" * 513},
        {"model": "https://attacker.example/x/y"},
        {"model": "../x"},
    ],
)
def test_validation_failures_have_structured_envelope(client, body):
    response = client.post("/api/v1/analyses", json=body)
    assert response.status_code == 422
    assert set(response.json()["error"]) == {"code", "message", "request_id"}


def test_missing_provider_does_not_block_prepared_graph_and_inspection(client):
    result = client.get("/api/v1/examples").json()["examples"]
    assert {e["id"] for e in result} == {"example-gpt2", "example-smollm"}
    detail = client.get("/api/v1/analyses/example-smollm").json()
    assert detail["prepared"] is True
    assert detail["status"] == "partial"
    assert "SmolLM" in detail["model_id"]
    gpt = client.get("/api/v1/analyses/example-gpt2").json()
    assert gpt["parameter_count"] == 124439808
    graph = client.get(f"/api/v1/analyses/example-gpt2/graphs/{gpt['root_graph_id']}").json()
    assert graph["nodes"] and graph["edges"]
    assert all("child_graph" not in n and "child_graph_id" in n for n in graph["nodes"])
    repeated = next(n for n in graph["nodes"] if n["child_graph_id"])
    assert client.get(f"/api/v1/analyses/example-gpt2/graphs/{repeated['child_graph_id']}").status_code == 200
    inspect = client.get(f"/api/v1/analyses/example-gpt2/nodes/{graph['id']}/{repeated['id']}?limit=1").json()
    assert inspect["tensor_count"] >= len(inspect["tensors"])
    assert len(inspect["tensors"]) == 1
    assert inspect["tensors"][0]["artifact"].startswith("https://huggingface.co/")
    question = client.post(
        "/api/v1/analyses/example-gpt2/questions", json={"question": "How many parameters?"}
    )
    assert question.status_code == 503
    assert question.json()["error"]["code"] == "ai_unavailable"


@pytest.mark.parametrize("format", ["json", "mermaid", "markdown", "diagnostics", "audit", "zip"])
def test_exports_use_same_prepared_ir(client, format):
    response = client.get(f"/api/v1/analyses/example-gpt2/exports/{format}")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    if format == "json":
        assert response.json()["schema_version"] == "1.1"
    if format == "zip":
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert {"model.json", "root.mmd", "diagnostics.json", "artifact_audit.json"} <= set(
                archive.namelist()
            )


def test_dependency_failure_preserves_examples_and_health(tmp_path):
    settings = Settings(data_dir=tmp_path)
    service = JobService(settings)
    service.initialize = lambda: (_ for _ in ()).throw(ServiceError(503, "offline", "offline"))
    with TestClient(create_app(settings, service=service), base_url="http://localhost") as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["checks"]["database"] is False
        assert client.get("/api/v1/analyses/example-gpt2").status_code == 200
        assert client.get("/api/v1/analyses").status_code == 503


def test_settings_exact_environment_names_and_failclosed(monkeypatch):
    monkeypatch.setenv("MODEL_ANATOMY_PORT", "8001")
    monkeypatch.setenv("MODEL_ANATOMY_SUPABASE_KEY", "test-secret")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-ai-secret")
    monkeypatch.setenv("NEBIUS_MODEL", "nvidia/test-nemotron")
    settings = Settings()
    assert settings.port == 8001
    assert settings.supabase_service_role_key.get_secret_value() == "test-secret"
    assert settings.nebius_api_key.get_secret_value() == "test-ai-secret"
    assert settings.nebius_model_id == "nvidia/test-nemotron"
    assert "test-ai-secret" not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(environment="production")
    with pytest.raises(ValidationError):
        Settings(cors_origins=["*"])


def test_invalid_production_settings_never_render_secret_inputs():
    marker = "synthetic-credential-that-must-not-appear"
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            database_url=f"postgresql://user:{marker}@localhost/db",
            supabase_service_role_key=marker,
            nebius_api_key=marker,
        )
    assert marker not in str(exc.value)
    assert "input_value" not in str(exc.value)


def test_docs_can_be_disabled_and_openapi_matches_service(client, tmp_path):
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/health"]["get"]["responses"]["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Health"
    }
    assert "/api/v1/analyses/{analysis_id}/questions" in schema["paths"]
    with TestClient(
        create_app(Settings(data_dir=tmp_path, docs_enabled=False)), base_url="http://localhost"
    ) as other:
        assert other.get("/docs").status_code == 404


def test_http_startup_and_prepared_reads_do_not_import_model_libraries(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from fastapi.testclient import TestClient; from model_anatomy.api.app import create_app; "
                "from model_anatomy.api.settings import Settings; import sys; from pathlib import Path; "
                "app=create_app(Settings(data_dir=Path(sys.argv[1]))); "
                "client=TestClient(app,base_url='http://localhost'); "
                "assert client.get('/api/v1/analyses/example-gpt2').status_code==200; "
                "assert not {'torch','transformers','safetensors'} & sys.modules.keys()"
            ),
            str(tmp_path),
        ],
        cwd=backend,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_openapi_json_routes_have_concrete_response_models(client):
    schema = client.get("/openapi.json").json()
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if method not in {"get", "post", "delete"} or "/exports/" in path:
                continue
            response = operation["responses"].get("202", operation["responses"].get("200"))
            assert response["content"]["application/json"]["schema"].get("$ref"), path
    assert "GraphNode" in schema["components"]["schemas"]
    assert "child_graph" not in schema["components"]["schemas"]["GraphNode"]["properties"]
    assert schema["components"]["schemas"]["QuestionAnswer"]["properties"]["ai_generated"]["const"] is True


@pytest.mark.parametrize("declared_size,read_count", [(20000, 0), (None, 3), ("4", 3)])
def test_body_limit_stops_stream_without_buffering_remaining_chunks(tmp_path, declared_size, read_count):
    app = create_app(Settings(data_dir=tmp_path))
    received, sent = [], []
    chunks = iter([b"x" * 8192] * 5)

    async def receive():
        received.append(1)
        return {"type": "http.request", "body": next(chunks), "more_body": True}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"localhost"), (b"content-type", b"application/json")]
    if declared_size is not None:
        headers.append((b"content-length", str(declared_size).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/analyses",
        "raw_path": b"/api/v1/analyses",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("localhost", 80),
    }
    asyncio.run(app(scope, receive, send))
    assert len(received) == read_count
    assert sent[0]["status"] == 413
    body = b"".join(message.get("body", b"") for message in sent[1:])
    assert json.loads(body)["error"]["code"] == "request_too_large"
