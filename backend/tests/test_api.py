import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from model_anatomy.api.app import create_app
from model_anatomy.api.settings import Settings


@pytest.fixture
def client():
    with TestClient(create_app(Settings()), base_url="http://127.0.0.1") as client:
        yield client


def test_live_service_does_not_claim_reconstruction_is_ready(client):
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "reason": "reconstruction_not_implemented",
    }
    assert client.get("/api/v1/capabilities").json() == {
        "api_version": "v1",
        "reconstruction_available": False,
        "supported_models": [],
    }


def test_request_ids_are_generated_and_errors_are_correlated(client):
    response = client.get("/missing", headers={"X-Request-ID": "untrusted-value"})
    assert response.status_code == 404
    request_id = response.headers["X-Request-ID"]
    assert UUID(request_id).version == 4
    assert response.json()["error"] == {
        "code": "http_404",
        "message": "Not Found",
        "request_id": request_id,
    }
    assert client.get("/healthz").headers["X-Request-ID"] != request_id


def test_unknown_hosts_are_rejected(client):
    assert client.get("/healthz", headers={"Host": "attacker.example"}).status_code == 400


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [("http://localhost:5173", True), ("https://attacker.example", False)],
)
def test_cors_only_allows_configured_origins(client, origin, allowed):
    response = client.options(
        "/api/v1/capabilities",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == (200 if allowed else 400)
    assert ("access-control-allow-origin" in response.headers) is allowed
    assert "access-control-allow-credentials" not in response.headers


def test_settings_use_environment_and_validate_port(monkeypatch):
    monkeypatch.setenv("MODEL_ANATOMY_PORT", "8001")
    monkeypatch.setenv("MODEL_ANATOMY_CORS_ORIGINS", '["http://localhost:3000"]')
    settings = Settings()
    assert settings.port == 8001
    assert settings.cors_origins == ["http://localhost:3000"]
    monkeypatch.setenv("MODEL_ANATOMY_PORT", "70000")
    with pytest.raises(ValidationError):
        Settings()


def test_docs_can_be_disabled():
    with TestClient(
        create_app(Settings(docs_enabled=False)), base_url="http://127.0.0.1"
    ) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_openapi_contract_includes_unavailable_readiness(client):
    schema = client.get("/openapi.json").json()
    assert "503" in schema["paths"]["/readyz"]["get"]["responses"]
    assert "/api/v1/capabilities" in schema["paths"]


def test_http_startup_does_not_import_model_libraries():
    backend = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from model_anatomy.api.app import create_app; import sys; "
                "create_app(); "
                "assert not {'torch', 'transformers', 'safetensors'} & sys.modules.keys()"
            ),
        ],
        cwd=backend,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
