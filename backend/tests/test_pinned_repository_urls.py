"""Pinned repository catalog actions preserve revision and automatic checkpoint selection."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_anatomy.api.app import create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, model_reference, request_selector
from model_anatomy.service.views import load_ir
from model_anatomy.service.worker import execute_job
from model_anatomy.src.artifacts import parse_reference
from model_anatomy.src.errors import ArtifactError

SHA = "a" * 40
URL = f"https://huggingface.co/owner/model/tree/{SHA}"


def test_pinned_repository_url_retains_automatic_checkpoint_selection():
    assert model_reference(URL) == (URL, "owner/model")
    assert request_selector(URL) == ("owner/model", SHA, "model.safetensors", False)
    reference = parse_reference(URL)
    assert reference.model_id == "owner/model" and reference.revision == SHA
    assert reference.filename == "model.safetensors" and reference.explicit_filename is False


@pytest.mark.parametrize(
    "value",
    [
        f"http://huggingface.co/owner/model/tree/{SHA}",
        f"https://evil.example/owner/model/tree/{SHA}",
        "https://huggingface.co/owner/model/tree/main",
        "https://huggingface.co/owner/model/tree/..",
        "https://huggingface.co/owner/model/tree/%2e%2e",
        f"{URL}?download=true",
        f"{URL}#files",
        f"{URL}/extra",
        f"{URL}/%2e%2e",
    ],
)
def test_new_repository_url_shape_keeps_rejecting_invalid_references(value):
    with pytest.raises(ServiceError):
        model_reference(value)
    with pytest.raises(ArtifactError):
        parse_reference(value)


def test_public_api_accepts_revision_only_repository_url(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        client.app.state.jobs.acquire_worker("worker")
        response = client.post("/api/v1/analyses", json={"model": URL})
        assert response.status_code == 202
        assert response.json()["model_id"] == "owner/model"
        assert response.json()["status"] == "queued"


def test_pinned_catalog_flow_resolves_and_caches_exact_commit(tmp_path):
    service = JobService(Settings(data_dir=tmp_path, guest_jobs_per_hour=5))
    service.initialize()
    service.acquire_worker("worker")
    guest, _ = service.guest(None)
    ir = load_ir((Path(__file__).resolve().parents[2] / "output/gpt2/model.json").read_bytes())
    revision = ir.metadata["revision"]
    url = f"https://huggingface.co/{ir.model_id}/tree/{revision}"
    resolved = []

    class Client:
        def resolve(self, reference):
            assert reference.revision == revision
            assert reference.explicit_filename is False
            resolved.append(reference)
            return revision

    def reconstruct(value, *, client, revision, on_progress):
        assert value == url and revision == ir.metadata["revision"]
        return ir

    job = service.create(guest, url)
    service.claim("worker")
    execute_job(service, job["id"], "worker", Client(), reconstruct, parse_reference)
    ready = service.get(job["id"], guest)
    assert ready.revision == revision
    cached = service.create(guest, url)
    assert cached["cached"] is True and cached["revision"] == revision
    assert service.get(cached["id"], guest).attempts == 0 and len(resolved) == 1
    service.db.engine.dispose()


def test_all_250_catalog_actions_preserve_their_pinned_selection():
    import json

    root = Path(__file__).resolve().parents[2]
    cases = json.loads((root / "benchmarks/model-matrix-250.json").read_text())["cases"]
    assert len(cases) == 250
    for case in cases:
        value = case.get("reference") or f"https://huggingface.co/{case['model_id']}/tree/{case['revision']}"
        assert model_reference(value)[1] == case["model_id"]
        model_id, revision, artifact, explicit = request_selector(value)
        native = parse_reference(value)
        assert model_id == native.model_id == case["model_id"]
        assert revision == native.revision == case["revision"]
        assert explicit == native.explicit_filename == bool(case.get("reference"))
        assert artifact == native.filename
