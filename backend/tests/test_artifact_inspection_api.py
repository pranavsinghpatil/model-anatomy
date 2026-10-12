import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from model_anatomy.api.app import COOKIE, create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.jobs import cache_key
from model_anatomy.service.views import artifact_view
from model_anatomy.src.ir import Evidence, Graph, ModelIR


@pytest.fixture
def inspected(tmp_path):
    settings = Settings(data_dir=tmp_path, guest_jobs_per_hour=10)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        service = client.app.state.jobs
        service.acquire_worker("worker")
        job = client.post("/api/v1/analyses", json={"model": "publisher/adapter"}).json()
        assert service.claim("worker") == job["id"]
        revision = "a" * 40
        service.resolved(job["id"], "worker", "publisher/adapter", revision)
        ir = ModelIR(
            "publisher/adapter",
            "Unavailable",
            Graph("architecture_unavailable", [], []),
            metadata={
                "revision": revision,
                "engine_version": settings.engine_version,
                "inspection": {
                    "kind": "artifact_inspection",
                    "reason_code": "adapter_requires_base_model",
                    "message": "This adapter requires a base model; no architecture was reconstructed.",
                    "files": ["README.md", "adapter_config.json", "adapter_model.safetensors"],
                    "checkpoint_available": True,
                },
                "checkpoint_tensor_element_count": 12,
                "analysis_report": {"mode": "artifact_inspection", "warnings": []},
                "artifacts": {"weight_payload_bytes_read": 0},
                "limitations": ["Tensor elements are not a model parameter count."],
            },
            tensors={
                name: {
                    "name": name,
                    "dtype": "F32",
                    "shape": [2, 3],
                    "parameter_count": 6,
                    "artifact": f"https://huggingface.co/publisher/adapter/resolve/{revision}/adapter_model.safetensors",
                    "parameter_refs": [],
                    "evidence_kind": "direct",
                }
                for name in ("lora_A.weight", "lora_B.weight")
            },
            evidence=[Evidence("safetensors_header", "direct", "adapter_model.safetensors")],
        )
        key = cache_key(ir.model_id, revision, settings.engine_version, settings.schema_version)
        service.finish(job["id"], "worker", ir.to_json().encode(), key, cached=False)
        yield client, job["id"], ir, key


def test_inspected_is_terminal_without_architecture_or_parameter_claim(inspected):
    client, job_id, _, _ = inspected
    path = f"/api/v1/analyses/{job_id}"
    detail = client.get(path).json()
    assert detail["status"] == "inspected"
    assert detail["architecture"] is None and detail["root_graph_id"] is None
    assert detail["parameter_count"] is None
    assert detail["inspection"]["tensor_count"] == 2
    assert detail["inspection"]["file_count"] == 3
    assert detail["quality"]["semantic_coverage"] is None
    assert client.delete(path).json()["status"] == "inspected"
    assert client.get("/api/v1/analyses").json()["analyses"][0]["status"] == "inspected"


def test_artifacts_paginate_verified_records_and_retain_provenance(inspected):
    client, job_id, _, _ = inspected
    response = client.get(f"/api/v1/analyses/{job_id}/artifacts?offset=1&limit=1")
    assert response.status_code == 200
    page = response.json()
    assert page["files"] == ["adapter_config.json"]
    assert page["file_count"] == 3 and page["tensor_count"] == 2
    assert page["tensors"][0]["name"] == "lora_B.weight"
    assert page["tensors"][0]["shape"] == [2, 3]
    assert page["tensors"][0]["evidence_kind"] == "direct"
    assert page["checkpoint_tensor_element_count"] == 12
    assert page["evidence"][0]["source"] == "safetensors_header"
    assert page["revision"] == "a" * 40
    assert client.get(f"/api/v1/analyses/{job_id}/artifacts?limit=101").status_code == 422
    assert client.get(f"/api/v1/analyses/{job_id}/artifacts?offset=-1").status_code == 422


def test_gguf_artifact_view_preserves_selected_split_file_scope(inspected):
    _, _, ir, _ = inspected
    ir.metadata["checkpoint_tensor_element_count"] = None
    ir.metadata["artifacts"].update(
        checkpoint_format="gguf",
        selected_file_tensor_element_count=12,
        selected_file_split_metadata={"split.no": 0, "split.count": 2, "split.tensors.count": 4},
    )
    page = artifact_view(ir)
    assert page["checkpoint_format"] == "gguf"
    assert page["checkpoint_tensor_element_count"] is None
    assert page["selected_file_tensor_element_count"] == 12
    assert page["selected_file_split_metadata"]["split.count"] == 2


def test_artifact_results_do_not_expose_fake_graphs_or_spend_ai_quota(inspected):
    client, job_id, _, _ = inspected
    path = f"/api/v1/analyses/{job_id}"
    for suffix in (
        "graphs/architecture_unavailable",
        "nodes/architecture_unavailable/anything",
        "exports/mermaid",
    ):
        response = client.get(f"{path}/{suffix}")
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "architecture_unavailable"
    response = client.post(f"{path}/questions", json={"question": "Invent an attention graph"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "architecture_unavailable"


def test_artifact_exports_never_contain_a_diagram(inspected):
    client, job_id, _, _ = inspected
    path = f"/api/v1/analyses/{job_id}/exports"
    assert client.get(f"{path}/json").json()["root_graph"]["nodes"] == []
    markdown = client.get(f"{path}/markdown").text
    assert "Architecture unavailable" in markdown and "```mermaid" not in markdown
    assert client.get(f"{path}/audit").json()["weight_payload_bytes_read"] == 0
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"{path}/zip").content))
    assert all(not name.endswith(".mmd") for name in archive.namelist())
    assert "model.json" in archive.namelist()


def test_inspection_access_remains_visitor_owned(inspected):
    client, job_id, _, _ = inspected
    client.cookies.delete(COOKIE)
    for suffix in ("", "/artifacts", "/exports/json"):
        assert client.get(f"/api/v1/analyses/{job_id}{suffix}").status_code == 404


def test_inspected_cache_reuses_real_report_without_changing_outcome(inspected):
    client, _, ir, key = inspected
    service = client.app.state.jobs
    new_job = client.post("/api/v1/analyses", json={"model": ir.model_id}).json()
    service.claim("worker")
    service.resolved(new_job["id"], "worker", ir.model_id, ir.metadata["revision"])
    data = service.cached_result(key)
    assert data == ir.to_json().encode()
    service.finish(new_job["id"], "worker", data, key, cached=True)
    detail = client.get(f"/api/v1/analyses/{new_job['id']}").json()
    assert detail["cached"] is True and detail["status"] == "inspected"


def test_markdown_inventory_does_not_interpret_repository_filenames(inspected):
    from model_anatomy.service.views import export_files

    _, _, ir, _ = inspected
    malicious = "`\n<script>bad()</script>\n[external](https://example.com)"
    ir.metadata["inspection"]["files"] = [malicious]
    markdown = export_files(ir)["model.md"].decode()
    assert "<script>" not in markdown
    assert "\n[external]" not in markdown
    assert "&lt;script&gt;" in markdown and "&#10;" in markdown
