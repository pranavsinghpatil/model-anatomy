"""Opt-in real HTTP -> durable worker -> Hugging Face inspection, with no mocks."""

import io
import os
import re
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_HTTP_INTEGRATION") != "1", reason="Set RUN_HTTP_INTEGRATION=1"),
]
ROOT = Path(__file__).resolve().parents[2]


def wait_for_result(client: httpx.Client, analysis_id: str) -> dict:
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analyses/{analysis_id}")
        response.raise_for_status()
        detail = response.json()
        if detail["status"] not in {"queued", "running"}:
            return detail
        time.sleep(0.2)
    raise AssertionError("Real HTTP analysis exceeded its deadline")


@pytest.fixture
def live_backend(tmp_path):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    # No credentials or previous cache can influence this run.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MODEL_ANATOMY_", "NEBIUS_")) and key != "DATABASE_URL"
    }
    env.update(
        MODEL_ANATOMY_DATA_DIR=str(tmp_path / "live-data"),
        PYTHONPATH=str(ROOT / "backend"),
    )
    processes = []
    logs = []
    try:
        for name, arguments in (
            (
                "api",
                [
                    "uvicorn",
                    "model_anatomy.api.app:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--no-access-log",
                ],
            ),
            ("worker", ["model_anatomy.service.worker"]),
        ):
            log = (tmp_path / f"{name}.log").open("wb")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", *arguments],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert all(process.poll() is None for process in processes), "A live service exited"
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
            else:
                raise AssertionError("Live services did not become healthy")
            yield client
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        timeout=15,
                        check=False,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for log in logs:
            log.close()


@pytest.mark.parametrize(
    ("model", "revision", "parameters", "component_kind"),
    [
        ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", 361821120, "embedding"),
        ("google/vit-base-patch16-224", "10064cc17b69d623a025c9e90e11230cc8ba6967", 86567656, "classifier"),
    ],
)
def test_real_model_submission_graph_tensors_exports_and_cache(
    live_backend,
    model,
    revision,
    parameters,
    component_kind,
):
    client = live_backend
    model_reference = f"https://huggingface.co/{model}/blob/{revision}/model.safetensors"
    health = client.get("/health").json()
    assert health["analysis_available"] and not health["ai_available"]
    # The same Origin header sent by Swagger/browser POST is accepted.
    response = client.post(
        "/api/v1/analyses",
        json={"model": model_reference},
        headers={"Origin": str(client.base_url).rstrip("/")},
    )
    assert response.status_code == 202
    queued = response.json()
    assert queued["status"] == "queued" and "prepared" not in queued
    detail = wait_for_result(client, queued["id"])
    assert detail["status"] in {"completed", "partial"}, detail.get("error")
    assert detail["model_id"] == model and "prepared" not in detail and detail["cached"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", detail["revision"])
    assert detail["revision"] == revision
    assert detail["parameter_count"] == parameters
    with httpx.Client(timeout=30, follow_redirects=True, trust_env=False) as provider:
        config_response = provider.get(
            f"https://huggingface.co/{model}/resolve/{detail['revision']}/config.json"
        )
        config_response.raise_for_status()
        config = config_response.json()
    assert detail["dimensions"]["hidden"] == config["hidden_size"]
    assert detail["dimensions"]["layers"] == config["num_hidden_layers"]
    path = f"/api/v1/analyses/{detail['id']}"
    graph = client.get(f"{path}/graphs/{detail['root_graph_id']}").json()
    repeated = next(node for node in graph["nodes"] if node["child_graph_id"])
    assert client.get(f"{path}/graphs/{repeated['child_graph_id']}").json()["nodes"]
    index = client.get(f"{path}/graph-index").json()
    component = next(node for node in index["nodes"] if node["kind"] == component_kind)
    node = client.get(f"{path}/nodes/{component['graph_id']}/{component['id']}").json()
    assert node["tensor_count"] > 0 and node["evidence"]
    report = client.get(f"{path}/exports/json").json()
    assert report["model_id"] == model
    assert report["metadata"]["artifacts"]["weight_payload_bytes_read"] == 0
    assert report["tensors"] and report["metadata"]["analysis_report"]["checkpoint_parameter_coverage"] == 1
    if component_kind == "classifier":
        assert report["metadata"]["inputs"]["input_specs"]["pixel_values"]["shape"] is None
        assert [tensor["name"] for tensor in node["tensors"]] == ["classifier.bias", "classifier.weight"]
        assert component["child_graph_id"] is None
        assert report["metadata"]["analysis_report"]["module_inventory_coverage"] == 1
    with zipfile.ZipFile(io.BytesIO(client.get(f"{path}/exports/zip").content)) as archive:
        assert "model.json" in archive.namelist()
    with httpx.Client(base_url=str(client.base_url), trust_env=False) as other:
        assert other.get(path).status_code == 404
    cached = client.post("/api/v1/analyses", json={"model": model_reference})
    assert cached.status_code == 202
    reused = wait_for_result(client, cached.json()["id"])
    assert reused["cached"] is True and "prepared" not in reused
    assert reused["revision"] == detail["revision"]
    assert reused["parameter_count"] == detail["parameter_count"]
    assert client.post("/api/v1/analyses", json={"model": model_reference}).status_code == 429
    with httpx.Client(base_url=str(client.base_url), timeout=15, trust_env=False) as other:
        missing_model = f"model-anatomy-verification/missing-{uuid4().hex}"
        unavailable = other.post("/api/v1/analyses", json={"model": missing_model})
        assert unavailable.status_code == 202
        failed = wait_for_result(other, unavailable.json()["id"])
        assert failed["model_id"] == missing_model
        assert failed["status"] == "failed" and failed["error"]["code"] == "artifact_http"
        assert "prepared" not in failed and failed["root_graph_id"] is None
        assert failed["parameter_count"] is None and failed["architecture"] is None
        assert other.get(f"/api/v1/analyses/{failed['id']}/exports/json").status_code == 503
    print(
        f"Real live report: {model}@{detail['revision']}; {detail['parameter_count']} parameters; "
        f"{len(report['tensors'])} tensors; cold+cached+honest failure verified"
    )
