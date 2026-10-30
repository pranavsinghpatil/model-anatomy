"""Public submission validation accepts explicit GGUF selection without weakening URLs."""

import pytest
from fastapi.testclient import TestClient

from model_anatomy.api.app import create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import cache_key, model_reference


def test_explicit_gguf_submission_and_cache_identity():
    revision = "b" * 40
    url = f"https://huggingface.co/prism-ml/model/blob/{revision}/weights.gguf"
    assert model_reference(url) == (url, "prism-ml/model")
    nested = f"https://huggingface.co/prism-ml/model/blob/{revision}/quantized/weights.gguf"
    assert model_reference(nested) == (nested, "prism-ml/model")
    assert cache_key("prism-ml/model", revision, "v1", "1.1", "weights.gguf", True) != cache_key(
        "prism-ml/model", revision, "v1", "1.1", "other.gguf", True
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/a/b/blob/main/weights.gguf",
        "https://huggingface.co/a/b/blob/main/weights.gguf?download=true",
        "https://huggingface.co/a/b/blob/main/weights.bin",
        "https://huggingface.co/a/b/blob/../weights.gguf",
        "https://huggingface.co/a/b/blob/main/%2e%2e/weights.gguf",
        "https://huggingface.co/a/b/blob/main/folder%2fweights.gguf",
        "https://huggingface.co/a/b/blob/main/folder%5cweights.gguf",
    ],
)
def test_invalid_gguf_submission_rejected(url):
    with pytest.raises(ServiceError):
        model_reference(url)


def test_api_accepts_pinned_gguf_files_as_distinct_durable_jobs(tmp_path):
    settings = Settings(data_dir=tmp_path, guest_jobs_per_hour=10)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        service = client.app.state.jobs
        service.acquire_worker("test-worker")
        ids = []
        for variant in ("F16", "PTQ1_0", "PQ2_0"):
            url = f"https://huggingface.co/prism-ml/bonsai/blob/{'b' * 40}/bonsai-{variant}.gguf"
            response = client.post("/api/v1/analyses", json={"model": url})
            assert response.status_code == 202
            result = response.json()
            assert result["status"] == "queued" and result["model_id"] == "prism-ml/bonsai"
            ids.append(result["id"])
        assert len(set(ids)) == 3
        assert len(client.get("/api/v1/analyses").json()["analyses"]) == 3
