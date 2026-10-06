from types import SimpleNamespace

import pytest
from model_anatomy.resolver.model import (
    ModelReferenceError,
    ModelResolutionError,
    ModelResolver,
    normalize_model_reference,
)


class FakeHubClient:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def model_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> object:
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "files_metadata": files_metadata,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_normalize_model_id_and_url_revision() -> None:
    assert normalize_model_reference("org/model").model_id == "org/model"
    assert normalize_model_reference(
        "https://huggingface.co/org/model/tree/release-1"
    ).revision == "release-1"


@pytest.mark.parametrize(
    "reference",
    [
        "model",
        "org/model/extra",
        "http://huggingface.co/org/model",
        "https://example.com/org/model",
        "https://huggingface.co/org/model/commits/main",
    ],
)
def test_normalize_rejects_invalid_references(reference: str) -> None:
    with pytest.raises(ModelReferenceError):
        normalize_model_reference(reference)


def test_normalize_rejects_conflicting_revisions() -> None:
    with pytest.raises(ModelReferenceError, match="conflicts"):
        normalize_model_reference(
            "https://huggingface.co/org/model/tree/main", revision="release-1"
        )


def test_resolve_pins_revision_and_returns_sorted_metadata() -> None:
    client = FakeHubClient(
        SimpleNamespace(
            sha="abc123",
            config={"model_type": "llama"},
            siblings=[
                {"rfilename": "model.safetensors", "lfs": {"size": 42}},
                {"rfilename": "config.json", "size": 10},
            ],
        )
    )

    resolved = ModelResolver(client).resolve(
        "https://huggingface.co/org/model/tree/release-1"
    )

    assert client.calls == [
        {"repo_id": "org/model", "revision": "release-1", "files_metadata": True}
    ]
    assert resolved.resolved_revision == "abc123"
    assert [artifact.path for artifact in resolved.artifacts] == [
        "config.json",
        "model.safetensors",
    ]
    assert resolved.artifacts[1].size_bytes == 42
    assert resolved.artifacts[1].is_weight is True
    assert resolved.to_dict()["config"] == {"model_type": "llama"}


def test_resolve_wraps_hub_errors() -> None:
    resolver = ModelResolver(FakeHubClient(ConnectionError("offline")))

    with pytest.raises(ModelResolutionError, match="Could not resolve"):
        resolver.resolve("org/model")
