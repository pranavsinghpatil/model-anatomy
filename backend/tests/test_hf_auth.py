"""Optional Hugging Face credentials stay on the canonical metadata origin."""

import io
import json

import pytest
import requests

from model_anatomy.src.artifacts import MetadataClient, parse_reference
from model_anatomy.src.errors import ArtifactError

TOKEN = "hf_fake_test_token"
MODEL_URL = "https://huggingface.co/org/model/resolve/main/config.json"


class Response:
    def __init__(self, body=b'{"ok":true}', status=200, headers=None):
        self.raw = Raw(body)
        self.status_code = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ArtifactError("HTTP failure")


class Raw(io.BytesIO):
    def read(self, size=-1, decode_content=False):
        return super().read(size)


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {"Authorization": "caller-owned"}

    def get(self, url, **kwargs):
        self.calls.append((url, dict(kwargs["headers"]), kwargs))
        return self.responses.pop(0)


def test_explicit_token_authenticates_json_and_ranges_without_audit_leak(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_other_token")
    session = Session(
        Response(),
        Response(b"abcdefgh", 206, {"Content-Range": "bytes 0-7/9"}),
    )
    client = MetadataClient(session, token=TOKEN)

    assert client.json(MODEL_URL) == {"ok": True}
    assert client.range(MODEL_URL, 0, 7) == (b"abcdefgh", 9)
    assert [call[1]["Authorization"] for call in session.calls] == [f"Bearer {TOKEN}"] * 2
    assert session.calls[0][1]["Accept-Encoding"] == "identity"
    assert session.calls[1][1]["Range"] == "bytes=0-7"
    assert all(call[2]["allow_redirects"] is False for call in session.calls)
    assert TOKEN not in str(client.audit)
    assert session.headers == {"Authorization": "caller-owned"}


def test_standard_environment_token_and_explicit_empty_override(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    from_environment = Session(Response())
    anonymous = Session(Response())

    MetadataClient(from_environment).json(MODEL_URL)
    MetadataClient(anonymous, token="").json(MODEL_URL)

    assert from_environment.calls[0][1]["Authorization"] == f"Bearer {TOKEN}"
    assert anonymous.calls[0][1]["Authorization"] is None


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/org/model/config.json",
        "https://other.example/org/model/config.json",
        "https://huggingface.co.evil.example/org/model/config.json",
        "https://user@huggingface.co/org/model/config.json",
        "https://huggingface.co:8443/org/model/config.json",
    ],
)
def test_token_is_not_attached_to_other_origins(monkeypatch, url):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    session = Session(Response())

    MetadataClient(session).json(url)

    assert session.calls[0][1]["Authorization"] is None


def test_redirect_to_checkpoint_host_drops_token_and_preserves_range(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    redirected = "https://cdn-lfs.huggingface.co/object"
    session = Session(
        Response(status=302, headers={"Location": redirected}),
        Response(b"abcdefgh", 206, {"Content-Range": "bytes 0-7/9"}),
    )

    assert MetadataClient(session).range(MODEL_URL, 0, 7) == (b"abcdefgh", 9)
    assert session.calls[0][1]["Authorization"] == f"Bearer {TOKEN}"
    assert session.calls[1][1]["Authorization"] is None
    assert session.calls[1][1]["Range"] == "bytes=0-7"


def test_redirect_back_to_hf_does_not_restore_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    session = Session(
        Response(status=302, headers={"Location": "https://other.example/file"}),
        Response(status=302, headers={"Location": MODEL_URL}),
        Response(),
    )

    MetadataClient(session).json(MODEL_URL)

    assert "Authorization" in session.calls[0][1]
    assert all(call[1]["Authorization"] is None for call in session.calls[1:])


def test_requests_session_defaults_are_suppressed_without_mutation(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    class Adapter(requests.adapters.BaseAdapter):
        def __init__(self):
            self.prepared = []

        def send(self, request, **_kwargs):
            self.prepared.append(request)
            response = requests.Response()
            response.status_code = 200
            response.url = request.url
            response.raw = Raw(b'{"ok":true}')
            return response

        def close(self):
            pass

    adapter = Adapter()
    session = requests.Session()
    session.trust_env = False
    session.headers["Authorization"] = "caller-owned"
    session.auth = requests.auth.HTTPBasicAuth("user", "caller-password")
    session.mount("https://", adapter)

    client = MetadataClient(session)
    client.json(MODEL_URL)
    client.json("https://other.example/data.json")

    assert adapter.prepared[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert "Authorization" not in adapter.prepared[1].headers
    assert session.headers["Authorization"] == "caller-owned"
    assert session.auth.password == "caller-password"


@pytest.mark.parametrize("privacy", [True, None, "missing"])
def test_authenticated_resolve_rejects_private_or_unverified_repositories(monkeypatch, privacy):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    metadata = {"sha": "a" * 40, "siblings": [{"rfilename": "config.json"}]}
    if privacy != "missing":
        metadata["private"] = privacy
    session = Session(Response(json.dumps(metadata).encode()))
    client = MetadataClient(session)

    with pytest.raises(ArtifactError, match="Private or unverified") as error:
        client.resolve(parse_reference("org/model"))

    assert TOKEN not in str(error.value)
    assert len(session.calls) == 1
    assert client.repository_files == {}
    assert client.repository_metadata == {}


def test_public_gated_repository_can_resolve_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    metadata = {
        "sha": "a" * 40,
        "private": False,
        "gated": "manual",
        "siblings": [{"rfilename": "config.json"}],
    }
    session = Session(Response(json.dumps(metadata).encode()))
    client = MetadataClient(session)

    assert client.resolve(parse_reference("org/model")) == "a" * 40
    assert client.repository_files[("org/model", "a" * 40)] == {"config.json"}
    assert len(session.calls) == 1


def test_invalid_token_fails_without_exposing_its_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake\nsecret")

    with pytest.raises(ArtifactError, match="Invalid Hugging Face token configuration") as error:
        MetadataClient(Session(Response()))

    assert "hf_fake" not in str(error.value)


def test_local_launcher_excludes_hf_token_from_frontend():
    from pathlib import Path

    launcher = (Path(__file__).resolve().parents[2] / "scripts" / "start-local.ps1").read_text()
    secret_list = launcher.split("$frontendSecretNames =", 1)[1].split("\n", 1)[0]
    assert "'HF_TOKEN'" in secret_list


def test_compose_exposes_optional_hf_token_only_to_worker():
    from pathlib import Path

    compose = (Path(__file__).resolve().parents[2] / "compose.yaml").read_text()
    shared, worker = compose.split("  worker:", 1)
    worker, _frontend = worker.split("  frontend:", 1)
    assert 'HF_TOKEN: ""' in shared
    assert "HF_TOKEN: ${HF_TOKEN:-}" in worker
