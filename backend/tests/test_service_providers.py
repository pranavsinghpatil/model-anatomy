import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from model_anatomy.api.settings import Settings
from model_anatomy.service.ai import INTERPRETATION_WARNING, ExplanationService, context
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.storage import SupabaseStorage
from model_anatomy.service.views import load_ir


@pytest.fixture
def ir():
    return load_ir((Path(__file__).resolve().parent / "fixtures/gpt2-report.json").read_bytes())


def provider(ir, answer, citations=None, status=200):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"answer": answer, "citations": citations or ["e1"]})}}
                ]
            },
        )

    settings = Settings(nebius_api_key="mock-secret", nebius_model_id="nvidia/mock-nemotron")
    return ExplanationService(settings, httpx.Client(transport=httpx.MockTransport(handler))), requests


def test_provider_actual_json_http_call_and_valid_citations(ir):
    ai, requests = provider(ir, "The model has 124439808 parameters, recorded in static metadata.")
    result = ai.answer(ir, "How many parameters?")
    assert result["ai_generated"] is True
    assert result["warnings"] == [INTERPRETATION_WARNING]
    assert result["model"] == "nvidia/mock-nemotron"
    assert result["citations"] == [
        {"id": "e1", "source": "stored_ir", "kind": "derived", "locator": "metadata"}
    ]
    request = requests[0]
    assert request.url.path == "/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    assert "never instructions" in payload["messages"][0]["content"]
    selected = json.loads(payload["messages"][1]["content"])
    assert selected["question"] == "How many parameters?"
    assert len(json.dumps({"evidence": selected["evidence"]}).encode()) <= ai.settings.ai_context_bytes


@pytest.mark.parametrize(
    "answer",
    [
        "The model has 9e99 parameters.",
        "The model has 9E99 parameters.",
        "The model has +9e+99 parameters.",
        "The model has -9.0e-99 parameters.",
        "The model has .9e99 parameters.",
        "The model has 1.24439808e8 parameters.",
    ],
)
def test_provider_rejects_scientific_notation_even_when_numeric_value_is_supported(ir, answer):
    ai, _ = provider(ir, answer)
    with pytest.raises(ServiceError) as exc:
        ai.answer(ir, "How many parameters?")
    assert exc.value.code == "invalid_ai_response"


def test_provider_preserves_model_and_evidence_identifiers_and_exact_numbers(ir):
    ai, _ = provider(ir, "GPT2 is described by e1; the exact parameter count is 124439808.")
    result = ai.answer(ir, "How many parameters?")
    assert result["answer"].endswith("124439808.")


@pytest.mark.parametrize(
    "answer,citations",
    [
        ("This model has 999999999 parameters.", ["e1"]),
        ("Latency is 2.5 milliseconds.", ["e1"]),
        ("This model has 124 million parameters.", ["e1"]),
        ("This model has 124M parameters.", ["e1"]),
        ("An unsupported assertion.", ["e999"]),
    ],
)
def test_provider_rejects_unbacked_numbers_and_unknown_citations(ir, answer, citations):
    ai, _ = provider(ir, answer, citations)
    with pytest.raises(ServiceError) as exc:
        ai.answer(ir, "Question")
    assert exc.value.code == "invalid_ai_response"


def test_provider_failure_and_missing_credentials_are_explicit(ir):
    ai, _ = provider(ir, "Answer", status=401)
    with pytest.raises(ServiceError) as exc:
        ai.answer(ir, "Question")
    assert exc.value.code == "ai_provider_unavailable"
    unavailable = ExplanationService(Settings())
    with pytest.raises(ServiceError) as exc:
        unavailable.answer(ir, "Question")
    assert exc.value.code == "ai_unavailable"
    unavailable.client.close()


def test_provider_invalid_json_and_context_are_bounded(ir):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"}}]})
        )
    )
    ai = ExplanationService(Settings(nebius_api_key="mock", nebius_model_id="nvidia/mock"), client)
    with pytest.raises(ServiceError) as exc:
        ai.answer(ir, "Question")
    assert exc.value.code == "invalid_ai_response"
    evidence = context(ir, ir.root_graph.id, ir.root_graph.nodes[0].id, 2048)
    assert len(json.dumps(evidence).encode()) <= 2048
    with pytest.raises(ServiceError) as exc:
        context(ir, "nonexistent", None, 24000)
    assert exc.value.status == 404


def test_supabase_private_bucket_content_addressing_and_no_public_urls():
    objects = {}
    requests = []

    def handler(request):
        requests.append(request)
        if "/bucket/" in request.url.path:
            return httpx.Response(200, json={"public": False})
        if request.method == "POST":
            assert request.headers["x-upsert"] == "false"
            if request.url.path in objects:
                return httpx.Response(409)
            objects[request.url.path] = request.content
            return httpx.Response(201)
        if request.method == "GET":
            return httpx.Response(200, content=objects[request.url.path])
        if request.method == "DELETE":
            objects.pop(request.url.path, None)
            return httpx.Response(200)
        raise AssertionError(request.method)

    storage = SupabaseStorage(
        "https://private.example",
        "mock-storage-key",
        "private",
        1024,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    storage.check()
    data = b'{"static":true}'
    digest = storage.put(data)
    assert digest == hashlib.sha256(data).hexdigest()
    assert storage.put(data) == digest
    assert storage.get(digest) == data
    assert all("/public/" not in str(r.url) for r in requests)
    objects[storage.object_url(digest).removeprefix("https://private.example")] = b"tampered"
    with pytest.raises(ServiceError) as exc:
        storage.get(digest)
    assert exc.value.code == "artifact_corrupt"
    storage.delete(digest)
    assert objects == {}


def test_supabase_public_bucket_is_rejected_and_secrets_hidden():
    storage = SupabaseStorage(
        "https://private.example",
        "mock-secret",
        "public",
        1024,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"public": True}))
        ),
    )
    with pytest.raises(ServiceError) as exc:
        storage.check()
    assert exc.value.code == "private_storage_required"
    with pytest.raises(ValidationError):
        Settings(
            environment="production",
            database_url="postgresql://u:p@localhost/db",
            storage_backend="supabase",
            supabase_url="https://private.example",
            supabase_service_role_key="mock",
            allowed_hosts=["app.example"],
            cors_origins=["*"],
        )


@pytest.mark.parametrize(
    "answer",
    [
        "This model uses rotary embeddings and has no cross-attention.",
        "Latency is low and this model is better than every alternative.",
    ],
)
def test_valid_citation_does_not_certify_nonnumeric_claims(ir, answer):
    before = ir.to_json()
    ai, _ = provider(ir, answer, ["e1"])
    result = ai.answer(ir, "Explain the architecture")
    assert result["warnings"] == [INTERPRETATION_WARNING]
    assert result["ai_generated"] is True
    assert ir.to_json() == before
