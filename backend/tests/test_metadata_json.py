"""Strict JSON handling for bounded Hugging Face metadata reads."""

import io

import pytest

from model_anatomy.src.artifacts import MetadataClient
from model_anatomy.src.errors import ArtifactError


class FakeResponse:
    status_code = 200

    def __init__(self, body: bytes):
        self.raw = FakeRaw(body)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None


class FakeRaw(io.BytesIO):
    def read(self, size=-1, decode_content=False):
        return super().read(size)


class FakeTransport:
    def __init__(self, body: bytes):
        self.response = FakeResponse(body)

    def get(self, *_args, **_kwargs):
        return self.response


@pytest.mark.parametrize("number", [b"Infinity", b"-Infinity", b"NaN", b"1e309"])
def test_json_rejects_nonfinite_numbers_as_invalid_metadata(number):
    client = MetadataClient(session=FakeTransport(b'{"value":' + number + b"}"))

    with pytest.raises(ArtifactError, match="Invalid JSON metadata"):
        client.json("https://huggingface.co/config.json")

    assert client.audit == []


def test_json_accepts_finite_nested_floats_and_records_audit():
    client = MetadataClient(session=FakeTransport(b'{"value":1.25,"nested":[-0.5,2e100]}'))

    assert client.json("https://huggingface.co/config.json") == {
        "value": 1.25,
        "nested": [-0.5, 2e100],
    }
    assert client.audit == [
        {
            "url": "https://huggingface.co/config.json",
            "bytes_read": len(b'{"value":1.25,"nested":[-0.5,2e100]}'),
            "status": 200,
        }
    ]
