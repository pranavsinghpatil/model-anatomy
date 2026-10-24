"""Oversized checkpoint verification retains only bounded aggregate evidence."""

import json
import struct

import pytest

from model_anatomy.src import artifacts
from model_anatomy.src.artifacts import MetadataClient, ModelReference
from model_anatomy.src.errors import ArtifactError

REVISION = "a" * 40


def shard_bytes(names, *, bad_offset=False, dtype="F32"):
    entries = {}
    cursor = 0
    for name in names:
        end = cursor + artifacts.DTYPE_BYTES[dtype]
        entries[name] = {"dtype": dtype, "shape": [1], "data_offsets": [cursor, end]}
        cursor = end
    if bad_offset:
        entries[names[0]]["data_offsets"] = [0, 5]
    header = json.dumps(entries).encode()
    return struct.pack("<Q", len(header)) + header + bytes(cursor)


class IndexedClient(MetadataClient):
    def __init__(self, mapping, shards):
        super().__init__()
        self.mapping = mapping
        self.shards = shards
        self.reads = []
        self.reference = ModelReference("example/model")
        self.repository_files[(self.reference.model_id, REVISION)] = {
            "model.safetensors.index.json", *shards
        }

    def json(self, url, max_bytes=1024 * 1024):
        assert max_bytes == 16 * 1024 * 1024
        return {"weight_map": self.mapping}

    def range(self, url, start, end):
        name = url.rsplit("/", 1)[-1]
        body = self.shards[name]
        self.reads.append((name, start, end))
        # A header-only verifier must never request a payload byte.
        header_end = 8 + struct.unpack("<Q", body[:8])[0]
        assert end < header_end
        data = body[start : end + 1]
        self.checkpoint_header_bytes_read += len(data)
        return data, len(body)


def test_verifies_oversized_index_without_retaining_tensor_records(monkeypatch):
    monkeypatch.setattr(artifacts.limits, "MAX_CHECKPOINT_TENSORS", 1)
    client = IndexedClient(
        {"a": "one.safetensors", "b": "two.safetensors"},
        {"one.safetensors": shard_bytes(["a"]), "two.safetensors": shard_bytes(["b"])},
    )
    result = client.verify_indexed_checkpoint_headers(client.reference, REVISION)
    assert result.tensor_count == 2
    assert result.tensor_element_count == 2
    assert result.shard_count == 2
    assert result.weight_payload_bytes_read == 0
    assert result.header_bytes_read == sum(end - start + 1 for _, start, end in client.reads)
    assert set(result.shards) == {"one.safetensors", "two.safetensors"}
    assert all("tensors" not in item for item in result.shards.values())


def test_float8_e8m0_scale_uses_one_byte_per_element():
    client = IndexedClient(
        {"layers.0.attn.wkv.scale": "one.safetensors"},
        {"one.safetensors": shard_bytes(["layers.0.attn.wkv.scale"], dtype="F8_E8M0")},
    )
    result = client.verify_indexed_checkpoint_headers(client.reference, REVISION)
    assert result.tensor_count == 1
    assert result.shards["one.safetensors"]["file_size"] == (
        result.shards["one.safetensors"]["header_length"] + 9
    )


@pytest.mark.parametrize(
    ("mapping", "shard_names", "message"),
    [
        ({"a": "one.safetensors", "b": "two.safetensors"},
         {"one.safetensors": ["a"], "two.safetensors": []}, "omit indexed"),
        ({"a": "one.safetensors", "b": "two.safetensors"},
         {"one.safetensors": ["a"], "two.safetensors": ["a", "b"]}, "Duplicate or unindexed"),
        ({"a": "two.safetensors", "b": "one.safetensors"},
         {"one.safetensors": ["a"], "two.safetensors": ["b"]}, "disagree with the index"),
        ({"a": "one.safetensors"},
         {"one.safetensors": ["a", "extra"]}, "Duplicate or unindexed"),
    ],
)
def test_rejects_missing_duplicate_misplaced_and_unindexed_tensors(mapping, shard_names, message):
    client = IndexedClient(mapping, {name: shard_bytes(names) for name, names in shard_names.items()})
    with pytest.raises(ArtifactError, match=message):
        client.verify_indexed_checkpoint_headers(client.reference, REVISION)


def test_rejects_malformed_tensor_descriptor():
    client = IndexedClient(
        {"a": "one.safetensors"}, {"one.safetensors": shard_bytes(["a"], bad_offset=True)}
    )
    with pytest.raises(ArtifactError, match="byte offsets"):
        client.verify_indexed_checkpoint_headers(client.reference, REVISION)


def test_rejects_missing_shard_before_any_range_request():
    client = IndexedClient({"a": "missing.safetensors"}, {"one.safetensors": shard_bytes(["a"])})
    with pytest.raises(ArtifactError, match="invalid or missing shard"):
        client.verify_indexed_checkpoint_headers(client.reference, REVISION)
    assert client.reads == []


def test_rejects_explicit_single_file_selection():
    client = IndexedClient({"a": "one.safetensors"}, {"one.safetensors": shard_bytes(["a"])})
    reference = ModelReference("example/model", REVISION, "one.safetensors", True)
    with pytest.raises(ArtifactError, match="indexed checkpoint selection"):
        client.verify_indexed_checkpoint_headers(reference, REVISION)
    assert client.reads == []


def test_total_header_budget_stops_before_later_shard_body(monkeypatch):
    first = shard_bytes(["a"])
    monkeypatch.setattr(artifacts, "MAX_TOTAL_HEADER_BYTES", len(first))
    client = IndexedClient(
        {"a": "one.safetensors", "b": "two.safetensors"},
        {"one.safetensors": first, "two.safetensors": shard_bytes(["b"])},
    )
    with pytest.raises(ArtifactError, match="total inspection budget"):
        client.verify_indexed_checkpoint_headers(client.reference, REVISION)
    assert ("two.safetensors", 0, 7) in client.reads
    assert not any(name == "two.safetensors" and start == 8 for name, start, _ in client.reads)
