"""Bounded GGUF header inspection and explicit checkpoint selection."""

from __future__ import annotations

import hashlib
import os
import string
import struct

import pytest

from model_anatomy.src.artifacts import InspectionBudgetError, MetadataClient, parse_reference
from model_anatomy.src.errors import ArtifactError
from model_anatomy.src.gguf import parse_gguf_header
from model_anatomy.src.pipeline import reconstruct

SHA = "b" * 40
PINNED_REVISION = "b072e1d3b35a0a630cece372c2127528e0994386"
MODEL_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_FILE = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
GGUF_FILES = {
    "Ternary-Bonsai-2-27B-F16.gguf",
    "Ternary-Bonsai-2-27B-PQ2_0.gguf",
    MODEL_FILE,
    "Ternary-Bonsai-2-27B-mmproj-BF16.gguf",
    "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
}


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def make_gguf(
    *,
    tokens: int = 0,
    duplicate_key: bool = False,
    tensor_type: int = 4095,
    split_metadata: bool | dict[str, object] = False,
    tensor_count: int = 1,
    minimal_tensors: bool = False,
    extra_metadata: int = 0,
):
    metadata = [
        gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("qwen2"),
    ]
    metadata.extend(
        gguf_string(f"benchmark.entry.{index}") + struct.pack("<II", 4, index)
        for index in range(extra_metadata)
    )
    if tokens:
        token_values = b"".join(gguf_string(f"token-{index}") for index in range(tokens))
        metadata.append(
            gguf_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, tokens) + token_values
        )
    if duplicate_key:
        metadata.append(gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("other"))
    if split_metadata:
        split_values = (
            split_metadata
            if isinstance(split_metadata, dict)
            else {"split.no": 1, "split.count": 4, "split.tensors.count": 851}
        )
        for key, value in split_values.items():
            if type(value) is bool:
                encoded = struct.pack("<I?", 7, value)
            elif isinstance(value, str):
                encoded = struct.pack("<I", 8) + gguf_string(value)
            else:
                encoded = struct.pack("<Ii", 5, value)
            metadata.append(gguf_string(key) + encoded)
    if minimal_tensors:
        names = (string.ascii_letters + string.digits)[:tensor_count]
        assert len(names) == tensor_count
        descriptors = b"".join(
            gguf_string(name) + struct.pack("<IQIQ", 1, 1, tensor_type, 0) for name in names
        )
    else:
        descriptors = gguf_string("blk.0.attn_q.weight") + struct.pack("<IQQIQ", 2, 4096, 32, tensor_type, 0)
    header = struct.pack("<4sIQQ", b"GGUF", 3, tensor_count, len(metadata)) + b"".join(metadata) + descriptors
    data_start = (len(header) + 31) & ~31
    data = header + bytes(data_start - len(header)) + bytes(512)
    return data, data_start


def test_gguf_header_reads_do_not_cross_into_tensor_data():
    data, data_start = make_gguf(tokens=100_000)
    ranges = []

    def read_range(start, end):
        ranges.append((start, end))
        return data[start : end + 1], len(data)

    parsed = parse_gguf_header(read_range, len(data))

    tensor = parsed.tensors["blk.0.attn_q.weight"]
    assert tensor.dtype == "GGML_TYPE_4095"
    assert tensor.tensor_type_id == 4095
    assert tensor.upstream_type_interpretation is None
    assert tensor.shape == (4096, 32)
    assert tensor.public_dict()["shape_convention"] == "ggml_dimensions"
    assert tensor.absolute_data_offset == data_start
    assert parsed.metadata == {"general.architecture": "qwen2"}
    assert parsed.tensor_count == 1
    assert parsed.metadata_entry_count == 2
    assert len(ranges) < 30
    assert all(end < data_start for _, end in ranges)
    assert parsed.header_sha256 == hashlib.sha256(data[: parsed.header_length]).hexdigest()


def test_minimum_descriptors_do_not_make_structural_prefetch_cross_payload():
    data, data_start = make_gguf(tensor_count=40, minimal_tensors=True)
    ranges = []

    def read_range(start, end):
        ranges.append((start, end))
        return data[start : end + 1], len(data)

    parsed = parse_gguf_header(read_range, len(data))

    assert parsed.tensor_count == 40
    assert len(parsed.tensors) == 40
    assert all(end < data_start for _, end in ranges)


def test_large_vocabulary_with_many_metadata_and_descriptors_has_bounded_requests():
    data, data_start = make_gguf(
        tokens=100_000,
        extra_metadata=47,
        tensor_count=40,
        minimal_tensors=True,
    )
    ranges = []

    def read_range(start, end):
        ranges.append((start, end))
        return data[start : end + 1], len(data)

    parsed = parse_gguf_header(read_range, len(data))

    assert parsed.metadata_entry_count == 49
    assert parsed.tensor_count == 40
    assert len(parsed.tensors) == 40
    assert len(ranges) < 30
    assert all(end < data_start for _, end in ranges)


def test_duplicate_gguf_metadata_keys_are_rejected():
    data, _ = make_gguf(duplicate_key=True)

    with pytest.raises(ArtifactError, match="keys must be non-empty and unique"):
        parse_gguf_header(lambda start, end: (data[start : end + 1], len(data)), len(data))


def test_header_byte_budget_is_not_reported_as_corruption(monkeypatch):
    data, _ = make_gguf()
    monkeypatch.setattr("model_anatomy.src.gguf.GGUF_MAX_HEADER_BYTES", 32)
    with pytest.raises(InspectionBudgetError, match="byte budget"):
        parse_gguf_header(lambda start, end: (data[start : end + 1], len(data)), len(data))


def test_unknown_array_element_type_is_reported_as_artifact_error():
    data = (
        struct.pack("<4sIQQ", b"GGUF", 3, 0, 1)
        + gguf_string("metadata.array")
        + struct.pack("<IIQ", 9, 0xFFFF, 0)
    )

    with pytest.raises(ArtifactError, match="array has an unsupported value type"):
        parse_gguf_header(lambda start, end: (data[start : end + 1], len(data)), len(data))


def test_upstream_dtype_name_is_only_an_interpretation():
    data, _ = make_gguf(tensor_type=0)
    parsed = parse_gguf_header(lambda start, end: (data[start : end + 1], len(data)), len(data))
    tensor = parsed.tensors["blk.0.attn_q.weight"].public_dict()

    assert tensor["dtype"] == "GGML_TYPE_0"
    assert tensor["tensor_type_id"] == 0
    assert tensor["upstream_type_interpretation"] == "F32"
    assert tensor["type_mapping_source"] == "gguf==0.19.0"
    assert tensor["type_mapping_scope"] == "upstream_interpretation_only"


@pytest.mark.parametrize(
    ("split_values", "message"),
    [
        ({"split.no": "0", "split.count": 4, "split.tensors.count": 851}, "must be an integer"),
        ({"split.no": True, "split.count": 4, "split.tensors.count": 851}, "must be an integer"),
        ({"split.no": -1, "split.count": 4, "split.tensors.count": 851}, "non-negative"),
        ({"split.no": 4, "split.count": 4, "split.tensors.count": 851}, "less than split.count"),
        ({"split.no": 0, "split.count": 0, "split.tensors.count": 851}, "must be positive"),
        (
            {"split.no": 0, "split.count": 4, "split.tensors.count": 0},
            "smaller than this file's tensor count",
        ),
    ],
)
def test_invalid_split_metadata_is_rejected(split_values, message):
    data, _ = make_gguf(split_metadata=split_values)

    with pytest.raises(ArtifactError, match=message):
        parse_gguf_header(lambda start, end: (data[start : end + 1], len(data)), len(data))


class HeaderResponse:
    status_code = 206
    headers: dict[str, str]

    def __init__(self, body: bytes, start: int, end: int, size: int):
        self.raw = self
        self.body = body
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
            "Content-Encoding": "identity",
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def raise_for_status(self):
        return None

    def read(self, size=-1, decode_content=False):
        return self.body if size < 0 else self.body[:size]


class HeaderSession:
    def __init__(self, data: bytes):
        self.data = data

    def get(self, url, *, headers, **kwargs):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return HeaderResponse(self.data[start : end + 1], start, end, len(self.data))


class PinnedGGUFClient(MetadataClient):
    def __init__(self, data: bytes):
        super().__init__(HeaderSession(data))

    def resolve(self, reference):
        self.repository_files[(reference.model_id, SHA)] = set(GGUF_FILES)
        return SHA


def test_repository_with_five_gguf_files_requires_selection():
    data, _ = make_gguf()
    result = reconstruct(
        "prism-ml/Ternary-Bonsai-2-27B-gguf",
        client=PinnedGGUFClient(data),
        revision=SHA,
        allow_artifact_inspection=True,
    )

    assert result.metadata["inspection"]["reason_code"] == "checkpoint_selection_required"
    assert result.metadata["inspection"]["checkpoint_available"] is True
    assert result.metadata["inspection"]["selected_checkpoint_files"] == []
    assert result.tensors == {}
    assert result.root_graph.nodes == []


def test_explicit_pinned_ptq1_file_returns_header_only_metadata():
    data, data_start = make_gguf(tensor_type=4095, split_metadata=True)
    value = f"https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/{SHA}/{MODEL_FILE}"
    client = PinnedGGUFClient(data)
    result = reconstruct(value, client=client, allow_artifact_inspection=True)

    tensor = result.tensors["blk.0.attn_q.weight"]
    ranges = [item for item in result.metadata["artifacts"]["requests"] if "range" in item]
    assert result.metadata["inspection"]["reason_code"] == "gguf_header_only"
    assert result.metadata["revision"] == SHA
    assert result.metadata["inspection"]["selected_checkpoint_files"] == [MODEL_FILE]
    assert result.metadata["artifacts"]["checkpoint_format"] == "gguf"
    assert "checkpoint_tensor_element_count" not in result.metadata
    assert result.metadata["artifacts"]["selected_file_tensor_count"] == 1
    assert result.metadata["artifacts"]["selected_file_tensor_element_count"] == 4096 * 32
    assert result.metadata["artifacts"]["selected_file_scope"] == "single_selected_gguf_file"
    assert result.metadata["artifacts"]["selected_file_split_metadata"] == {
        "split.no": 1,
        "split.count": 4,
        "split.tensors.count": 851,
    }
    assert result.metadata["artifacts"]["checkpoint_header_bytes_read"] == sum(
        item["bytes_read"] for item in ranges
    )
    assert result.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert "safetensors_bytes_read" not in result.metadata["artifacts"]
    assert tensor["format"] == "gguf"
    assert tensor["shape_convention"] == "ggml_dimensions"
    assert tensor["tensor_type_id"] == 4095
    assert tensor["dtype"] == "GGML_TYPE_4095"
    assert tensor["upstream_type_interpretation"] is None
    assert tensor["type_mapping_scope"] == "upstream_interpretation_only"
    assert result.metadata["artifacts"]["shards"][MODEL_FILE]["metadata"]["split.count"] == 4
    assert all(item["format"] == "gguf" and item["range"][1] < data_start for item in ranges)
    assert not any(item.source == "safetensors_header" for item in result.evidence)


def test_reference_parser_accepts_explicit_gguf_file_only():
    reference = parse_reference(f"https://huggingface.co/org/model/blob/{SHA}/weights.gguf")
    assert reference.explicit_filename
    assert reference.filename == "weights.gguf"
    nested = parse_reference(f"https://huggingface.co/org/model/resolve/{SHA}/nested%20dir/%E2%98%83.gguf")
    assert nested.filename == "nested dir/\u2603.gguf"
    assert nested.url(nested.filename).endswith("/nested%20dir/%E2%98%83.gguf")
    with pytest.raises(ArtifactError):
        parse_reference("https://huggingface.co/org/model/blob/main/weights.bin")
    for path in ("%2E%2E/weights.gguf", "nested%2Fescape.gguf", "nested%5Cescape.gguf"):
        with pytest.raises(ArtifactError, match="invalid file path"):
            parse_reference(f"https://huggingface.co/org/model/blob/{SHA}/{path}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests"
)
def test_real_pinned_bonsai_inventory_and_explicit_ptq1_header():
    inventory = reconstruct(MODEL_ID, revision=PINNED_REVISION, allow_artifact_inspection=True)
    assert inventory.metadata["revision"] == PINNED_REVISION
    assert inventory.metadata["inspection"]["reason_code"] == "checkpoint_selection_required"
    assert inventory.metadata["inspection"]["checkpoint_available"] is True
    assert inventory.tensors == {}
    assert inventory.root_graph.nodes == []

    url = f"https://huggingface.co/{MODEL_ID}/resolve/{PINNED_REVISION}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    inspected = reconstruct(url, allow_artifact_inspection=True)
    assert inspected.metadata["revision"] == PINNED_REVISION
    assert inspected.metadata["inspection"]["selected_checkpoint_files"] == [MODEL_FILE]
    assert inspected.metadata["inspection"]["reason_code"] == "gguf_header_only"
    assert inspected.tensors
    assert inspected.root_graph.nodes == [] and inspected.root_graph.edges == []
    assert inspected.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert "safetensors_bytes_read" not in inspected.metadata["artifacts"]
    assert not any(item.source == "safetensors_header" for item in inspected.evidence)
    shard = inspected.metadata["artifacts"]["shards"][MODEL_FILE]
    assert shard["format"] == "gguf"
    assert shard["tensor_count"] == 851
    assert shard["metadata_entry_count"] == 49
    assert shard["file_size"] == 5_946_648_928
    assert shard["header_hash_coverage"] == "complete_bytes_0_to_header_length_exclusive"
    ranges = [item for item in inspected.metadata["artifacts"]["requests"] if "range" in item]
    assert ranges and all(item["range"][1] < shard["data_start"] for item in ranges)
    assert all(item["format"] == "gguf" for item in ranges)
