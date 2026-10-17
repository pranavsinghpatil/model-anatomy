"""Bounded GGUF header parsing without reading tensor payloads."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any

from . import limits
from .errors import ArtifactError

GGUF_MAX_HEADER_BYTES = 16 * 1024 * 1024
GGUF_MAX_TOTAL_HEADER_BYTES = 64 * 1024 * 1024
GGUF_MAX_RANGE_REQUESTS = 10_000
GGUF_MAX_METADATA_ENTRIES = 100_000
GGUF_MAX_ARRAY_VALUES = 1_000_000
GGUF_MAX_STRING_BYTES = 1024 * 1024
GGUF_MAX_DIMENSIONS = 4
GGUF_READ_CHUNK_BYTES = 1024 * 1024
GGUF_DEFAULT_ALIGNMENT = 32

_VALUE_FORMATS = {
    0: ("B", "uint8"),
    1: ("b", "int8"),
    2: ("H", "uint16"),
    3: ("h", "int16"),
    4: ("I", "uint32"),
    5: ("i", "int32"),
    6: ("f", "float32"),
    7: ("?", "bool"),
    10: ("Q", "uint64"),
    11: ("q", "int64"),
    12: ("d", "float64"),
}


@dataclass(frozen=True)
class GGUFTensorMetadata:
    name: str
    dtype: str
    tensor_type_id: int
    upstream_type_interpretation: str | None
    shape: tuple[int, ...]
    parameter_count: int
    data_offset: int
    absolute_data_offset: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "tensor_type_id": self.tensor_type_id,
            "upstream_type_interpretation": self.upstream_type_interpretation,
            "type_mapping_source": "gguf==0.19.0",
            "type_mapping_scope": "upstream_interpretation_only",
            "shape": list(self.shape),
            "shape_convention": "ggml_dimensions",
            "parameter_count": self.parameter_count,
            "data_offset": self.data_offset,
            "absolute_data_offset": self.absolute_data_offset,
        }


@dataclass
class GGUFMetadata:
    tensors: dict[str, GGUFTensorMetadata]
    metadata: dict[str, Any]
    tensor_count: int
    metadata_entry_count: int
    header_sha256: str
    header_length: int
    file_size: int
    data_start: int
    requests: list[dict[str, Any]]


class _HeaderReader:
    def __init__(self, read_range: Callable[[int, int], tuple[bytes, int]], file_size: int):
        self.read_range = read_range
        self.file_size = file_size
        self.position = 0
        self.bytes_read = 0
        self.request_count = 0
        self.header_bytes = bytearray()
        self.buffer = b""
        self.buffer_start = 0
        self.fetched_ranges: list[tuple[int, int]] = []
        self.minimum_tail_bytes = 0

    def _fetch(self, start: int, length: int) -> bytes:
        if length < 0 or self.bytes_read + length > GGUF_MAX_TOTAL_HEADER_BYTES:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF headers exceed the total inspection budget")
        if self.request_count >= GGUF_MAX_RANGE_REQUESTS:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF header requires too many Range requests")
        if start + length > self.file_size:
            raise ArtifactError("GGUF header is truncated")
        if start + length > GGUF_MAX_HEADER_BYTES:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF header exceeds the inspection byte budget")
        body, observed_size = self.read_range(start, start + length - 1)
        if observed_size != self.file_size:
            raise ArtifactError("GGUF file changed between Range requests")
        self.request_count += 1
        self.bytes_read += len(body)
        self.fetched_ranges.append((start, start + len(body) - 1))
        return body

    def read(self, length: int) -> bytes:
        if length == 0:
            return b""
        if self.buffer and self.position >= self.buffer_start:
            offset = self.position - self.buffer_start
            if offset < len(self.buffer):
                count = min(length, len(self.buffer) - offset)
                data = self.buffer[offset : offset + count]
                self.header_bytes.extend(data)
                self.position += count
                if count == length:
                    return data
                return data + self.read(length - count)
        if length > GGUF_MAX_HEADER_BYTES - self.position:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF header exceeds the inspection byte budget")
        fetch_length = max(length, min(GGUF_READ_CHUNK_BYTES, self.minimum_tail_bytes))
        self.buffer_start = self.position
        self.buffer = self._fetch(self.position, fetch_length)
        return self.read(length)

    def unpack(self, format_string: str) -> tuple[Any, ...]:
        size = struct.calcsize(format_string)
        try:
            return struct.unpack(format_string, self.read(size))
        except struct.error as exc:
            raise ArtifactError("Malformed GGUF header field") from exc

    def string(self) -> str:
        length = self.unpack("<Q")[0]
        if length > GGUF_MAX_STRING_BYTES:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF string exceeds the supported inspection budget")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactError("GGUF header contains an invalid UTF-8 string") from exc

    def skip_string(self) -> None:
        length = self.unpack("<Q")[0]
        if length > GGUF_MAX_STRING_BYTES:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF string exceeds the supported inspection budget")
        self.read(length)

    def prefetch_minimum_records(self, remaining: int, minimum_bytes: int) -> None:
        if remaining <= 0:
            return
        if self.buffer and self.position < self.buffer_start + len(self.buffer):
            return
        length = min(
            GGUF_READ_CHUNK_BYTES,
            remaining * minimum_bytes + self.minimum_tail_bytes,
        )
        self.buffer_start = self.position
        self.buffer = self._fetch(self.position, length)


def parse_gguf_header(read_range: Callable[[int, int], tuple[bytes, int]], file_size: int) -> GGUFMetadata:
    """Read only GGUF header fields, with every remote range ending in the header."""
    try:
        from gguf.constants import GGMLQuantizationType
    except ImportError as exc:
        raise ArtifactError("GGUF inspection requires the optional gguf engine dependency") from exc

    reader = _HeaderReader(read_range, file_size)
    magic, version, tensor_count, metadata_count = reader.unpack("<4sIQQ")
    if magic != b"GGUF" or version not in {2, 3}:
        raise ArtifactError("Unsupported or invalid GGUF magic/version")
    if tensor_count > limits.MAX_CHECKPOINT_TENSORS:
        from .artifacts import InspectionBudgetError

        raise InspectionBudgetError("GGUF exceeds the supported tensor record count")
    if metadata_count > GGUF_MAX_METADATA_ENTRIES:
        from .artifacts import InspectionBudgetError

        raise InspectionBudgetError("GGUF exceeds the supported metadata entry count")

    metadata: dict[str, Any] = {}
    seen_metadata_keys: set[str] = set()
    for index in range(metadata_count):
        # Future KV records need at least 13 bytes each; tensor descriptors need
        # at least 33. Keep this suffix budget available to bounded read refills.
        reader.minimum_tail_bytes = (metadata_count - index - 1) * 13 + tensor_count * 33
        reader.prefetch_minimum_records(1, 13)
        key = reader.string()
        if not key or key in seen_metadata_keys:
            raise ArtifactError("GGUF metadata keys must be non-empty and unique")
        seen_metadata_keys.add(key)
        value_type = reader.unpack("<I")[0]
        retain = key in {
            "general.architecture",
            "general.name",
            "general.file_type",
            "general.quantization_version",
            "general.alignment",
            "split.no",
            "split.count",
            "split.tensors.count",
        } or key.endswith(
            (
                ".block_count",
                ".embedding_length",
                ".context_length",
                ".feed_forward_length",
                ".attention.head_count",
                ".attention.head_count_kv",
            )
        )
        value = _read_value(reader, value_type, retain=retain)
        if retain and value is not None:
            metadata[key] = value

    alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if type(alignment) is not int or alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
        raise ArtifactError("GGUF has an invalid general.alignment value")
    _validate_split_metadata(metadata, tensor_count)

    tensors: dict[str, GGUFTensorMetadata] = {}
    seen_names: set[str] = set()
    for index in range(tensor_count):
        # Reserve 32 bytes for the current descriptor and 33 for each later
        # one; valid descriptors are at least 33 bytes, leaving a byte of slack.
        reader.minimum_tail_bytes = (tensor_count - index - 1) * 33
        reader.prefetch_minimum_records(1, 32)
        name = reader.string()
        if not name or name in seen_names:
            raise ArtifactError("GGUF tensor names must be non-empty and unique")
        seen_names.add(name)
        dimension_count = reader.unpack("<I")[0]
        if not 1 <= dimension_count <= GGUF_MAX_DIMENSIONS:
            raise ArtifactError(f"GGUF tensor has an unsupported dimension count: {name}")
        dimensions = reader.unpack("<" + "Q" * dimension_count)
        if any(dimension == 0 for dimension in dimensions):
            raise ArtifactError(f"GGUF tensor has an empty dimension: {name}")
        tensor_type_id, data_offset = reader.unpack("<IQ")
        if data_offset >= file_size:
            raise ArtifactError(f"GGUF tensor offset is outside the file: {name}")
        # Data section start is computed after all descriptors below. Keep raw
        # GGML dimension order because GGUF tensor axes differ from PyTorch order.
        try:
            upstream_type_interpretation = GGMLQuantizationType(tensor_type_id).name
        except ValueError:
            upstream_type_interpretation = None
        tensors[name] = GGUFTensorMetadata(
            name,
            f"GGML_TYPE_{tensor_type_id}",
            tensor_type_id,
            upstream_type_interpretation,
            tuple(dimensions),
            prod(dimensions),
            data_offset,
            data_offset,
        )

    data_start = (reader.position + alignment - 1) & ~(alignment - 1)
    if data_start > file_size:
        raise ArtifactError("GGUF data section starts outside the file")
    if any(end >= data_start for _, end in reader.fetched_ranges):
        raise ArtifactError("GGUF header inspection range reached the tensor data section")
    if any(tensor.data_offset >= file_size - data_start for tensor in tensors.values()):
        raise ArtifactError("GGUF tensor offset points outside the payload area")
    tensors = {
        name: GGUFTensorMetadata(
            tensor.name,
            tensor.dtype,
            tensor.tensor_type_id,
            tensor.upstream_type_interpretation,
            tensor.shape,
            tensor.parameter_count,
            tensor.data_offset,
            data_start + tensor.data_offset,
        )
        for name, tensor in tensors.items()
    }
    return GGUFMetadata(
        tensors,
        metadata,
        tensor_count,
        metadata_count,
        hashlib.sha256(reader.header_bytes).hexdigest(),
        reader.position,
        file_size,
        data_start,
        [],
    )


def _validate_split_metadata(metadata: dict[str, Any], tensor_count: int) -> None:
    split_keys = ("split.no", "split.count", "split.tensors.count")
    for key in split_keys:
        if key in metadata and type(metadata[key]) is not int:
            raise ArtifactError(f"GGUF {key} metadata must be an integer")
    split_no = metadata.get("split.no")
    split_count = metadata.get("split.count")
    split_tensor_count = metadata.get("split.tensors.count")
    if split_no is not None and split_no < 0:
        raise ArtifactError("GGUF split.no metadata must be non-negative")
    if split_count is not None and split_count < 1:
        raise ArtifactError("GGUF split.count metadata must be positive")
    if split_no is not None and split_count is not None and split_no >= split_count:
        raise ArtifactError("GGUF split.no must be less than split.count")
    if split_tensor_count is not None and split_tensor_count < tensor_count:
        raise ArtifactError("GGUF split.tensors.count is smaller than this file's tensor count")


def _read_value(reader: _HeaderReader, value_type: int, *, retain: bool) -> Any:
    from gguf.constants import GGUFValueType

    try:
        kind = GGUFValueType(value_type)
    except ValueError as exc:
        raise ArtifactError(f"GGUF metadata has an unsupported value type: {value_type}") from exc
    if kind == GGUFValueType.STRING:
        return reader.string()
    if kind == GGUFValueType.ARRAY:
        element_type = reader.unpack("<I")[0]
        count = reader.unpack("<Q")[0]
        if count > GGUF_MAX_ARRAY_VALUES:
            from .artifacts import InspectionBudgetError

            raise InspectionBudgetError("GGUF metadata array exceeds the supported value count")
        try:
            element_kind = GGUFValueType(element_type)
        except ValueError as exc:
            raise ArtifactError(f"GGUF metadata array has an unsupported value type: {element_type}") from exc
        if element_kind == GGUFValueType.STRING:
            values = [] if retain and count <= 4096 else None
            tail_after_array = reader.minimum_tail_bytes
            for index in range(count):
                reader.minimum_tail_bytes = tail_after_array + (count - index - 1) * 8
                reader.prefetch_minimum_records(1, 8)
                if values is None:
                    reader.skip_string()
                else:
                    values.append(reader.string())
            reader.minimum_tail_bytes = tail_after_array
            return values
        if element_type not in _VALUE_FORMATS:
            raise ArtifactError("GGUF metadata array has an unsupported value type")
        code, _ = _VALUE_FORMATS[element_type]
        element_size = struct.calcsize("<" + code)
        raw = reader.read(count * element_size)
        if not retain or count > 4096:
            return None
        return [item[0] for item in struct.iter_unpack("<" + code, raw)]
    if value_type not in _VALUE_FORMATS:
        raise ArtifactError(f"GGUF metadata has an unsupported value type: {value_type}")
    code, _ = _VALUE_FORMATS[value_type]
    return reader.unpack("<" + code)[0]
