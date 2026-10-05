"""Pinned Hugging Face metadata resolution with bounded, header-only reads."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from math import prod
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import requests

from .errors import ArtifactError

MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


@dataclass(frozen=True)
class ModelReference:
    model_id: str
    revision: str = "main"
    filename: str = "model.safetensors"
    explicit_filename: bool = False

    def url(self, filename: str, revision: str | None = None) -> str:
        return f"https://huggingface.co/{self.model_id}/resolve/{quote(revision or self.revision, safe='')}/{filename}"


def parse_reference(value: str) -> ModelReference:
    value = value.strip()
    revision, filename = "main", "model.safetensors"
    explicit_filename = False
    if "://" in value:
        url = urlsplit(value)
        if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ArtifactError("Expected an HTTPS Hugging Face model URL without query or fragment")
        parts = url.path.strip("/").split("/")
        if len(parts) == 5 and parts[2] in {"blob", "resolve"}:
            revision, filename = unquote(parts[3]), parts[4]
            explicit_filename = True
        elif len(parts) != 2:
            raise ArtifactError("Expected a model repository or a model.safetensors file URL")
        value = "/".join(parts[:2])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or any(p in {".", ".."} for p in value.split("/")):
        raise ArtifactError("Expected a namespace/model repository ID")
    if (
        not revision
        or revision in {".", ".."}
        or not (filename.endswith((".safetensors", ".safetensors.index.json")))
    ):
        raise ArtifactError("Expected a Safetensors file or index reference")
    return ModelReference(value, revision, filename, explicit_filename)


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    parameter_count: int
    data_offsets: tuple[int, int]

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "parameter_count": self.parameter_count,
        }


@dataclass
class SafetensorsMetadata:
    tensors: dict[str, TensorMetadata]
    metadata: dict[str, str]
    header_sha256: str
    header_length: int
    file_size: int
    header: dict[str, Any]
    requests: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CheckpointMetadata:
    tensors: dict[str, TensorMetadata]
    shards: dict[str, SafetensorsMetadata]
    tensor_files: dict[str, str]
    index: dict[str, Any] | None = None

    @property
    def requests(self) -> list[dict[str, Any]]:
        return [request for shard in self.shards.values() for request in shard.requests]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_safetensors_header(header_bytes: bytes, file_size: int) -> SafetensorsMetadata:
    try:
        header = json.loads(header_bytes, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise ArtifactError(f"Invalid Safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict) or not header_bytes.startswith(b"{"):
        raise ArtifactError("Safetensors header must be a JSON object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise ArtifactError("Safetensors file metadata must contain strings")
    tensors: dict[str, TensorMetadata] = {}
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ArtifactError(f"Invalid tensor entry: {name}")
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ArtifactError(f"Unsupported Safetensors dtype for {name}: {dtype}")
        if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
            raise ArtifactError(f"Invalid tensor shape: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(d) is int and d >= 0 for d in offsets)
            or offsets[1] - offsets[0] != prod(shape) * DTYPE_BYTES[dtype]
        ):
            raise ArtifactError(f"Invalid tensor byte offsets: {name}")
        tensors[name] = TensorMetadata(name, dtype, tuple(shape), prod(shape), tuple(offsets))
        spans.append((offsets[0], offsets[1]))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ArtifactError("Safetensors payload contains overlapping tensors or holes")
        cursor = end
    if 8 + len(header_bytes) + cursor != file_size:
        raise ArtifactError("Safetensors header offsets disagree with file size")
    return SafetensorsMetadata(
        tensors, metadata, hashlib.sha256(header_bytes).hexdigest(), len(header_bytes), file_size, header
    )


class MetadataClient:
    """Never consume a weight-file body unless the exact requested range is honored."""

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.audit: list[dict[str, Any]] = []
        self.repository_files: dict[tuple[str, str], set[str]] = {}

    def json(self, url: str, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
        with self.session.get(
            url, stream=True, timeout=self.timeout, headers={"Accept-Encoding": "identity"}
        ) as response:
            response.raise_for_status()
            body = response.raw.read(max_bytes + 1, decode_content=True)
            if len(body) > max_bytes:
                raise ArtifactError("JSON metadata exceeds the size limit")
            try:
                result = json.loads(body, object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError) as exc:
                raise ArtifactError(f"Invalid JSON metadata: {url}") from exc
            if not isinstance(result, dict):
                raise ArtifactError("Expected a metadata object")
            self.audit.append({"url": url, "bytes_read": len(body), "status": response.status_code})
            return result

    def resolve(self, reference: ModelReference) -> str:
        info = self.json(
            f"https://huggingface.co/api/models/{reference.model_id}/revision/{quote(reference.revision, safe='')}"
        )
        sha = info.get("sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ArtifactError("Hugging Face did not return an immutable commit SHA")
        siblings = info.get("siblings", [])
        if not isinstance(siblings, list) or not all(
            isinstance(item, dict) and isinstance(item.get("rfilename"), str) for item in siblings
        ):
            raise ArtifactError("Hugging Face repository file metadata is invalid")
        files = {item["rfilename"] for item in siblings}
        if not any(name.endswith((".safetensors", ".safetensors.index.json")) for name in files):
            raise ArtifactError("Repository has no Safetensors checkpoint; it is outside the target scope")
        if "config.json" not in files:
            raise ArtifactError("Repository has no config.json")
        self.repository_files[(reference.model_id, sha)] = files
        return sha

    def checkpoint(self, reference: ModelReference, revision: str) -> CheckpointMetadata:
        files = self.repository_files[(reference.model_id, revision)]
        index_name = "model.safetensors.index.json"
        if reference.filename.endswith(".index.json"):
            index_name = reference.filename
        if reference.explicit_filename and reference.filename not in files:
            raise ArtifactError(f"Requested Safetensors artifact is missing: {reference.filename}")
        if index_name in files and (
            not reference.explicit_filename or reference.filename.endswith(".index.json")
        ):
            index = self.json(reference.url(index_name, revision), max_bytes=16 * 1024 * 1024)
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ArtifactError("Safetensors index has no weight_map")
            shard_names = (
                sorted(set(mapping.values())) if all(isinstance(v, str) for v in mapping.values()) else []
            )
            if not shard_names or any(
                name not in files
                or not name.endswith(".safetensors")
                or name.startswith("/")
                or ".." in name.split("/")
                for name in shard_names
            ):
                raise ArtifactError("Safetensors index references invalid or missing shard files")
        else:
            index = None
            filename = reference.filename
            if filename not in files:
                alternatives = sorted(n for n in files if "/" not in n and n.endswith(".safetensors"))
                if len(alternatives) != 1:
                    raise ArtifactError("Cannot select an unambiguous Safetensors checkpoint")
                filename = alternatives[0]
            shard_names = [filename]
            mapping = None
        tensors, shards, tensor_files = {}, {}, {}
        for filename in shard_names:
            shard = self.safetensors(reference.url(filename, revision))
            if shard.metadata.get("format") not in {None, "pt"}:
                raise ArtifactError("Safetensors checkpoint is not in the PyTorch target scope")
            shards[filename] = shard
            for name, tensor in shard.tensors.items():
                if name in tensors:
                    raise ArtifactError(f"Duplicate tensor across Safetensors shards: {name}")
                tensors[name], tensor_files[name] = tensor, filename
        if mapping is not None and mapping != tensor_files:
            raise ArtifactError("Safetensors shard headers disagree with the index weight_map")
        return CheckpointMetadata(tensors, shards, tensor_files, index)

    def range(self, url: str, start: int, end: int) -> tuple[bytes, int]:
        with self.session.get(
            url,
            stream=True,
            timeout=self.timeout,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            expected = end - start + 1
            if (
                response.status_code != 206
                or match is None
                or int(match[1]) != start
                or int(match[2]) != end
                or int(match[3]) <= end
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or response.headers.get("Content-Length", str(expected)) != str(expected)
            ):
                raise ArtifactError(
                    f"Server did not honor exact Range bytes={start}-{end}; body was not read"
                )
            body = response.raw.read(expected)
            if len(body) != expected:
                raise ArtifactError("Truncated Safetensors range response")
            self.audit.append(
                {
                    "url": url,
                    "range": [start, end],
                    "status": 206,
                    "content_range": content_range,
                    "bytes_read": len(body),
                }
            )
            return body, int(match[3])

    def safetensors(self, url: str) -> SafetensorsMetadata:
        audit_start = len(self.audit)
        prefix, file_size = self.range(url, 0, 7)
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or length > MAX_HEADER_BYTES or 8 + length > file_size:
            raise ArtifactError(f"Invalid or excessive Safetensors header length: {length}")
        body, second_size = self.range(url, 8, 8 + length - 1)
        if second_size != file_size:
            raise ArtifactError("Safetensors file changed between Range requests")
        result = parse_safetensors_header(body, file_size)
        result.requests = self.audit[audit_start:]
        return result
