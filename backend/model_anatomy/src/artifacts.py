"""Pinned Hugging Face metadata resolution with bounded, header-only reads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isfinite, prod
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

from . import limits
from .errors import ArtifactError

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_CHECKPOINT_SHARDS = 64
MAX_TOTAL_HEADER_BYTES = 64 * 1024 * 1024
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
        encoded_filename = "/".join(quote(part, safe="") for part in filename.split("/"))
        return (
            f"https://huggingface.co/{self.model_id}/resolve/"
            f"{quote(revision or self.revision, safe='')}/{encoded_filename}"
        )


def parse_reference(value: str) -> ModelReference:
    value = value.strip()
    revision, filename = "main", "model.safetensors"
    explicit_filename = False
    if "://" in value:
        url = urlsplit(value)
        if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ArtifactError("Expected an HTTPS Hugging Face model URL without query or fragment")
        parts = url.path.strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree" and re.fullmatch(r"[0-9a-f]{40}", parts[3]):
            revision = parts[3]
        elif len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            revision = unquote(parts[3])
            filename_parts = [unquote(part) for part in parts[4:]]
            if any(
                not part or part in {".", ".."} or "/" in part or "\\" in part or "\x00" in part
                for part in filename_parts
            ):
                raise ArtifactError("Checkpoint URL contains an invalid file path")
            filename = "/".join(filename_parts)
            explicit_filename = True
        elif len(parts) != 2:
            raise ArtifactError("Expected a model repository or a model.safetensors file URL")
        value = "/".join(parts[:2])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or any(p in {".", ".."} for p in value.split("/")):
        raise ArtifactError("Expected a namespace/model repository ID")
    if (
        not revision
        or revision in {".", ".."}
        or not filename.endswith((".safetensors", ".safetensors.index.json", ".gguf"))
    ):
        raise ArtifactError("Expected a Safetensors or GGUF file URL")
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
    tensors: dict[str, Any]
    shards: dict[str, Any]
    tensor_files: dict[str, str]
    index: dict[str, Any] | None = None
    format: str = "safetensors"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return [request for shard in self.shards.values() for request in shard.requests]


class AmbiguousCheckpointError(ArtifactError):
    """The repository has multiple checkpoint candidates without an authoritative index."""


class InspectionBudgetError(ArtifactError):
    """Well-formed checkpoint metadata exceeded a configured inspection budget."""

    def __init__(self, message: str, *, index_summary: dict[str, Any] | None = None):
        super().__init__(message)
        self.index_summary = index_summary


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_safetensors_header(header_bytes: bytes, file_size: int) -> SafetensorsMetadata:
    try:
        header = json.loads(header_bytes, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ArtifactError(f"Invalid Safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict) or not header_bytes.startswith(b"{"):
        raise ArtifactError("Safetensors header must be a JSON object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise ArtifactError("Safetensors file metadata must contain strings")
    tensors: dict[str, TensorMetadata] = {}
    tensor_budget_exceeded = False
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if len(tensors) >= limits.MAX_CHECKPOINT_TENSORS:
            tensor_budget_exceeded = True
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
        if not tensor_budget_exceeded:
            tensors[name] = TensorMetadata(name, dtype, tuple(shape), prod(shape), tuple(offsets))
        spans.append((offsets[0], offsets[1]))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ArtifactError("Safetensors payload contains overlapping tensors or holes")
        cursor = end
    if 8 + len(header_bytes) + cursor != file_size:
        raise ArtifactError("Safetensors header offsets disagree with file size")
    if tensor_budget_exceeded:
        raise InspectionBudgetError("Checkpoint exceeds the supported tensor record count")
    return SafetensorsMetadata(
        tensors, metadata, hashlib.sha256(header_bytes).hexdigest(), len(header_bytes), file_size, header
    )


class MetadataClient:
    """Never consume a weight-file body unless the exact requested range is honored."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        token: str | None = None,
    ):
        self.session = session or requests.Session()
        self.timeout = timeout
        selected_token = token if token is not None else os.environ.get("HF_TOKEN", "")
        if not isinstance(selected_token, str) or (
            selected_token.strip() and any(not 33 <= ord(char) <= 126 for char in selected_token)
        ):
            raise ArtifactError("Invalid Hugging Face token configuration")
        self._token = selected_token.strip()
        self.audit: list[dict[str, Any]] = []
        self.repository_files: dict[tuple[str, str], set[str]] = {}
        self.repository_metadata: dict[tuple[str, str], dict[str, str | None]] = {}
        self.header_bytes_read = 0
        self.checkpoint_header_bytes_read = 0

    @staticmethod
    def _trusted_hf_url(url: str) -> bool:
        if any(ord(char) < 33 for char in url):
            return False
        try:
            parsed = urlsplit(url)
            return (
                parsed.scheme == "https"
                and parsed.hostname == "huggingface.co"
                and parsed.port in (None, 443)
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            return False

    @contextmanager
    def _get(self, url: str, headers: dict[str, str]) -> Iterator[requests.Response]:
        current_url = url
        request_headers: dict[str, str | None] = dict(headers)
        if self._token and self._trusted_hf_url(current_url):
            request_headers["Authorization"] = f"Bearer {self._token}"
        else:
            # A caller-owned Session may have default auth headers; suppress them
            # per request without changing that Session's configuration.
            request_headers["Authorization"] = None
        for _ in range(6):
            with self.session.get(
                current_url,
                stream=True,
                timeout=self.timeout,
                headers=request_headers,
                auth=lambda prepared: prepared,
                allow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308} and response.headers.get("Location"):
                    next_url = urljoin(current_url, response.headers["Location"])
                    if not (
                        self._trusted_hf_url(current_url)
                        and self._trusted_hf_url(next_url)
                    ):
                        request_headers["Authorization"] = None
                    current_url = next_url
                    continue
                yield response
                return
        raise ArtifactError("Artifact request exceeded the redirect limit")

    def json(self, url: str, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
        with self._get(url, {"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            body = response.raw.read(max_bytes + 1, decode_content=True)
            if len(body) > max_bytes:
                raise ArtifactError("JSON metadata exceeds the size limit")
            try:
                result = json.loads(
                    body,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                    parse_float=_finite_float,
                )
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ArtifactError(f"Invalid JSON metadata: {url}") from exc
            if not isinstance(result, dict):
                raise ArtifactError("Expected a metadata object")
            self.audit.append({"url": url, "bytes_read": len(body), "status": response.status_code})
            return result

    def resolve(self, reference: ModelReference) -> str:
        info = self.json(
            f"https://huggingface.co/api/models/{reference.model_id}/revision/{quote(reference.revision, safe='')}"
        )
        if info.get("private") is True or (self._token and info.get("private") is not False):
            raise ArtifactError("Private or unverified Hugging Face repository is not supported")
        sha = info.get("sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ArtifactError("Hugging Face did not return an immutable commit SHA")
        siblings = info.get("siblings", [])
        if not isinstance(siblings, list):
            raise ArtifactError("Hugging Face repository file metadata is invalid")
        if len(siblings) > limits.MAX_REPOSITORY_FILES:
            raise ArtifactError("Repository exceeds the supported file record count")
        if not all(isinstance(item, dict) and isinstance(item.get("rfilename"), str) for item in siblings):
            raise ArtifactError("Hugging Face repository file metadata is invalid")
        files = {item["rfilename"] for item in siblings}
        self.repository_files[(reference.model_id, sha)] = files
        self.repository_metadata[(reference.model_id, sha)] = {
            key: info.get(key) if isinstance(info.get(key), str) else None
            for key in ("pipeline_tag", "library_name")
        }
        return sha

    def checkpoint(self, reference: ModelReference, revision: str) -> CheckpointMetadata:
        files = self.repository_files[(reference.model_id, revision)]
        if reference.explicit_filename and reference.filename.endswith(".gguf"):
            if reference.filename not in files:
                raise ArtifactError(f"Requested GGUF artifact is missing: {reference.filename}")
            return self.gguf_checkpoint(reference, revision, reference.filename)

        has_safetensors = any(name.endswith((".safetensors", ".safetensors.index.json")) for name in files)
        if not has_safetensors:
            gguf_files = sorted(name for name in files if "/" not in name and name.endswith(".gguf"))
            if len(gguf_files) == 1:
                return self.gguf_checkpoint(reference, revision, gguf_files[0])
            if len(gguf_files) > 1:
                raise AmbiguousCheckpointError("Cannot select an unambiguous GGUF checkpoint")
            raise ArtifactError("Repository has no supported checkpoint artifact")
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
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()):
                raise ArtifactError("Safetensors index has an invalid weight_map")
            directory = index_name.rpartition("/")[0]
            # Component indexes name shards relative to their own subfolder.
            if directory:
                if any(
                    value.startswith("/")
                    or any(part in {"", ".", ".."} for part in value.split("/"))
                    or "\\" in value
                    or "\x00" in value
                    for value in mapping.values()
                ):
                    raise ArtifactError("Safetensors index references invalid shard paths")
                mapping = {name: f"{directory}/{filename}" for name, filename in mapping.items()}
            shard_names = sorted(set(mapping.values()))
            if not shard_names or any(
                name not in files
                or not name.endswith(".safetensors")
                or name.startswith("/")
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or "\\" in name
                or "\x00" in name
                for name in shard_names
            ):
                raise ArtifactError("Safetensors index references invalid or missing shard files")
            if len(mapping) > limits.MAX_CHECKPOINT_TENSORS:
                raise InspectionBudgetError(
                    "Checkpoint index exceeds the supported tensor record count",
                    index_summary={
                        "url": reference.url(index_name, revision),
                        "canonical_sha256": hashlib.sha256(
                            json.dumps(index, sort_keys=True).encode()
                        ).hexdigest(),
                        "declared_tensor_names": len(mapping),
                        "declared_shards": len(shard_names),
                        "tensor_record_limit": limits.MAX_CHECKPOINT_TENSORS,
                    },
                )
        else:
            index = None
            filename = reference.filename
            if filename not in files:
                alternatives = sorted(n for n in files if "/" not in n and n.endswith(".safetensors"))
                if len(alternatives) != 1:
                    raise AmbiguousCheckpointError("Cannot select an unambiguous Safetensors checkpoint")
                filename = alternatives[0]
            shard_names = [filename]
            mapping = None
        tensors, shards, tensor_files = {}, {}, {}
        if len(shard_names) > MAX_CHECKPOINT_SHARDS:
            raise InspectionBudgetError("Checkpoint exceeds the supported shard count")
        for filename in shard_names:
            if len(tensors) >= limits.MAX_CHECKPOINT_TENSORS:
                raise InspectionBudgetError("Checkpoint exceeds the supported tensor record count")
            shard = self.safetensors(reference.url(filename, revision))
            shards[filename] = shard
            for name, tensor in shard.tensors.items():
                if name in tensors:
                    raise ArtifactError(f"Duplicate tensor across Safetensors shards: {name}")
                if len(tensors) >= limits.MAX_CHECKPOINT_TENSORS:
                    raise InspectionBudgetError("Checkpoint exceeds the supported tensor record count")
                tensors[name], tensor_files[name] = tensor, filename
        if mapping is not None and mapping != tensor_files:
            raise ArtifactError("Safetensors shard headers disagree with the index weight_map")
        return CheckpointMetadata(tensors, shards, tensor_files, index)

    def gguf_checkpoint(self, reference: ModelReference, revision: str, filename: str) -> CheckpointMetadata:
        from .gguf import parse_gguf_header

        audit_start = len(self.audit)
        url = reference.url(filename, revision)
        _, file_size = self.range(url, 0, 23)
        self.audit[-1]["format"] = "gguf"

        def read_range(start: int, end: int) -> tuple[bytes, int]:
            result = self.range(url, start, end)
            self.audit[-1]["format"] = "gguf"
            return result

        shard = parse_gguf_header(read_range, file_size)
        for item in self.audit[audit_start:]:
            if "range" in item:
                item["format"] = "gguf"
        shard.requests = self.audit[audit_start:]
        tensors = shard.tensors
        return CheckpointMetadata(
            tensors,
            {filename: shard},
            {name: filename for name in tensors},
            format="gguf",
        )

    def range(self, url: str, start: int, end: int) -> tuple[bytes, int]:
        with self._get(url, {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as response:
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
            self.checkpoint_header_bytes_read += len(body)
            return body, int(match[3])

    def safetensors(self, url: str) -> SafetensorsMetadata:
        audit_start = len(self.audit)
        prefix, file_size = self.range(url, 0, 7)
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or 8 + length > file_size:
            raise ArtifactError(f"Invalid Safetensors header length: {length}")
        if length > MAX_HEADER_BYTES:
            raise InspectionBudgetError("Safetensors header length is excessive for the inspection budget")
        if self.header_bytes_read + length + 8 > MAX_TOTAL_HEADER_BYTES:
            raise InspectionBudgetError("Checkpoint metadata exceeds the total inspection budget")
        body, second_size = self.range(url, 8, 8 + length - 1)
        self.header_bytes_read += length + 8
        if second_size != file_size:
            raise ArtifactError("Safetensors file changed between Range requests")
        result = parse_safetensors_header(body, file_size)
        result.requests = self.audit[audit_start:]
        return result
