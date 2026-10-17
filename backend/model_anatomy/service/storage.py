"""Private immutable metadata objects, addressed by SHA-256, never weight payloads."""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx

from .errors import ServiceError


def validate_digest(digest: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid content digest")


class Storage(Protocol):
    def put(self, data: bytes) -> str: ...
    def get(self, digest: str) -> bytes: ...
    def delete(self, digest: str) -> None: ...
    def check(self) -> None: ...


class LocalStorage:
    def __init__(self, root: Path, max_bytes: int = 32 * 1024**2):
        self.root, self.max_bytes = root.resolve(), max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, digest: str) -> Path:
        validate_digest(digest)
        return self.root / digest[:2] / f"{digest}.json"

    def check(self) -> None:
        with tempfile.TemporaryFile(dir=self.root) as handle:
            handle.write(b"check")

    def put(self, data: bytes) -> str:
        if len(data) > self.max_bytes:
            raise ServiceError(503, "artifact_too_large", "Metadata exceeds storage limit")
        digest = hashlib.sha256(data).hexdigest()
        path = self.path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(name, path)
            except FileExistsError:
                if self.get(digest) != data:
                    raise ServiceError(
                        503, "artifact_corrupt", "Existing object disagrees with content digest"
                    ) from None
            if os.name == "posix":
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            Path(name).unlink(missing_ok=True)
        return digest

    def get(self, digest: str) -> bytes:
        try:
            with self.path(digest).open("rb") as handle:
                data = handle.read(self.max_bytes + 1)
        except FileNotFoundError as exc:
            raise ServiceError(503, "artifact_unavailable", "Stored metadata is unavailable") from exc
        if len(data) > self.max_bytes or hashlib.sha256(data).hexdigest() != digest:
            raise ServiceError(503, "artifact_corrupt", "Stored metadata failed integrity verification")
        return data

    def delete(self, digest: str) -> None:
        self.path(digest).unlink(missing_ok=True)


class SupabaseStorage:
    def __init__(self, url: str, key: str, bucket: str, max_bytes: int, client=None):
        self.url, self.bucket, self.max_bytes = url.rstrip("/"), bucket, max_bytes
        self.client = client or httpx.Client(timeout=30, follow_redirects=False)
        self.headers = {"Authorization": f"Bearer {key}", "apikey": key}

    def request(self, method: str, url: str, **kwargs):
        try:
            return self.client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise ServiceError(503, "storage_unavailable", "Private storage is unavailable") from exc

    def check(self) -> None:
        response = self.request(
            "GET", f"{self.url}/storage/v1/bucket/{quote(self.bucket, safe='')}", headers=self.headers
        )
        try:
            private = response.status_code == 200 and response.json().get("public") is False
        except (ValueError, AttributeError):
            private = False
        if not private:
            raise ServiceError(503, "private_storage_required", "Storage bucket must exist and be private")

    def object_url(self, digest: str) -> str:
        validate_digest(digest)
        return f"{self.url}/storage/v1/object/{quote(self.bucket, safe='')}/{digest[:2]}/{digest}.json"

    def put(self, data: bytes) -> str:
        if len(data) > self.max_bytes:
            raise ServiceError(503, "artifact_too_large", "Metadata exceeds storage limit")
        digest = hashlib.sha256(data).hexdigest()
        response = self.request(
            "POST",
            self.object_url(digest),
            content=data,
            headers={**self.headers, "Content-Type": "application/json", "x-upsert": "false"},
        )
        # Verify existing content rather than accepting arbitrary conflict responses.
        if response.status_code not in {200, 201} and (
            response.status_code not in {400, 409} or self.get(digest) != data
        ):
            raise ServiceError(503, "storage_unavailable", "Private storage write failed")
        return digest

    def get(self, digest: str) -> bytes:
        try:
            with self.client.stream("GET", self.object_url(digest), headers=self.headers) as response:
                if response.status_code != 200:
                    raise ServiceError(503, "artifact_unavailable", "Private metadata is unavailable")
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > self.max_bytes:
                        raise ServiceError(503, "artifact_too_large", "Metadata exceeds storage limit")
        except httpx.HTTPError as exc:
            raise ServiceError(503, "storage_unavailable", "Private storage is unavailable") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise ServiceError(503, "artifact_corrupt", "Stored metadata failed integrity verification")
        return bytes(data)

    def delete(self, digest: str) -> None:
        response = self.request("DELETE", self.object_url(digest), headers=self.headers)
        if response.status_code not in {200, 204, 404}:
            raise ServiceError(503, "storage_unavailable", "Private metadata deletion failed")


def storage_for(settings) -> Storage:
    if settings.storage_backend == "supabase":
        return SupabaseStorage(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
            settings.supabase_bucket,
            settings.max_blob_bytes,
        )
    return LocalStorage(settings.data_dir / "objects", settings.max_blob_bytes)
