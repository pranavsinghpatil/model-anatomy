"""Resolve a Hugging Face model reference without loading model weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi

_HUB_HOSTS = {"huggingface.co", "www.huggingface.co"}
_REVISION_URL_PREFIXES = {"tree", "resolve", "blob"}
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


class ModelReferenceError(ValueError):
    """Raised when an input does not identify a supported Hub model."""


class ModelResolutionError(RuntimeError):
    """Raised when the Hub cannot provide a stable model inventory."""


class HubClient(Protocol):
    """Small interface used by the resolver and its deterministic tests."""

    def model_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> Any:
        """Return metadata for a model repository."""


@dataclass(frozen=True)
class ModelReference:
    """A normalized Hub model ID and an optional user-selected revision."""

    model_id: str
    revision: str | None = None


@dataclass(frozen=True)
class Artifact:
    """Metadata that can be obtained without downloading the artifact."""

    path: str
    size_bytes: int | None
    is_weight: bool


@dataclass(frozen=True)
class ResolvedModel:
    """A revision-pinned inventory for the next reconstruction stage."""

    requested_reference: str
    model_id: str
    requested_revision: str | None
    resolved_revision: str
    artifacts: tuple[Artifact, ...]
    config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a frontend-safe, JSON-compatible representation."""

        return {
            "requested_reference": self.requested_reference,
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "artifacts": [
                {
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes,
                    "is_weight": artifact.is_weight,
                }
                for artifact in self.artifacts
            ],
            "config": dict(self.config),
        }


def normalize_model_reference(
    reference: str, *, revision: str | None = None
) -> ModelReference:
    """Normalize a Hub model ID or model URL.

    Supported URLs identify the repository directly or use the ``tree``,
    ``resolve``, or ``blob`` form. The repository revision in a URL is honored
    unless the caller supplies a conflicting revision explicitly.
    """

    if not isinstance(reference, str) or not reference.strip():
        raise ModelReferenceError("A non-empty Hugging Face model ID or URL is required.")

    reference = reference.strip()
    requested_revision = _normalize_revision(revision)

    if "://" not in reference:
        return ModelReference(
            model_id=_normalize_model_id(reference), revision=requested_revision
        )

    parsed = urlparse(reference)
    if parsed.scheme != "https" or parsed.hostname not in _HUB_HOSTS:
        raise ModelReferenceError(
            "Expected an https://huggingface.co/<namespace>/<model> URL."
        )
    if parsed.query or parsed.fragment:
        raise ModelReferenceError("Model URLs must not contain query parameters or fragments.")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ModelReferenceError("A model URL must include namespace and model name.")

    model_id = _normalize_model_id("/".join(parts[:2]))
    url_revision: str | None = None
    remainder = parts[2:]
    if remainder:
        if len(remainder) < 2 or remainder[0] not in _REVISION_URL_PREFIXES:
            raise ModelReferenceError(
                "Supported model URLs end at the repository or use /tree/, /resolve/, or /blob/."
            )
        url_revision = _normalize_revision(remainder[1])

    if requested_revision and url_revision and requested_revision != url_revision:
        raise ModelReferenceError(
            "The explicit revision conflicts with the revision encoded in the URL."
        )
    return ModelReference(model_id=model_id, revision=requested_revision or url_revision)


class ModelResolver:
    """Resolve and pin model metadata before any model or weight loading."""

    def __init__(self, hub_client: HubClient | None = None) -> None:
        self._hub_client = hub_client or HfApi()

    def resolve(
        self, reference: str, *, revision: str | None = None
    ) -> ResolvedModel:
        """Return a stable artifact inventory for a Hugging Face model.

        This method deliberately requests metadata only. Weight download and
        module instantiation belong to later extraction stages.
        """

        normalized = normalize_model_reference(reference, revision=revision)
        try:
            info = self._hub_client.model_info(
                normalized.model_id,
                revision=normalized.revision,
                files_metadata=True,
            )
        except Exception as exc:  # The Hub client exposes several transport errors.
            raise ModelResolutionError(
                f"Could not resolve Hugging Face model '{normalized.model_id}'."
            ) from exc

        resolved_revision = _required_string(getattr(info, "sha", None), "commit SHA")
        artifacts = tuple(
            sorted(
                (_artifact_from_sibling(sibling) for sibling in getattr(info, "siblings", []) or []),
                key=lambda artifact: artifact.path,
            )
        )
        config = getattr(info, "config", None)
        return ResolvedModel(
            requested_reference=reference,
            model_id=normalized.model_id,
            requested_revision=normalized.revision,
            resolved_revision=resolved_revision,
            artifacts=artifacts,
            config=dict(config) if isinstance(config, Mapping) else {},
        )


def _normalize_model_id(model_id: str) -> str:
    parts = model_id.strip().split("/")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise ModelReferenceError(
            "Expected a model ID in the form '<namespace>/<model>'."
        )
    return "/".join(part.strip() for part in parts)


def _normalize_revision(revision: str | None) -> str | None:
    if revision is None:
        return None
    if not isinstance(revision, str) or not revision.strip():
        raise ModelReferenceError("A supplied revision must be a non-empty string.")
    return revision.strip()


def _artifact_from_sibling(sibling: Any) -> Artifact:
    path = _required_string(_field(sibling, "rfilename"), "artifact filename")
    size = _field(sibling, "size")
    if not isinstance(size, int):
        lfs = _field(sibling, "lfs")
        size = _field(lfs, "size")
    return Artifact(
        path=path,
        size_bytes=size if isinstance(size, int) else None,
        is_weight=path.lower().endswith(_WEIGHT_SUFFIXES),
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelResolutionError(f"Hub response did not include a {description}.")
    return value
