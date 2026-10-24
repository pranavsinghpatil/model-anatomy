"""Isolate native Transformers catalogue access and implementation provenance."""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .errors import UnsupportedModelError


class NativeResolutionError(UnsupportedModelError):
    """A native capability is unavailable without executing publisher code."""

    def __init__(self, code: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        details.setdefault("provider_versions", provider_versions())
        self.diagnostic: dict[str, Any] = {"status": "blocked", "code": code, "stage": stage,
                           "message": message, **details}


def provider_versions() -> dict[str, str | None]:
    result = {}
    for name in ("transformers", "torch", "safetensors"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def config_class(model_type: str, path: str = "config") -> type:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if model_type not in CONFIG_MAPPING:
        raise NativeResolutionError(
            "native_config_missing", "config_resolution",
            f"No native Transformers config for model_type={model_type} at {path}; "
            f"installed transformers={provider_versions()['transformers']}; publisher code remains disabled",
            model_type=model_type, config_path=path, provider_versions=provider_versions(),
        )
    try:
        implementation = CONFIG_MAPPING[model_type]
    except ImportError as exc:
        raise NativeResolutionError("native_dependency_missing", "config_resolution", str(exc),
                                    model_type=model_type, config_path=path) from exc
    if not implementation.__module__.startswith("transformers."):
        raise NativeResolutionError("custom_config_required", "config_resolution",
                                    f"Config implementation is not native: {model_type}")
    return implementation


def model_candidates(config: Any) -> list[tuple[str, type, str]]:
    """Inspect the installed provider's task mappings without importing every model."""
    from transformers.models.auto import modeling_auto

    model_type = getattr(config, "model_type", None)
    candidates = []
    for name, auto_class in vars(modeling_auto).items():
        if not name.startswith("AutoModel") or not isinstance(auto_class, type):
            continue
        mapping = auto_class.__dict__.get("_model_mapping")
        names = getattr(mapping, "_model_mapping", None)
        if not isinstance(names, dict):
            continue
        declared = names.get(model_type)
        if isinstance(declared, str):
            declared = [declared]
        if isinstance(declared, (list, tuple)) and all(isinstance(item, str) for item in declared):
            candidates.extend((name, auto_class, item) for item in declared)
    return candidates


def implementation_evidence(model: Any) -> dict[str, Any]:
    implementation = type(model)
    try:
        source_hash = hashlib.sha256(inspect.getsource(implementation).encode()).hexdigest()
    except (OSError, TypeError):
        source_hash = None
    config = model.config
    declarations = getattr(config, "auto_map", None)
    return {
        "status": "constructed", "stage": "meta_construction", "provider_versions": provider_versions(),
        "implementation": f"{implementation.__module__}.{implementation.__qualname__}",
        "implementation_source_sha256": source_hash,
        "publisher_auto_map": declarations if isinstance(declarations, dict) else None,
        "interpretation": "installed_native_implementation",
        "publisher_implementation_equivalence": "not_established" if declarations else "not_claimed",
        "remote_code_executed": False, "forward_executed": False, "weight_payload_bytes_read": 0,
    }


def require_supported_provider() -> None:
    """Reject stale no-sync environments before they can read or populate release caches."""
    from .version import NATIVE_TRANSFORMERS_VERSION

    actual = provider_versions()["transformers"]
    if actual != NATIVE_TRANSFORMERS_VERSION:
        raise NativeResolutionError(
            "provider_version_mismatch", "provider_runtime",
            f"This engine requires transformers=={NATIVE_TRANSFORMERS_VERSION}; installed={actual or 'missing'}. "
            "Synchronize the engine extra with uv sync --locked --extra engine (include --extra api for the worker).",
            required_transformers=NATIVE_TRANSFORMERS_VERSION, installed_transformers=actual,
        )
