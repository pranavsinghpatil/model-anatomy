"""Describe repository declarations separately from verified architecture facts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .artifacts import ModelReference
from .errors import ArtifactError


def config_filename(reference: ModelReference) -> str:
    directory = reference.filename.rpartition("/")[0] if reference.explicit_filename else ""
    return f"{directory}/config.json" if directory else "config.json"


def native_dimensions(config: Any) -> dict[str, Any]:
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None)
    depths = getattr(config, "depths", None)
    if (
        getattr(config, "model_type", None) in {"swin", "swinv2", "donut-swin", "maskformer-swin"}
        and isinstance(depths, list)
        and all(type(depth) is int and depth > 0 for depth in depths)
    ):
        layers = sum(depths)
    return {
        "hidden": getattr(config, "hidden_size", None) or getattr(config, "d_model", None),
        "heads": getattr(config, "num_attention_heads", None) or getattr(config, "num_heads", None),
        "layers": layers,
        "vocabulary": getattr(config, "vocab_size", None),
    }


def repository_profile(
    reference: ModelReference,
    files: set[str],
    config: dict[str, Any] | None,
    repository: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    formats = set()
    for filename in files:
        for suffix, format_name in (
            (".safetensors", "safetensors"),
            (".gguf", "gguf"),
            (".bin", "pytorch_binary"),
            (".onnx", "onnx"),
            (".h5", "hdf5"),
            (".msgpack", "flax"),
            (".tflite", "tflite"),
        ):
            if filename.endswith(suffix):
                formats.add(format_name)
    component_files: dict[str, list[str]] = {}
    for filename in sorted(files):
        directory, separator, _ = filename.partition("/")
        if separator and filename.endswith((".safetensors", ".safetensors.index.json", ".gguf")):
            component_files.setdefault(directory, []).append(filename)
    components = []
    if manifest is not None:
        for path, declaration in sorted(manifest.items()):
            if path.startswith("_") or not isinstance(declaration, list):
                continue
            if (
                not path
                or path in {".", ".."}
                or any(char in path for char in "/\\\x00")
                or len(declaration) != 2
                or any(item is not None and not isinstance(item, str) for item in declaration)
            ):
                raise ArtifactError("Invalid pipeline component declaration in model_index.json")
            components.append(
                {
                    "path": path,
                    "library": declaration[0],
                    "declared_class": declaration[1],
                    "checkpoint_files": component_files.get(path, []),
                }
            )
    config = config or {}
    architectures = config.get("architectures")
    directory = reference.filename.rpartition("/")[0] if reference.explicit_filename else ""
    adapter_path = f"{directory}/adapter_config.json" if directory else "adapter_config.json"
    kind = (
        "diffusers_pipeline"
        if manifest is not None
        else "adapter"
        if adapter_path in files
        else "gguf_collection"
        if "gguf" in formats and "safetensors" not in formats
        else "transformers"
        if isinstance(config.get("model_type"), str)
        else "artifact_repository"
    )
    return {
        "repository_kind": kind,
        "declared_task": repository.get("pipeline_tag"),
        "declared_library": repository.get("library_name"),
        "config_model_type": config.get("model_type") if isinstance(config.get("model_type"), str) else None,
        "declared_architectures": architectures
        if isinstance(architectures, list) and all(isinstance(item, str) for item in architectures)
        else [],
        "formats": sorted(formats),
        "selected_file": reference.filename if reference.explicit_filename else None,
        "selected_config": config_filename(reference) if config_filename(reference) in files else None,
        "components": components,
        "architecture_blockers": [],
        "manifest_canonical_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if manifest is not None
        else None,
        "metadata_basis": "Publisher declarations and pinned file paths; not verified computation or modality semantics",
    }
