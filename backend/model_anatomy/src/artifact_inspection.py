"""Build an honest inventory report when native architecture reconstruction is unavailable."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .artifacts import CheckpointMetadata, ModelReference
from .ir import Evidence, Graph, ModelIR
from .version import ENGINE_VERSION

INSPECTION_MESSAGES = {
    "native_architecture_unavailable": (
        "The installed native Transformers implementation cannot construct this architecture. "
        "Only verified artifact inventory is available; no architecture graph was inferred."
    ),
    "adapter_requires_base_model": (
        "This repository contains adapter artifacts without an inspected base-model architecture. "
        "Only verified artifact inventory is available; no architecture graph was inferred."
    ),
    "architecture_config_unavailable": (
        "This repository has no architecture configuration. Only verified artifact inventory is "
        "available; no architecture graph was inferred."
    ),
    "safetensors_unavailable": (
        "This repository has no Safetensors checkpoint. Only the pinned repository file inventory "
        "is available; no architecture graph was inferred."
    ),
    "checkpoint_selection_required": (
        "This repository contains multiple checkpoint components without a unique selection. "
        "Only the pinned repository file inventory is available; no tensors or graph were inferred."
    ),
    "metadata_budget_exceeded": (
        "Checkpoint metadata exceeds the configured inspection budget. The pinned repository and "
        "available configuration are shown without tensor or architecture claims."
    ),
    "gguf_header_only": (
        "GGUF tensor metadata was read from the pinned file header only. No architecture graph or "
        "tensor payload was inferred."
    ),
    "pipeline_components_require_selection": (
        "This repository declares a multi-component pipeline. Select a component checkpoint to "
        "inspect its tensors or supported native architecture. Pipeline execution connections "
        "have not been reconstructed."
    ),
    "checkpoint_framework_unavailable": (
        "Safetensors headers were verified, but their declared framework is not supported by the "
        "native PyTorch architecture engine. No architecture graph or parameter ownership was inferred."
    ),
}


def has_adapter_tensors(checkpoint: CheckpointMetadata | None) -> bool:
    return bool(
        checkpoint and any("lora_" in name.lower() or "lora." in name.lower() for name in checkpoint.tensors)
    )


def build_artifact_inspection(
    reference: ModelReference,
    revision: str,
    files: set[str],
    audit: list[dict[str, Any]],
    *,
    audit_start: int,
    config_data: dict[str, Any] | None,
    checkpoint: CheckpointMetadata | None,
    reason_code: str,
    model_profile: dict[str, Any] | None = None,
) -> ModelIR:
    """Return only pinned file and validated Safetensors-header facts."""
    message = INSPECTION_MESSAGES[reason_code]
    shard_audit = {
        filename: {
            "url": reference.url(filename, revision),
            "header_sha256": shard.header_sha256,
            "header_length": shard.header_length,
            "file_size": shard.file_size,
            "format": checkpoint.format,
            **(
                {
                    "data_start": shard.data_start,
                    "alignment_padding_bytes": shard.data_start - shard.header_length,
                    "header_hash_coverage": "complete_bytes_0_to_header_length_exclusive",
                    "header_validation_scope": (
                        "structure_and_tensor_descriptors; skipped metadata strings are not decoded"
                    ),
                    "tensor_count": shard.tensor_count,
                    "metadata_entry_count": shard.metadata_entry_count,
                    "metadata": shard.metadata,
                }
                if checkpoint.format == "gguf"
                else {}
            ),
        }
        for filename, shard in (checkpoint.shards.items() if checkpoint else [])
    }
    selected_checkpoint_files = set(checkpoint.tensor_files.values()) if checkpoint else set()
    if checkpoint and checkpoint.index is not None:
        index_name = (
            reference.filename
            if reference.explicit_filename and reference.filename.endswith(".index.json")
            else "model.safetensors.index.json"
        )
        if index_name in files:
            selected_checkpoint_files.add(index_name)
    tensors = {}
    if checkpoint:
        for name, tensor in sorted(checkpoint.tensors.items()):
            filename = checkpoint.tensor_files[name]
            tensors[name] = {
                **tensor.public_dict(),
                "artifact": reference.url(filename, revision),
                "role": "unclassified",
                "evidence_kind": "direct",
                "format": checkpoint.format,
            }

    artifacts = {
        "repository_files": sorted(files),
        "shards": shard_audit,
        "sharded": bool(checkpoint and checkpoint.index is not None),
        "checkpoint_header_bytes_read": sum(
            item["bytes_read"] for item in audit[audit_start:] if "range" in item
        ),
        "requests": audit[audit_start:],
    }
    has_gguf_ranges = any(item.get("format") == "gguf" for item in audit[audit_start:])
    if checkpoint is None and has_gguf_ranges:
        # A malformed declaration can make bounded prefetch cross the real
        # boundary. Do not claim zero payload reads without a validated header.
        artifacts["payload_read_status"] = "not_established_header_validation_failed"
    else:
        artifacts["weight_payload_bytes_read"] = 0
    if (
        checkpoint is not None
        and checkpoint.format == "safetensors"
        or checkpoint is None
        and not has_gguf_ranges
    ):
        artifacts["safetensors_bytes_read"] = artifacts["checkpoint_header_bytes_read"]
    if checkpoint is not None:
        artifacts["checkpoint_format"] = checkpoint.format
    if checkpoint is not None and checkpoint.format == "gguf":
        selected_shard = next(iter(checkpoint.shards.values()))
        artifacts["selected_file_tensor_count"] = len(checkpoint.tensors)
        artifacts["selected_file_tensor_element_count"] = sum(
            item.parameter_count for item in checkpoint.tensors.values()
        )
        artifacts["selected_file_scope"] = "single_selected_gguf_file"
        split = {
            key: selected_shard.metadata[key]
            for key in ("split.no", "split.count", "split.tensors.count")
            if key in selected_shard.metadata
        }
        if split:
            artifacts["selected_file_split_metadata"] = split
    if checkpoint and checkpoint.index is not None:
        artifacts["index"] = {
            "url": next(
                (
                    item["url"]
                    for item in audit[audit_start:]
                    if item["url"].endswith(".safetensors.index.json")
                ),
                reference.url("model.safetensors.index.json", revision),
            ),
            "canonical_sha256": hashlib.sha256(
                json.dumps(checkpoint.index, sort_keys=True).encode()
            ).hexdigest(),
            "metadata": checkpoint.index.get("metadata", {}),
            "tensor_count": len(checkpoint.index["weight_map"]),
        }
    if len(shard_audit) == 1 and checkpoint and checkpoint.format == "safetensors":
        shard = next(iter(shard_audit.values()))
        artifacts.update(
            safetensors_url=shard["url"],
            header_length=shard["header_length"],
            header_sha256=shard["header_sha256"],
            file_size=shard["file_size"],
        )

    metadata = {
        "revision": revision,
        "requested_revision": reference.revision,
        "inspection": {
            "kind": "artifact_inspection",
            "reason_code": reason_code,
            "message": message,
            "files": sorted(files),
            "selected_checkpoint_files": sorted(selected_checkpoint_files),
            "checkpoint_available": any(
                name.endswith((".safetensors", ".safetensors.index.json", ".gguf")) for name in files
            ),
        },
        "artifacts": artifacts,
        "engine_version": ENGINE_VERSION,
        "analysis_report": {
            "mode": "artifact_inspection",
            "forward_executed": False,
            "runtime_trace_available": False,
        },
    }
    if model_profile is not None:
        metadata["model_profile"] = model_profile
    if checkpoint is None or checkpoint.format == "safetensors":
        metadata["checkpoint_tensor_element_count"] = (
            sum(item.parameter_count for item in checkpoint.tensors.values()) if checkpoint else None
        )
    if config_data is not None:
        metadata["source_config"] = config_data
        config_file = (model_profile or {}).get("selected_config") or "config.json"
        metadata["artifacts"]["config_url"] = reference.url(config_file, revision)
        metadata["artifacts"]["config_canonical_sha256"] = hashlib.sha256(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()

    evidence = [
        Evidence(
            "hf_repository_files",
            "direct",
            f"https://huggingface.co/api/models/{reference.model_id}/revision/{revision}",
            {"revision": revision, "files": sorted(files)},
        )
    ]
    if config_data is not None:
        evidence.append(
            Evidence(
                "hf_config",
                "direct",
                metadata["artifacts"]["config_url"],
                {"revision": revision, "canonical_sha256": artifacts["config_canonical_sha256"]},
            )
        )
    if model_profile and model_profile.get("manifest_canonical_sha256"):
        evidence.append(
            Evidence(
                "hf_pipeline_manifest",
                "direct",
                reference.url("model_index.json", revision),
                {
                    "canonical_sha256": model_profile["manifest_canonical_sha256"],
                    "declared_components": model_profile["components"],
                },
            )
        )
    if checkpoint and checkpoint.format == "gguf":
        evidence.append(
            Evidence(
                "gguf_header",
                "direct",
                next(iter(shard_audit.values()))["url"],
                {
                    "shards": shard_audit,
                    "selected_file_tensor_count": len(checkpoint.tensors),
                    "payload_bytes_read": 0,
                },
            )
        )
    elif checkpoint:
        evidence.append(
            Evidence(
                "safetensors_header",
                "direct",
                reference.model_id,
                {"shards": shard_audit, "payload_bytes_read": 0},
            )
        )
    ir = ModelIR(
        reference.model_id,
        "Unavailable",
        Graph("architecture_unavailable", [], []),
        metadata,
        tensors,
        evidence=evidence,
    )
    ir.root_graph.validate()
    return ir
