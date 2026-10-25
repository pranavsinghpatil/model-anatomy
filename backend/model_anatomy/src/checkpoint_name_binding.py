"""Bind a prior exact-pin publisher state-key audit to a new partial report."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files as package_files
from typing import Any

from .artifacts import ModelReference
from .checkpoint_audit_binding import AUDIT_DOCUMENT_SHA256, INDEX_SHA256
from .ir import Evidence, ModelIR
from .publisher_name_coverage import CONVERTER_SHA256
from .publisher_topology import (
    CONFIG_CANONICAL_SHA256,
    MODEL_ID,
    REVIEWED_SOURCE_SHA256,
    REVISION,
)

NAME_AUDIT_FILE = "data/deepseek-v41-name-coverage.json"
NAME_AUDIT_SHA256 = "8f0f95214ad0dbafcf5ac1031deaba3d062b943ecfe86cd83678a9d44ace6512"


def _unbound(ir: ModelIR, reason_code: str) -> bool:
    ir.metadata["artifacts"]["name_audit_binding"] = {
        "status": "not_bound",
        "reason_code": reason_code,
    }
    if ir.metadata["artifacts"].get("verified_checkpoint_headers"):
        ir.metadata["analysis_report"]["warnings"].append(
            f"The prior pinned checkpoint-name audit was not bound ({reason_code}); "
            "reference-component name correspondence remains unverified in this report."
        )
    return False


def validate_name_audit(document: dict[str, Any], ir: ModelIR) -> dict[str, Any] | None:
    """Accept only the complete prior audit at the same bound index and source pin."""
    artifacts = ir.metadata.get("artifacts", {})
    headers = artifacts.get("verified_checkpoint_headers")
    source = ir.metadata.get("source_topology", {})
    if not isinstance(headers, dict) or not isinstance(source, dict):
        return None
    if (
        artifacts.get("header_audit_binding") != {"status": "bound"}
        or headers.get("basis") != "previously_verified_pinned_header_audit"
        or headers.get("audit_document_sha256") != AUDIT_DOCUMENT_SHA256
        or headers.get("index_sha256") != INDEX_SHA256
        or headers.get("index_sha256") != document.get("index_sha256")
        or headers.get("weight_payload_bytes_read_by_audit") != 0
        or headers.get("tensor_count") != document.get("state_key_count")
        or headers.get("shard_count") != document.get("index_shard_count")
        or source.get("config_canonical_sha256") != CONFIG_CANONICAL_SHA256
        or source.get("reviewed_sources") != REVIEWED_SOURCE_SHA256
        or document.get("config_canonical_sha256") != CONFIG_CANONICAL_SHA256
        or document.get("reviewed_sources_sha256") != REVIEWED_SOURCE_SHA256
        or document.get("converter_sha256") != CONVERTER_SHA256
        or document.get("basis") != "reviewed_publisher_state_key_inventory"
    ):
        return None
    positive = lambda value: type(value) is int and value > 0
    zero = lambda value: type(value) is int and value == 0
    if (
        not positive(document.get("state_key_count"))
        or document.get("matched_state_key_count") != document["state_key_count"]
        or not positive(document.get("component_count"))
        or not positive(document.get("normalized_pattern_count"))
        or not all(
            zero(document.get(key))
            for key in (
                "missing_state_key_count",
                "unexpected_state_key_count",
                "converter_renamed_names",
                "converter_skipped_names",
                "converter_name_collisions",
                "weight_payload_bytes_read",
            )
        )
    ):
        return None
    counts = document.get("component_state_key_counts")
    patterns = document.get("normalized_patterns")
    if not isinstance(counts, dict) or len(counts) != document["component_count"]:
        return None
    if not isinstance(patterns, dict) or len(patterns) != document["normalized_pattern_count"]:
        return None
    nodes = {node.module_path for graph in ir.root_graph.walk() for node in graph.nodes}
    if (
        not all(isinstance(path, str) and path in nodes and positive(count) for path, count in counts.items())
        or not all(isinstance(pattern, str) and positive(count) for pattern, count in patterns.items())
        or sum(counts.values()) != document["state_key_count"]
        or sum(patterns.values()) != document["state_key_count"]
    ):
        return None
    return {
        "basis": "previously_audited_publisher_state_keys",
        "index_sha256": document["index_sha256"],
        "name_audit_document_sha256": NAME_AUDIT_SHA256,
        "state_key_count": document["state_key_count"],
        "component_count": document["component_count"],
        "unmatched_state_key_count": 0,
        "converter_renamed_names": 0,
        "converter_skipped_names": 0,
        "converter_name_collisions": 0,
        "weight_payload_bytes_read_by_audit": 0,
    }


def bind_pinned_name_audit(ir: ModelIR, reference: ModelReference, revision: str) -> bool:
    """Expose a digest-pinned prior audit without reading checkpoint payloads."""
    if reference.model_id != MODEL_ID or ir.model_id != MODEL_ID or revision != REVISION:
        return _unbound(ir, "model_or_revision_mismatch")
    try:
        raw = package_files("model_anatomy.src").joinpath(NAME_AUDIT_FILE).read_bytes()
    except OSError:
        return _unbound(ir, "bundled_name_audit_unavailable")
    if hashlib.sha256(raw).hexdigest() != NAME_AUDIT_SHA256:
        return _unbound(ir, "bundled_name_audit_digest_mismatch")
    document = json.loads(raw)
    summary = validate_name_audit(document, ir)
    if summary is None:
        return _unbound(ir, "index_source_or_component_mismatch")
    counts = document["component_state_key_counts"]
    locator = reference.url("model.safetensors.index.json", revision)
    for graph in ir.root_graph.walk():
        for node in graph.nodes:
            count = counts.get(node.module_path)
            if count is None:
                continue
            node.metadata["checkpoint_state_key_count"] = count
            node.metadata["checkpoint_name_basis"] = summary["basis"]
            node.evidence.append(
                Evidence(
                    "pinned_checkpoint_name_audit",
                    "derived",
                    locator,
                    {
                        "component_path": node.module_path,
                        "state_key_count": count,
                        "name_audit_document_sha256": NAME_AUDIT_SHA256,
                    },
                )
            )
    ir.metadata["artifacts"]["verified_checkpoint_names"] = summary
    ir.metadata["artifacts"]["name_audit_binding"] = {"status": "bound"}
    ir.metadata["analysis_report"]["warnings"] = [
        "This partial graph is from reviewed publisher reference source, not a constructed native model.",
        "Separate pinned audits verified shard headers and checkpoint-name correspondence; tensor shapes, parameter counts and runtime dataflow remain unavailable.",
    ]
    ir.metadata["limitations"].append(
        "Component state-key counts are names, not parameter elements or proof of converter tensor-value equivalence."
    )
    ir.evidence.append(Evidence("pinned_checkpoint_name_audit", "derived", locator, summary))
    return True
