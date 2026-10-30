"""Bind a previously verified, exact-pin header audit to a new partial report.

The audit is an immutable observation, not a shard fetch performed by this job.
No tensor-to-module mapping or parameter total follows from these aggregates.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files as package_files
from typing import Any

from .artifacts import ModelReference
from .ir import Evidence, ModelIR

AUDIT_DOCUMENT_SHA256 = "772fbc8a93975dae22daf651d74876747e36b6af00f738c77b8818e261a9baa5"
INDEX_SHA256 = "abc09b0491e66b9f1c3d9dd635896523f2eaef3ca3904a91cfaa216390de7ba4"
AUDIT_FILE = "data/deepseek-v41-headers.json"


def _unbound(ir: ModelIR, reason_code: str) -> bool:
    ir.metadata["artifacts"]["header_audit_binding"] = {
        "status": "not_bound",
        "reason_code": reason_code,
    }
    ir.metadata["analysis_report"]["warnings"].append(
        f"The prior pinned header audit was not bound ({reason_code}); "
        "shard headers remain unverified in this report."
    )
    return False


def validate_audit_document(
    document: dict[str, Any],
    reference: ModelReference,
    revision: str,
    repository_files: set[str],
    declared_index: dict[str, Any],
) -> dict[str, Any] | None:
    """Accept only the entire audited checkpoint at the matching pinned index."""
    if document.get("model_id") != reference.model_id or document.get("revision") != revision:
        return None
    verification = document.get("verification")
    if not isinstance(verification, dict) or verification.get("index_sha256") != INDEX_SHA256:
        return None
    if (
        declared_index.get("canonical_sha256") != INDEX_SHA256
        or declared_index.get("url") != reference.url("model.safetensors.index.json", revision)
        or "model.safetensors.index.json" not in repository_files
    ):
        return None
    shards = verification.get("shards")
    if not isinstance(shards, dict) or not shards:
        return None
    positive = lambda value: type(value) is int and value > 0
    nonnegative = lambda value: type(value) is int and value >= 0
    if (
        not positive(verification.get("tensor_count"))
        or not positive(verification.get("tensor_element_count"))
        or not positive(verification.get("shard_count"))
        or not positive(verification.get("header_bytes_read"))
        or type(verification.get("weight_payload_bytes_read")) is not int
        or verification["weight_payload_bytes_read"] != 0
        or verification["shard_count"] != len(shards)
        or declared_index.get("declared_tensor_names") != verification["tensor_count"]
        or declared_index.get("declared_shards") != verification["shard_count"]
    ):
        return None
    tensor_count = element_count = header_bytes = 0
    for name, shard in shards.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name)
            or name not in repository_files
            or not isinstance(shard, dict)
            or shard.get("url") != reference.url(name, revision)
            or not isinstance(shard.get("header_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", shard["header_sha256"])
            or not positive(shard.get("header_length"))
            or not positive(shard.get("file_size"))
            or shard["file_size"] < shard["header_length"] + 8
            or not nonnegative(shard.get("tensor_count"))
            or not nonnegative(shard.get("tensor_element_count"))
        ):
            return None
        tensor_count += shard["tensor_count"]
        element_count += shard["tensor_element_count"]
        header_bytes += shard["header_length"] + 8
    if (
        tensor_count != verification["tensor_count"]
        or element_count != verification["tensor_element_count"]
        or header_bytes != verification["header_bytes_read"]
    ):
        return None
    return {
        "basis": "previously_verified_pinned_header_audit",
        "index_sha256": INDEX_SHA256,
        "shard_count": len(shards),
        "tensor_count": tensor_count,
        "tensor_element_count_including_scales": element_count,
        "header_bytes_read_by_audit": header_bytes,
        "weight_payload_bytes_read_by_audit": 0,
    }


def bind_pinned_header_audit(
    ir: ModelIR, reference: ModelReference, revision: str, repository_files: set[str]
) -> bool:
    """Attach trusted prior evidence only when this report's index identity agrees."""
    declared = ir.metadata.get("artifacts", {}).get("declared_checkpoint_index")
    if not isinstance(declared, dict):
        return _unbound(ir, "index_declaration_missing")
    try:
        raw = package_files("model_anatomy.src").joinpath(AUDIT_FILE).read_bytes()
    except OSError:
        return _unbound(ir, "bundled_audit_unavailable")
    if hashlib.sha256(raw).hexdigest() != AUDIT_DOCUMENT_SHA256:
        return _unbound(ir, "bundled_audit_digest_mismatch")
    document = json.loads(raw)
    summary = validate_audit_document(document, reference, revision, repository_files, declared)
    if summary is None:
        return _unbound(ir, "index_or_shard_identity_mismatch")
    summary["audit_document_sha256"] = AUDIT_DOCUMENT_SHA256
    ir.metadata["artifacts"]["verified_checkpoint_headers"] = summary
    ir.metadata["artifacts"]["header_audit_binding"] = {"status": "bound"}
    ir.metadata["source_topology"]["checkpoint_headers_verified"] = True
    ir.metadata["analysis_report"]["warnings"] = [
        "This partial graph is from reviewed publisher reference source, not a constructed native model.",
        "A separate pinned audit verified checkpoint shard headers; tensor ownership, parameter counts and dataflow remain unavailable.",
    ]
    ir.metadata["limitations"] = [
        "The source topology is pinned to one reviewed revision and exact configuration.",
        "Reference-order links are not runtime dataflow or shape evidence.",
        "Checkpoint headers were verified by a prior separate audit, not reread during this inspection.",
        "Aggregate tensor elements include scales and do not establish model parameter count or module ownership.",
    ]
    ir.evidence.append(
        Evidence(
            "pinned_checkpoint_header_audit",
            "derived",
            reference.url("model.safetensors.index.json", revision),
            {**summary, "shards": document["verification"]["shards"]},
        )
    )
    return True
