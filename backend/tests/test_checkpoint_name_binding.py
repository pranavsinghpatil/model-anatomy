"""A prior name audit is bound only to a matching new exact-pin report."""

import copy
import json
from pathlib import Path

import pytest

from model_anatomy.api.schemas import ArtifactInspectionResponse, VerifiedCheckpointNames
from model_anatomy.service.jobs import ir_summary
from model_anatomy.service.views import artifact_view, node_view
from model_anatomy.src.artifacts import ModelReference
from model_anatomy.src.checkpoint_audit_binding import INDEX_SHA256
from model_anatomy.src.checkpoint_name_binding import (
    NAME_AUDIT_SHA256,
    bind_pinned_name_audit,
    validate_name_audit,
)
from model_anatomy.src.ir import Graph, ModelIR
from model_anatomy.src.publisher_topology import (
    MODEL_ID,
    REQUIRED_FILES,
    REVISION,
    build_reviewed_topology,
)

ROOT = Path(__file__).parents[2]
CONFIG = json.loads((Path(__file__).parent / "fixtures" / "deepseek_v41_config.json").read_text())
HEADER_DOCUMENT = json.loads(
    (ROOT / "backend" / "model_anatomy" / "src" / "data" / "deepseek-v41-headers.json").read_text()
)
NAME_PATH = ROOT / "backend" / "model_anatomy" / "src" / "data" / "deepseek-v41-name-coverage.json"
NAME_DOCUMENT = json.loads(NAME_PATH.read_text())


def fresh_ir() -> ModelIR:
    reference = ModelReference(MODEL_ID, REVISION)
    verification = HEADER_DOCUMENT["verification"]
    ir = ModelIR(
        MODEL_ID,
        "",
        Graph("empty", [], []),
        metadata={
            "revision": REVISION,
            "artifacts": {
                "declared_checkpoint_index": {
                    "url": reference.url("model.safetensors.index.json", REVISION),
                    "canonical_sha256": INDEX_SHA256,
                    "declared_tensor_names": verification["tensor_count"],
                    "declared_shards": verification["shard_count"],
                    "tensor_record_limit": 20000,
                }
            },
        },
    )
    files = REQUIRED_FILES | set(verification["shards"])
    assert build_reviewed_topology(reference, REVISION, files, CONFIG, ir) is ir
    return ir


def test_exact_pin_binds_name_evidence_without_parameter_claim():
    assert NAME_PATH.read_bytes() == (ROOT / "docs" / "evidence" / NAME_PATH.name).read_bytes()
    ir = fresh_ir()
    summary = ir_summary(ir.to_dict())
    names = summary["verified_checkpoint_names"]
    assert VerifiedCheckpointNames.model_validate(names)
    assert names["state_key_count"] == 96085
    assert names["component_count"] == 632
    assert names["name_audit_document_sha256"] == NAME_AUDIT_SHA256
    assert ir.metadata["artifacts"]["name_audit_binding"] == {"status": "bound"}
    assert ir.metadata["artifacts"]["header_audit_binding"] == {"status": "bound"}
    assert summary["result_status"] == "partial"
    assert summary["parameter_count"] is None
    assert summary["tensor_count"] == 0
    assert summary["quality"]["checkpoint_parameter_coverage"] is None
    assert ir.tensors == {}
    assert ArtifactInspectionResponse.model_validate(artifact_view(ir))
    assert artifact_view(ir)["verified_checkpoint_names"] == names
    layer = node_view(ir, "publisher_reference_topology", "layer_14")
    assert layer["node"]["metadata"]["checkpoint_state_key_count"] == 6
    assert layer["tensor_count"] == 0
    assert any(item["source"] == "pinned_checkpoint_name_audit" for item in layer["evidence"])
    image = node_view(ir, "publisher_reference_topology", "image_start")
    assert image["node"]["metadata"]["checkpoint_state_key_count"] == 1


@pytest.mark.parametrize(
    "change",
    ["index", "config", "source", "count", "component", "pattern", "converter", "header_status"],
)
def test_name_audit_rejects_changed_identity_or_incomplete_counts(change):
    ir = fresh_ir()
    document = copy.deepcopy(NAME_DOCUMENT)
    if change == "index":
        document["index_sha256"] = "a" * 64
    elif change == "config":
        document["config_canonical_sha256"] = "a" * 64
    elif change == "source":
        document["reviewed_sources_sha256"]["inference/model.py"] = "a" * 64
    elif change == "count":
        document["component_state_key_counts"]["image_start"] += 1
    elif change == "component":
        document["component_state_key_counts"]["unreviewed.component"] = 1
        document["component_count"] += 1
    elif change == "pattern":
        document["normalized_patterns"]["image_start"] += 1
    elif change == "converter":
        document["converter_skipped_names"] = 1
    else:
        ir.metadata["artifacts"]["header_audit_binding"] = {"status": "not_bound"}
    assert validate_name_audit(document, ir) is None


def test_missing_or_corrupt_bundle_remains_unbound(monkeypatch):
    from model_anatomy.src import checkpoint_name_binding as binding

    monkeypatch.setattr(binding, "NAME_AUDIT_SHA256", "0" * 64)
    ir = fresh_ir()
    assert ir.metadata["artifacts"]["name_audit_binding"] == {
        "status": "not_bound",
        "reason_code": "bundled_name_audit_digest_mismatch",
    }
    assert "verified_checkpoint_names" not in ir.metadata["artifacts"]
    assert all(
        "checkpoint_state_key_count" not in node.metadata
        for graph in ir.root_graph.walk()
        for node in graph.nodes
    )


def test_other_model_or_revision_cannot_bind():
    ir = fresh_ir()
    assert not bind_pinned_name_audit(ir, ModelReference("other/model", REVISION), REVISION)
    assert ir.metadata["artifacts"]["name_audit_binding"]["reason_code"] == "model_or_revision_mismatch"
