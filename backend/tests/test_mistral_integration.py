"""Pinned Mistral-family static-report acceptance; metadata only, no forward pass."""

import os

import pytest

from model_anatomy.service.jobs import ir_summary
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.version import ENGINE_VERSION


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests"
)
@pytest.mark.parametrize(
    ("model_id", "revision", "parameters"),
    [
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
            7_248_023_552,
        ),
        (
            "HuggingFaceH4/zephyr-7b-beta",
            "892b3d7a7b1cf10c7a701c60881cd93df615734c",
            7_241_732_096,
        ),
    ],
)
def test_pinned_mistral_family_has_complete_static_report(model_id, revision, parameters):
    ir = reconstruct(model_id, revision=revision)
    report = ir.metadata["analysis_report"]
    summary = ir_summary(ir.to_dict())

    assert ir.model_id == model_id
    assert ir.architecture == "MistralForCausalLM"
    assert ir.metadata["revision"] == revision
    assert ir.metadata["engine_version"] == ENGINE_VERSION
    assert ir.metadata["unique_parameter_count"] == parameters
    assert len(ir.tensors) == 291
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert report["forward_executed"] is False
    assert report["runtime_trace_available"] is False
    assert summary["result_status"] == "completed"
    assert summary["warnings"] == []
    assert summary["partial_reasons"] == []
    assert all(
        summary["quality"][key] == 1
        for key in (
            "semantic_coverage",
            "module_semantic_coverage",
            "checkpoint_parameter_coverage",
            "tensor_shape_coverage",
        )
    )
    assert summary["quality"]["unknown_nodes"] == 0
    assert summary["quality"]["unmapped_checkpoint_tensors"] == 0
    assert any(
        item["code"] == "reviewed_native_mask_alias"
        for flow in ir.raw_structure["static_flow_reports"].values()
        for item in flow["diagnostics"]
    )
    assert any("mask values and attention-mask dataflow" in item for item in ir.metadata["limitations"])
    assert all(
        edge.tensor is not None and edge.tensor.shape is not None
        for graph in ir.root_graph.walk()
        for edge in graph.edges
        if edge.relation == "dataflow"
    )
    ir.root_graph.validate()
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
