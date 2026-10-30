"""User-reported repositories, pinned and inspected without fetching weights."""

import os

import pytest

from model_anatomy.service.jobs import ir_summary
from model_anatomy.service.views import artifact_view, export_files
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests"
)
@pytest.mark.parametrize(
    "model_id,revision,reason,tensors",
    [
        ("Aleph-Alpha/Kolibri-1", "35bc4d3be745502227a67247de77d70e691614ee", "metadata_budget_exceeded", 0),
        (
            "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA",
            "0d91c4dfa22b3f5aa89c020eadc30d8f42cb51c0",
            "adapter_requires_base_model",
            384,
        ),
    ],
)
def test_real_artifact_inspection(model_id, revision, reason, tensors):
    ir = reconstruct(model_id, revision=revision, allow_artifact_inspection=True)
    assert ir.model_id == model_id and ir.metadata["revision"] == revision
    assert ir.metadata["inspection"]["reason_code"] == reason
    assert ir.metadata["inspection"]["files"]
    assert len(ir.tensors) == tensors
    assert ir.root_graph.nodes == [] and ir.root_graph.edges == []
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert "unique_parameter_count" not in ir.metadata
    assert "semantic_coverage" not in ir.metadata["analysis_report"]
    assert ir_summary(ir.to_dict())["result_status"] == "inspected"
    assert artifact_view(ir)["tensor_count"] == tensors
    assert not any(name.endswith(".mmd") for name in export_files(ir))
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
    if tensors:
        tensor = ir.tensors["diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight"]
        assert tensor["shape"] == [32, 4096] and tensor["dtype"] == "F16"
        assert tensor["artifact"].endswith(f"/{revision}/multiple_angles_step1000.safetensors")
