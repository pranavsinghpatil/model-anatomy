"""Pinned encoder checkpoints, including legacy names and honest partial reports."""

import os

import pytest

from model_anatomy.api.schemas import GraphResponse
from model_anatomy.service.jobs import ir_summary
from model_anatomy.service.views import graph_view
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1"),
]


@pytest.mark.parametrize(
    "model,revision,architecture,parameters,tensors",
    [
        (
            "google-bert/bert-base-uncased",
            "86b5e0934494bd15c9632b12f734a8a67f723594",
            "BertForMaskedLM",
            109514298,
            206,
        ),
        (
            "distilbert/distilbert-base-uncased",
            "12040accade4e8a0f71eabdb258fecc2e7e948be",
            "DistilBertForMaskedLM",
            66985530,
            105,
        ),
    ],
)
def test_pinned_encoder_preserves_parameters_and_exposes_remaining_gaps(
    model, revision, architecture, parameters, tensors
):
    ir = reconstruct(model, revision=revision)
    assert ir.model_id == model and ir.metadata["revision"] == revision
    assert ir.architecture == architecture
    assert ir.metadata["unique_parameter_count"] == parameters
    assert len(ir.tensors) == tensors
    assert ir.metadata["inputs"]["input_specs"] == {"input_ids": {"shape": ["B", "S"], "dtype": "I64"}}
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
    assert GraphResponse.model_validate(graph_view(ir, ir.root_graph.id)).nodes
    # BERT's extra pretraining head tensors and DistilBERT's unresolved semantics
    # remain visible; native construction alone must not become a complete badge.
    assert ir_summary(ir.to_dict())["result_status"] == "partial"
    if architecture == "BertForMaskedLM":
        name = "bert.embeddings.LayerNorm.gamma"
        assert ir.tensors[name]["parameter_refs"] == ["bert.embeddings.LayerNorm.weight"]
        assert ir.tensors[name]["artifact"].endswith(f"/{revision}/model.safetensors")
        assert set(ir.metadata["analysis_report"]["unmapped_checkpoint_tensors"]) == {
            "bert.pooler.dense.bias",
            "bert.pooler.dense.weight",
            "cls.seq_relationship.bias",
            "cls.seq_relationship.weight",
        }
