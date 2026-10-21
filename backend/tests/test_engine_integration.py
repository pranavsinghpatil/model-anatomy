"""Opt-in real pinned-model checks; no checkpoint weight downloads."""

import os

import pytest

from model_anatomy.service.views import node_view
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests"
)
@pytest.mark.parametrize(
    ("model_id", "revision", "parameters", "layers", "hidden"),
    [
        ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", 124439808, 12, 768),
        ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", 134515008, 30, 576),
        ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea", 4022468096, 36, 2560),
        ("google/vit-base-patch16-224", "10064cc17b69d623a025c9e90e11230cc8ba6967", 86567656, 12, 768),
    ],
)
def test_real_pinned_model(model_id, revision, parameters, layers, hidden):
    ir = reconstruct(model_id, revision=revision)
    assert ir.model_id == model_id
    assert ir.metadata["revision"] == revision
    assert ir.metadata["unique_parameter_count"] == parameters
    assert ir.metadata["dimensions"]["layers"] == layers
    assert ir.metadata["dimensions"]["hidden"] == hidden
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert ir.metadata["analysis_report"]["module_inventory_coverage"] == 1
    ir.root_graph.validate()
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
    assert ir.metadata["analysis_report"]["unmapped_checkpoint_tensors"] == []
    assert all(tensor["parameter_refs"] for tensor in ir.tensors.values() if tensor["role"] == "parameter")
    for graph in ir.root_graph.walk():
        assert all(node.module_path in ir.raw_structure["modules"] or node.kind == "input" for node in graph.nodes)
        assert all(edge.evidence for edge in graph.edges)
    if model_id == "google/vit-base-patch16-224":
        assert ir.architecture == "ViTForImageClassification"
        classifier = next(n for n in ir.root_graph.nodes if n.module_path == "classifier")
        assert classifier.kind == "classifier"
        assert classifier.parameter_count == 769000
        component = node_view(ir, ir.root_graph.id, classifier.id)
        assert [tensor["name"] for tensor in component["tensors"]] == [
            "classifier.bias", "classifier.weight",
        ]
