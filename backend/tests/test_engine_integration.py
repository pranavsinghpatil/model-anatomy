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
    ir.root_graph.validate()
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
    assert ir.metadata["analysis_report"]["unmapped_checkpoint_tensors"] == []
    assert all(tensor["parameter_refs"] for tensor in ir.tensors.values() if tensor["role"] == "parameter")
    if model_id == "google/vit-base-patch16-224":
        assert ir.architecture == "ViTForImageClassification"
        assert ir.metadata["inputs"]["input_specs"]["pixel_values"] == {
            "shape": ["B", 3, 224, 224],
            "dtype": "F32",
        }
        nodes = {node.id: node for node in ir.root_graph.nodes}
        assert nodes["pixel_values"].kind == "input"
        assert nodes["vit"].kind == "backbone"
        assert nodes["vit"].child_graph is not None
        assert nodes["classifier"].kind == "classifier"
        assert nodes["classifier"].parameter_count == 769000
        component = node_view(ir, ir.root_graph.id, "classifier")
        assert [tensor["name"] for tensor in component["tensors"]] == [
            "classifier.bias",
            "classifier.weight",
        ]
        assert {
            (edge.source, edge.target) for edge in ir.root_graph.edges if edge.relation == "dataflow"
        } >= {
            ("pixel_values", "vit"),
            ("vit", "selection"),
            ("selection", "classifier"),
            ("classifier", "output"),
        }
        assert nodes["vit"].outputs[0].shape == ["B", 197, 768]
        assert nodes["selection"].outputs[0].shape == ["B", 768]
        assert nodes["classifier"].inputs[0].shape == ["B", 768]
        assert nodes["classifier"].outputs[0].shape == ["B", 1000]
        assert ir.metadata["analysis_report"]["unknown_module_paths"] == []
