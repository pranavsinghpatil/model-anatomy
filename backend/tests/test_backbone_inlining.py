"""The registered backbone remains an inspectable graph boundary."""

from __future__ import annotations

import pytest

from model_anatomy.src.artifacts import CheckpointMetadata
from model_anatomy.src.completion import completion_report
from model_anatomy.src.facts import inspect_native_model
from model_anatomy.src.hierarchy import HierarchyBuilder
from model_anatomy.src.patterns import SemanticAnalyzer
from model_anatomy.src.registry import instantiate_native, load_native_config


@pytest.fixture(scope="module")
def gpt2_architecture():
    config = load_native_config(
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 32,
            "n_positions": 16,
            "n_embd": 16,
            "n_head": 2,
            "n_layer": 1,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    )
    model, _ = instantiate_native(config)
    raw = inspect_native_model(model, CheckpointMetadata({}, {}, {}))
    builder = HierarchyBuilder(model, raw, SemanticAnalyzer(model, raw))
    graph = builder.build()
    graph.validate()
    return raw, builder, graph


def test_backbone_and_head_remain_actual_registered_modules(gpt2_architecture):
    raw, _, graph = gpt2_architecture
    backbone = next(node for node in graph.nodes if node.module_path == "transformer")
    head = next(node for node in graph.nodes if node.module_path == "lm_head")

    assert backbone.child_graph is not None
    assert backbone.kind == "backbone"
    assert head.kind == "lm_head"
    assert {node.module_path for node in graph.nodes if node.module_path} == {
        "transformer", "lm_head"
    }
    assert backbone.metadata["parameter_refs"]
    assert all(ref in raw["parameters"] for ref in backbone.metadata["parameter_refs"])


def test_graph_nodes_are_registered_or_explicit_inputs(gpt2_architecture):
    raw, _, graph = gpt2_architecture
    nodes = [node for subgraph in graph.walk() for node in subgraph.nodes]

    assert any(node.module_path == "transformer.h.0" for node in nodes)
    assert all(node.module_path in raw["modules"] for node in nodes if node.module_path is not None)
    assert all(node.kind == "input" for node in nodes if node.module_path is None)
    assert all(edge.tensor is None for subgraph in graph.walk() for edge in subgraph.edges)


def test_missing_checkpoint_keeps_completion_partial_and_ownership_unknown(gpt2_architecture):
    raw, builder, graph = gpt2_architecture
    completion = completion_report(
        raw,
        builder.represented,
        builder.covered_by_leaf,
        {"architecture_selection": {"status": "declared_metadata_match"}},
        builder.topology.reports,
    )

    assert completion["checks"]["module_inventory"] is True
    assert completion["checks"]["parameter_mapping"] is False
    assert completion["status"] == "partial"
    assert completion["missing_parameter_paths"]
    assert all(
        node.outputs == [] for subgraph in graph.walk() for node in subgraph.nodes
        if node.module_path is not None
    )
