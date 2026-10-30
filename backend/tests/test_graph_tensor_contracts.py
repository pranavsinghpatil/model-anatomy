"""Current graph contract: registered modules and source-backed dependencies.

Tensor expressions belonged to the retired StaticFlow graph. The production
hierarchy deliberately leaves intermediate activation shapes unknown.
"""

from __future__ import annotations

import torch

from model_anatomy.src.artifacts import CheckpointMetadata, TensorMetadata
from model_anatomy.src.completion import completion_report
from model_anatomy.src.facts import inspect_native_model
from model_anatomy.src.hierarchy import HierarchyBuilder
from model_anatomy.src.patterns import SemanticAnalyzer


class Chain(torch.nn.Module):
    base_model_prefix = ""
    _keys_to_ignore_on_load_unexpected = ()

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 3)
        self.second = torch.nn.Linear(3, 3)
        self.registered_but_unused = torch.nn.Identity()

    def forward(self, x):
        hidden = self.first(x)
        activated = torch.relu(hidden)
        return self.second(activated)


class LayerLoop(torch.nn.Module):
    base_model_prefix = ""
    _keys_to_ignore_on_load_unexpected = ()

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _checkpoint(model: torch.nn.Module, *, omit: str | None = None, extra: bool = False):
    tensors = {}
    offset = 0
    for name, parameter in model.named_parameters():
        if name == omit:
            continue
        count = parameter.numel()
        tensors[name] = TensorMetadata(
            name, "F32", tuple(parameter.shape), count, (offset, offset + 4 * count)
        )
        offset += 4 * count
    if extra:
        tensors["orphan.weight"] = TensorMetadata(
            "orphan.weight", "F32", (1,), 1, (offset, offset + 4)
        )
    return CheckpointMetadata(tensors, {}, {name: "model.safetensors" for name in tensors})


def _build(model: torch.nn.Module, checkpoint: CheckpointMetadata):
    raw = inspect_native_model(model, checkpoint)
    builder = HierarchyBuilder(model, raw, SemanticAnalyzer(model, raw))
    graph = builder.build()
    graph.validate()
    return raw, builder, graph


def _completion(raw, builder):
    return completion_report(
        raw,
        builder.represented,
        builder.covered_by_leaf,
        {"architecture_selection": {"status": "declared_metadata_match"}},
        builder.topology.reports,
    )


def test_registered_modules_and_checkpoint_binding_survive_expression_elision():
    model = Chain()
    raw, builder, graph = _build(model, _checkpoint(model))

    paths = {node.module_path for subgraph in graph.walk() for node in subgraph.nodes}
    assert {"first", "second", "registered_but_unused"} <= paths
    assert {name for name, fact in raw["parameters"].items() if fact["checkpoint_matched"]} == {
        name for name, _ in model.named_parameters()
    }
    assert graph.metadata["registration_order_is_execution_order"] is False
    assert _completion(raw, builder)["checks"]["module_inventory"] is True

    # A source dependency through torch.relu is useful topology evidence, but
    # neither its activation values nor a tensor identity were inspected.
    first = next(node for node in graph.nodes if node.module_path == "first")
    second = next(node for node in graph.nodes if node.module_path == "second")
    unused = next(node for node in graph.nodes if node.module_path == "registered_but_unused")
    assert any(edge.source == first.id and edge.target == second.id for edge in graph.edges)
    assert all(edge.tensor is None and edge.relation in {"dependency", "conditional_dependency"}
               for edge in graph.edges)
    assert first.outputs == second.inputs == second.outputs == []
    assert unused.metadata["invocation_status"] == "registration_only"


def test_checkpoint_gaps_make_architecture_completion_partial_without_fake_tensor_facts():
    model = Chain()
    raw, builder, graph = _build(
        model, _checkpoint(model, omit="second.bias", extra=True)
    )
    result = _completion(raw, builder)

    assert result["status"] == "partial"
    assert result["checks"]["parameter_mapping"] is False
    assert "second.bias" in result["missing_parameter_paths"]
    assert result["unmapped_checkpoint_tensors"] == ["orphan.weight"]
    assert "Intermediate activation shapes" in result["excluded_from_scope"]
    assert all(edge.tensor is None for subgraph in graph.walk() for edge in subgraph.edges)


def test_module_list_retains_each_instance_and_only_source_proven_order():
    model = LayerLoop()
    raw, builder, graph = _build(model, _checkpoint(model))
    container = next(node for node in graph.nodes if node.module_path == "layers")
    child = container.child_graph

    assert child is not None
    assert container.metadata["instance_paths"] == ["layers.0", "layers.1"]
    assert [node.module_path for node in child.nodes] == ["layers.0", "layers.1"]
    assert any(
        edge.source == child.nodes[0].id and edge.target == child.nodes[1].id
        for edge in child.edges
    )
    assert all(edge.tensor is None for edge in child.edges)
    assert _completion(raw, builder)["scope"].startswith("Native registered module architecture")
