from types import SimpleNamespace

import torch

from model_anatomy.src import hierarchy
from model_anatomy.src.hierarchy import HierarchyBuilder, _output_port_specs
from model_anatomy.src.ir import Graph, Node, TensorSpec
from model_anatomy.src.static_flow import FlowEvent, StaticFlow, StaticFlowAnalyzer, StaticValue


class _ModuleListLoop(torch.nn.Module):
    def __init__(self, layer_count):
        super().__init__()
        self.config = SimpleNamespace(model_type="test", is_encoder_decoder=False, hidden_size=8)
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(layer_count)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _builder(flow):
    builder = object.__new__(HierarchyBuilder)
    builder.graph_ids = set()
    builder.flow_reports = {}
    builder.active = set()
    builder.model = SimpleNamespace()
    builder.raw = {"modules": {"": {"type": "Example", "children": []}}}
    builder.analyzer = SimpleNamespace(adapter=None, repetitions={}, classifications={})
    builder.inputs = SimpleNamespace(
        native_vit=False,
        spec=lambda name, path: TensorSpec(["B", "S", 8], "F32", name),
    )
    builder.evidence = lambda path, rule, kind="reconstructed": SimpleNamespace(
        source="test", kind=kind, locator=rule, details={}
    )
    return builder


def test_output_port_specs_preserve_named_ports_and_scalar_shape():
    specs = _output_port_specs(
        {
            "output_port_specs": {
                "output": {"shape": [], "dtype": "BOOL", "semantic_name": "condition"},
                "0": {"shape": ["B", 4], "dtype": "F32", "semantic_name": "left"},
                "1": {"shape": ["B", 5], "dtype": "F32", "semantic_name": "right"},
            }
        }
    )

    assert specs is not None
    assert specs["output"].shape == []
    assert specs["0"].shape == ["B", 4]
    assert specs["1"].shape == ["B", 5]


def test_output_port_specs_reject_malformed_contract_as_a_whole():
    assert _output_port_specs(
        {"output_port_specs": {"0": {"shape": [True], "dtype": "F32", "semantic_name": "bad"}}}
    ) is None
    assert _output_port_specs(
        {
            "output_port_specs": {
                "0": {"shape": None, "dtype": "F32", "semantic_name": "ok", "extra": 1}
            }
        }
    ) is None


def test_numeric_tuple_contract_does_not_claim_a_whole_tuple_output():
    builder = _builder(None)
    source = Node(
        "split",
        "Split",
        "split",
        outputs=[TensorSpec(["B", 4], "F32", "first_chunk")],
        metadata={
            "output_port_specs": {
                "0": {"shape": ["B", 4], "dtype": "F32", "semantic_name": "first_chunk"}
            }
        },
    )
    target = Node("out", "Output", "output")
    graph = Graph("g", [source, target], [])

    builder.connect(graph, source, target, source_port="output")

    assert graph.edges[0].tensor == TensorSpec(None, None, "unresolved_port:output")


def test_linear_does_not_rebuild_shape_from_default_input_or_mismatched_width():
    builder = _builder(None)
    linear = torch.nn.Linear(8, 4)
    builder.model.get_submodule = lambda path: linear
    builder.raw["modules"]["projection"] = {"type": "Linear"}
    node = Node("projection", "Projection", "projection", module_path="projection")

    unknown = builder.output_spec(node, TensorSpec(None, None, "lost_input_facts"))
    mismatch = builder.output_spec(node, TensorSpec(["B", 7], "F16", "incompatible_input"))
    valid = builder.output_spec(node, TensorSpec(["B", 8], "F16", "known_input"))

    assert unknown == TensorSpec(None, None, "unresolved_linear_input")
    assert mismatch == TensorSpec(None, None, "unresolved_linear_input")
    assert valid == TensorSpec(["B", 4], "F16", "projected_states")


def test_generic_unknown_and_passthrough_without_input_stay_unknown():
    builder = _builder(None)
    builder.model.get_submodule = lambda path: object()
    builder.raw["modules"]["mystery"] = {"type": "Mystery"}
    builder.raw["modules"]["norm"] = {"type": "LayerNorm"}

    unknown_module = builder.output_spec(Node("m", "Mystery", "projection", module_path="mystery"))
    missing_passthrough = builder.output_spec(Node("n", "Norm", "layer_norm", module_path="norm"))

    assert unknown_module == TensorSpec(None, None, "unknown")
    assert missing_passthrough == TensorSpec(None, None, "unresolved_passthrough_input")


def test_graph_uses_exact_output_port_specs_for_node_outputs_and_edges(monkeypatch):
    specs = {
        "0": {"shape": ["B", "S", 4], "dtype": "F32", "semantic_name": "first_chunk"},
        "1": {"shape": ["B", "S", 4], "dtype": "F32", "semantic_name": "second_chunk"},
    }
    event = FlowEvent(
        "split",
        None,
        "split",
        "Split",
        {"input": StaticValue(refs=[("input_ref", "output")])},
        12,
        {"output_port_specs": specs, "method": "chunk", "provenance": "test contract"},
    )
    flow = SimpleNamespace(
        inputs={"hidden_states": StaticValue(refs=[("input_ref", "output")])},
        events=[event],
        outputs=StaticValue(refs=[("split", "0"), ("split", "1")]),
        sequential_containers=set(),
        warnings=[],
        profile_assumptions=[],
        diagnostics=[{"code": "fixture_diagnostic"}],
    )
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: SimpleNamespace(analyze=lambda: flow))
    builder = _builder(flow)

    graph = builder.source_graph("")
    split = next(node for node in graph.nodes if node.kind == "split")
    split_edges = [edge for edge in graph.edges if edge.source == split.id]

    assert [spec.shape for spec in split.outputs] == [["B", "S", 4], ["B", "S", 4]]
    assert {edge.source_port: edge.tensor.shape for edge in split_edges} == {
        "0": ["B", "S", 4],
        "1": ["B", "S", 4],
    }
    assert graph.metadata["flow_diagnostics"] == [{"code": "fixture_diagnostic"}]
    assert graph.metadata["analysis_scope"]["profile"] == "canonical_inference"


def test_malformed_output_contract_does_not_fall_back_to_input_shape(monkeypatch):
    event = FlowEvent(
        "op",
        None,
        "operation",
        "Opaque operation",
        {"input": StaticValue(refs=[("input_ref", "output")])},
        20,
        {
            "output_port_specs": {
                "0": {"shape": [True], "dtype": "F32", "semantic_name": "bad"}
            }
        },
    )
    flow = SimpleNamespace(
        inputs={"hidden_states": StaticValue(refs=[("input_ref", "output")])},
        events=[event],
        outputs=StaticValue(refs=[("op", "0")]),
        sequential_containers=set(),
        warnings=[],
        profile_assumptions=[],
    )
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: SimpleNamespace(analyze=lambda: flow))
    graph = _builder(flow).source_graph("")
    operation = next(node for node in graph.nodes if node.kind == "operation")
    edge = next(edge for edge in graph.edges if edge.source == operation.id and edge.target_port == "input")

    assert operation.outputs[0] == TensorSpec(None, None, "unresolved_output_contract")
    assert edge.tensor == TensorSpec(None, None, "unresolved_port:0")


def test_contract_required_event_without_port_map_stays_unresolved(monkeypatch):
    event = FlowEvent(
        "split",
        None,
        "split",
        "Split",
        {"input": StaticValue(refs=[("input_ref", "output")])},
        21,
    )
    flow = SimpleNamespace(
        inputs={"hidden_states": StaticValue(refs=[("input_ref", "output")])},
        events=[event],
        outputs=StaticValue(refs=[("split", "output")]),
        sequential_containers=set(),
        warnings=[],
        profile_assumptions=[],
    )
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: SimpleNamespace(analyze=lambda: flow))
    graph = _builder(flow).source_graph("")
    split = next(node for node in graph.nodes if node.kind == "split")
    edge = next(edge for edge in graph.edges if edge.source == split.id and edge.target_port == "input")

    assert split.outputs[0] == TensorSpec(None, None, "unresolved_output_contract")
    assert edge.tensor == TensorSpec(None, None, "unresolved_port:output")


def test_required_primitive_contract_without_spec_cannot_reuse_incoming_shape(monkeypatch):
    event = FlowEvent(
        "primitive",
        None,
        "operation",
        "Primitive operation",
        {"input": StaticValue(refs=[("input_ref", "output")])},
        25,
        {"output_contract_required": True, "method": "validated_primitive"},
    )
    flow = SimpleNamespace(
        inputs={"hidden_states": StaticValue(refs=[("input_ref", "output")])},
        events=[event],
        outputs=StaticValue(refs=[("primitive", "output")]),
        sequential_containers=set(),
        warnings=[],
        profile_assumptions=[],
    )
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: SimpleNamespace(analyze=lambda: flow))
    graph = _builder(flow).source_graph("")
    primitive = next(node for node in graph.nodes if node.kind == "operation")
    edge = next(edge for edge in graph.edges if edge.source == primitive.id and edge.target_port == "input")

    assert primitive.outputs[0] == TensorSpec(None, None, "unresolved_output_contract")
    assert edge.tensor == TensorSpec(None, None, "unresolved_port:output")


def test_explicit_output_contract_survives_residual_shape_inference(monkeypatch):
    event = FlowEvent(
        "residual",
        None,
        "residual",
        "Residual",
        {
            "left": StaticValue(refs=[("left_ref", "output")]),
            "right": StaticValue(refs=[("right_ref", "output")]),
        },
        30,
        {
            "output_port_specs": {
                "output": {
                    "shape": ["B", "S", 4],
                    "dtype": "F16",
                    "semantic_name": "residual_result",
                }
            }
        },
    )
    flow = SimpleNamespace(
        inputs={
            "left": StaticValue(refs=[("left_ref", "output")]),
            "right": StaticValue(refs=[("right_ref", "output")]),
        },
        events=[event],
        outputs=StaticValue(refs=[("residual", "output")]),
        sequential_containers=set(),
        warnings=[],
        profile_assumptions=[],
    )
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: SimpleNamespace(analyze=lambda: flow))
    graph = _builder(flow).source_graph("")
    residual = next(node for node in graph.nodes if node.kind == "residual")
    output_edge = next(edge for edge in graph.edges if edge.source == residual.id and edge.target_port == "input")

    assert residual.outputs == [TensorSpec(["B", "S", 4], "F16", "residual_result")]
    assert output_edge.tensor == residual.outputs[0]


def test_residual_with_mixed_input_dtypes_keeps_shape_but_not_first_dtype(monkeypatch):
    event = FlowEvent(
        "residual",
        None,
        "residual",
        "Residual",
        {
            "left": StaticValue(refs=[("left_ref", "output")]),
            "right": StaticValue(refs=[("right_ref", "output")]),
        },
        31,
    )
    flow = SimpleNamespace(
        inputs={
            "left": StaticValue(refs=[("left_ref", "output")]),
            "right": StaticValue(refs=[("right_ref", "output")]),
        },
        events=[event],
        outputs=StaticValue(refs=[("residual", "output")]),
        sequential_containers=set(),
        warnings=[],
        profile_assumptions=[],
    )
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: SimpleNamespace(analyze=lambda: flow))
    builder = _builder(flow)
    input_specs = {
        "left": TensorSpec(["B", "S", 8], "F32", "left"),
        "right": TensorSpec(["B", "S", 8], "F16", "right"),
    }
    builder.inputs.spec = lambda name, path: input_specs[name]
    graph = builder.source_graph("")
    residual = next(node for node in graph.nodes if node.kind == "residual")

    assert residual.outputs == [TensorSpec(["B", "S", 8], None, "combined_states")]


def _primitive_graph(monkeypatch, input_spec, *, add_hook=False):
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="test", is_encoder_decoder=False)
    model.projection = torch.nn.Linear(8, 4)
    hook = model.projection.register_forward_hook(lambda *args: None) if add_hook else None
    event = FlowEvent(
        "projection_event",
        "projection",
        "module_call",
        "Projection",
        {"input": StaticValue(refs=[("input_ref", "output")])},
        40,
        {"output_contract_required": True},
    )
    flow = StaticFlow(
        inputs={"hidden_states": StaticValue(refs=[("input_ref", "output")])},
        events=[event],
        outputs=StaticValue(refs=[("projection_event", "output")]),
    )
    flow_analyzer = StaticFlowAnalyzer(model, "", {})
    flow_analyzer.analyze = lambda: flow
    monkeypatch.setattr(hierarchy, "StaticFlowAnalyzer", lambda *args: flow_analyzer)

    builder = _builder(flow)
    builder.model = model
    builder.raw["modules"] = {
        "": {"type": "Tiny", "children": ["projection"]},
        "projection": {"type": "Linear", "children": [], "attributes": {}},
    }
    builder.inputs.spec = lambda name, path: input_spec
    builder.call_node = lambda current_event, node_id: Node(
        node_id, "Projection", "projection", module_path=current_event.module_path
    )
    graph = builder.source_graph("")
    return graph, hook


def test_graph_input_spec_recovers_exact_required_linear_contract(monkeypatch):
    graph, _ = _primitive_graph(monkeypatch, TensorSpec(["B", "S", 8], "F32", "known_input"))
    projection = next(node for node in graph.nodes if node.module_path == "projection")
    exported = next(edge for edge in graph.edges if edge.source == projection.id and edge.target_port == "input")

    assert projection.outputs == [TensorSpec(["B", "S", 4], "F32", "native_linear_output")]
    assert projection.metadata["contract"] == "exact_native_torch_primitive.v1"
    assert projection.metadata["contract_provenance"] == "single_graph_input_edge"
    assert exported.tensor == projection.outputs[0]


def test_graph_reconciliation_rejects_hooks_mismatched_and_unknown_inputs(monkeypatch):
    cases = [
        (TensorSpec(["B", "S", 8], "F32", "known_input"), True),
        (TensorSpec(["B", "S", 7], "F32", "mismatched_input"), False),
        (TensorSpec(None, None, "unknown_input"), False),
    ]
    for input_spec, add_hook in cases:
        graph, hook = _primitive_graph(monkeypatch, input_spec, add_hook=add_hook)
        projection = next(node for node in graph.nodes if node.module_path == "projection")
        assert projection.outputs[0] == TensorSpec(None, None, "unresolved_output_contract")
        assert "contract" not in projection.metadata
        if hook is not None:
            hook.remove()


def _source_loop_builder(layer_count):
    model = _ModuleListLoop(layer_count)
    builder = _builder(None)
    builder.model = model
    builder.source_sequential_containers = {}
    builder.raw["modules"] = {
        "": {"type": "_ModuleListLoop", "children": ["layers"]},
        "layers": {"type": "ModuleList", "children": [f"layers.{i}" for i in range(layer_count)]},
        **{
            f"layers.{i}": {"type": "Identity", "children": [], "attributes": {}}
            for i in range(layer_count)
        },
    }
    builder.analyzer.classifications = {
        "": SimpleNamespace(kind="root"),
        "layers": SimpleNamespace(kind="container"),
    }
    builder.module_node = lambda path, node_id, repeat=None, build_child=True: Node(
        node_id, "Identity", "identity", module_path=path
    )
    return builder


def test_single_layer_module_list_uses_source_proven_sequential_graph():
    builder = _source_loop_builder(1)
    parent_graph = builder.source_graph("")

    assert "layers" in builder.source_sequential_containers
    graph = builder.build("layers")

    assert graph.metadata["topology"] == "static_source_proven_module_list"
    assert graph.metadata["sequence_evidence"]["source_module_path"] == ""
    assert graph.metadata["sequence_evidence"]["rule"]
    assert [edge.relation for edge in graph.edges] == ["dataflow", "dataflow"]
    assert parent_graph.metadata["topology"] == "static_source_dependencies"


def test_repeated_module_list_keeps_existing_repetition_graph_path():
    from model_anatomy.src.patterns import Repetition

    builder = _source_loop_builder(2)
    builder.analyzer.repetitions = {
        "layers": [Repetition("layers", ["layers.0", "layers.1"], "signature", "sequential")]
    }
    builder.source_graph("")

    graph = builder.build("layers")

    assert graph.metadata["topology"] == "repeated subtrees"
    assert graph.metadata["total_instances"] == 2
