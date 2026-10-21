"""Inlining keeps the actual source port and tensor facts at the graph boundary."""

from types import SimpleNamespace

from model_anatomy.src.hierarchy import HierarchyBuilder
from model_anatomy.src.ir import Edge, Graph, Node, TensorSpec

IDS = TensorSpec(["B", "S"], "I64", "input_ids")
HIDDEN = TensorSpec(["B", "S", 8], "F32", "hidden_states")
OTHER = TensorSpec(["B", "S", 4], "F32", "other_states")
UNKNOWN = TensorSpec(None, None, "unresolved_port:last_hidden_state")


def _graph(*, parent_port: str = "last_hidden_state", multiple: bool = False) -> Graph:
    child_input = Node("tokens", "Tokens", "input", outputs=[IDS], metadata={"argument_name": "input_ids"})
    hidden = Node("hidden", "Hidden", "operation", inputs=[IDS], outputs=[HIDDEN])
    child_output = Node("result", "Output", "output", inputs=[HIDDEN], outputs=[HIDDEN])
    child_nodes = [child_input, hidden, child_output]
    child_edges = [
        Edge("tokens", "hidden", IDS),
        Edge("hidden", "result", HIDDEN),
    ]
    fields = {"last_hidden_state": [["hidden", "output"]]}
    if multiple:
        other = Node("other", "Other", "operation", inputs=[IDS], outputs=[OTHER])
        child_nodes.insert(-1, other)
        child_output.inputs.append(OTHER)
        child_output.outputs.append(OTHER)
        child_edges.extend([Edge("tokens", "other", IDS), Edge("other", "result", OTHER)])
        fields["other_field"] = [["other", "output"]]
    child = Graph("child", child_nodes, child_edges, {"output_fields": fields})
    source = Node("source", "Input", "input", outputs=[IDS])
    backbone = Node(
        "transformer", "Decoder", "decoder", module_path="transformer", inputs=[IDS],
        outputs=[UNKNOWN], child_graph=child,
    )
    head = Node("head", "Head", "lm_head", inputs=[UNKNOWN], outputs=[UNKNOWN])
    output = Node("output", "Output", "output", inputs=[UNKNOWN], outputs=[UNKNOWN])
    return Graph(
        "parent", [source, backbone, head, output],
        [
            Edge("source", "transformer", IDS, target_port="input_ids"),
            Edge("transformer", "head", UNKNOWN, source_port=parent_port),
            Edge("head", "output", UNKNOWN),
        ],
        {"output_fields": {"logits": [["head", "output"]]}},
    )


def _flatten(graph: Graph) -> None:
    builder = object.__new__(HierarchyBuilder)
    builder.model = SimpleNamespace(base_model_prefix="transformer")
    builder.flatten_decoder_backbone(graph)


def test_named_child_field_rewires_port_and_recomputes_target_inputs() -> None:
    graph = _graph()
    graph.validate()

    _flatten(graph)

    graph.validate()
    assert all(node.id != "transformer" for node in graph.nodes)
    assert graph.metadata["backbone_path"] == "transformer"
    edge = next(edge for edge in graph.edges if edge.target == "head")
    assert (edge.source, edge.source_port, edge.tensor) == ("hidden", "output", HIDDEN)
    assert next(node for node in graph.nodes if node.id == "head").inputs == [HIDDEN]
    assert next(node for node in graph.nodes if node.id == "head").outputs == [UNKNOWN]


def test_multiple_child_outputs_use_exact_named_field() -> None:
    graph = _graph(multiple=True)

    _flatten(graph)

    graph.validate()
    edge = next(edge for edge in graph.edges if edge.target == "head")
    assert (edge.source, edge.tensor) == ("hidden", HIDDEN)
    assert edge.source != "other"


def test_unmapped_numeric_port_preserves_backbone_and_child_graph() -> None:
    graph = _graph(parent_port="0", multiple=True)
    before_nodes = list(graph.nodes)
    before_edges = list(graph.edges)

    _flatten(graph)

    graph.validate()
    assert graph.nodes == before_nodes
    assert graph.edges == before_edges
    assert "backbone_path" not in graph.metadata


def test_input_and_output_source_ports_survive_inlining_with_id_collision() -> None:
    graph = _graph()
    source = graph.nodes[0]
    source.outputs = [IDS]
    incoming = graph.edges[0]
    incoming.source_port = "tokens"
    collision = Node("hidden", "Existing", "input", outputs=[IDS])
    graph.nodes.insert(1, collision)

    _flatten(graph)

    graph.validate()
    input_edge = next(edge for edge in graph.edges if edge.target == "transformer_hidden")
    output_edge = next(edge for edge in graph.edges if edge.target == "head")
    assert (input_edge.source, input_edge.source_port, input_edge.tensor) == ("source", "tokens", IDS)
    assert (output_edge.source, output_edge.source_port, output_edge.tensor) == (
        "transformer_hidden", "output", HIDDEN,
    )
