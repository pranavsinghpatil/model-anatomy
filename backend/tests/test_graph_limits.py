"""Nested graphs have bounded, stable traversal before serialization."""

import pytest

from model_anatomy.src import limits
from model_anatomy.src.errors import AnatomyError
from model_anatomy.src.ir import Edge, Graph, Node


def nested_graph():
    left = Graph("left", [Node("item", "Item", "unknown")], [])
    right = Graph("right", [], [])
    root = Graph(
        "root",
        [Node("a", "A", "unknown", child_graph=left), Node("b", "B", "unknown", child_graph=right)],
        [],
    )
    return root


def test_stable_graph_preorder_at_budget_boundary(monkeypatch):
    monkeypatch.setattr(limits, "MAX_GRAPHS", 3)
    monkeypatch.setattr(limits, "MAX_GRAPH_NODES", 3)
    graph = nested_graph()
    assert [item.id for item in graph.walk()] == ["root", "left", "right"]
    graph.validate()


@pytest.mark.parametrize(
    "name,value", [("MAX_GRAPHS", 2), ("MAX_GRAPH_DEPTH", 0), ("MAX_GRAPH_NODES", 2), ("MAX_GRAPH_EDGES", 0)]
)
def test_graph_budgets_fail_explicitly(monkeypatch, name, value):
    monkeypatch.setattr(limits, name, value)
    graph = nested_graph()
    graph.edges.append(Edge("a", "b", None, relation="containment"))
    with pytest.raises(AnatomyError, match="metadata budget"):
        graph.validate()


def test_cycles_fail_without_recursing_indefinitely():
    graph = nested_graph()
    graph.nodes[0].child_graph.nodes[0].child_graph = graph
    with pytest.raises(AnatomyError, match="cycle"):
        graph.validate()
