"""Serializable graph-of-graphs IR, independent of extraction and rendering."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .errors import AnatomyError

Dimension = int | str


@dataclass
class Evidence:
    source: str
    kind: Literal["direct", "derived", "reconstructed", "unknown"]
    locator: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorSpec:
    shape: list[Dimension]
    dtype: str
    semantic_name: str


@dataclass
class Edge:
    source: str
    target: str
    tensor: TensorSpec
    source_port: str = "output"
    target_port: str = "input"
    label: str | None = None
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Node:
    id: str
    label: str
    kind: str
    module_path: str | None = None
    module_type: str | None = None
    repeat: int = 1
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    parameter_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    child_graph: Graph | None = None


@dataclass
class Graph:
    id: str
    nodes: list[Node]
    edges: list[Edge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator[Graph]:
        """Yield this graph and each nested graph in stable preorder."""
        yield self
        for node in self.nodes:
            if node.child_graph is not None:
                yield from node.child_graph.walk()

    def validate(self) -> None:
        graph_ids: set[str] = set()
        for graph in self.walk():
            if graph.id in graph_ids:
                raise AnatomyError(f"Duplicate graph ID: {graph.id}")
            graph_ids.add(graph.id)
            nodes = {node.id: node for node in graph.nodes}
            if len(nodes) != len(graph.nodes):
                raise AnatomyError(f"Duplicate node ID in {graph.id}")
            for node in graph.nodes:
                if node.repeat < 1 or node.parameter_count < 0:
                    raise AnatomyError(f"Invalid count on {graph.id}/{node.id}")
            for edge in graph.edges:
                if edge.source not in nodes or edge.target not in nodes:
                    raise AnatomyError(f"Dangling edge in {graph.id}: {edge.source} -> {edge.target}")
                source, target = nodes[edge.source], nodes[edge.target]
                if not any(
                    t.shape == edge.tensor.shape and t.dtype == edge.tensor.dtype for t in source.outputs
                ):
                    raise AnatomyError(f"Edge disagrees with source output: {graph.id}/{source.id}")
                if not any(
                    t.shape == edge.tensor.shape and t.dtype == edge.tensor.dtype for t in target.inputs
                ):
                    raise AnatomyError(f"Edge disagrees with target input: {graph.id}/{target.id}")


@dataclass
class ModelIR:
    model_id: str
    architecture: str
    root_graph: Graph
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_structure: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        self.root_graph.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, value: str) -> ModelIR:
        data = json.loads(value)
        if data.get("schema_version") != "1.0":
            raise AnatomyError("Unsupported IR schema version")

        def graph_from_dict(item: dict[str, Any]) -> Graph:
            nodes = []
            for original in item["nodes"]:
                node = dict(original)
                node["inputs"] = [TensorSpec(**t) for t in node["inputs"]]
                node["outputs"] = [TensorSpec(**t) for t in node["outputs"]]
                node["evidence"] = [Evidence(**e) for e in node["evidence"]]
                if node["child_graph"] is not None:
                    node["child_graph"] = graph_from_dict(node["child_graph"])
                nodes.append(Node(**node))
            edges = []
            for original in item["edges"]:
                edge = dict(original)
                edge["tensor"] = TensorSpec(**edge["tensor"])
                edge["evidence"] = [Evidence(**e) for e in edge["evidence"]]
                edges.append(Edge(**edge))
            return Graph(item["id"], nodes, edges, item["metadata"])

        data["root_graph"] = graph_from_dict(data["root_graph"])
        data["evidence"] = [Evidence(**e) for e in data["evidence"]]
        model = cls(**data)
        model.root_graph.validate()
        return model
