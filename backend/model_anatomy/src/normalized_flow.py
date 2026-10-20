"""Resolve operation dependencies through proven module argument and return ports."""

from __future__ import annotations

from collections import deque
from typing import Any

from .ir import Graph


def normalized_flow(root: Graph) -> dict[str, Any]:
    operations: dict[str, Any] = {}

    def flatten(graph: Graph, supplied: dict[str, set[str]]) -> dict[str, set[str]]:
        nodes = {node.id: node for node in graph.nodes}
        incoming = {identifier: [] for identifier in nodes}
        following = {identifier: [] for identifier in nodes}
        counts = {identifier: 0 for identifier in nodes}
        for edge in graph.edges:
            if edge.relation == "dataflow":
                incoming[edge.target].append(edge)
                following[edge.source].append(edge.target)
                counts[edge.target] += 1
        ready = deque(identifier for identifier in nodes if counts[identifier] == 0)
        values: dict[tuple[str, str], set[str]] = {}
        while ready:
            identifier = ready.popleft()
            node = nodes[identifier]
            dependencies = set().union(*(values.get((edge.source, edge.source_port), set()) for edge in incoming[identifier]))
            if node.kind == "input":
                outputs = {"output": supplied.get(node.metadata.get("argument_name"), {f"{graph.id}/{identifier}"})}
            elif node.child_graph is not None and not node.metadata.get("excluded_by_profile"):
                bindings: dict[str, set[str]] = {}
                for edge in incoming[identifier]:
                    role = node.metadata.get("argument_roles", {}).get(edge.target_port, edge.target_port)
                    bindings.setdefault(role, set()).update(values.get((edge.source, edge.source_port), set()))
                outputs = flatten(node.child_graph, bindings)
            elif node.kind in {"output", "parameter", "buffer"} or node.metadata.get("excluded_by_profile") or node.metadata.get("inventory_only"):
                outputs = {"output": dependencies}
            else:
                token = f"{graph.id}/{identifier}"
                operations[token] = {"node": node, "dependencies": dependencies, "graph_id": graph.id}
                outputs = {port: {token} for port in node.metadata.get("output_ports", {"output": None})}
            for port, tokens in outputs.items():
                values[identifier, port] = tokens
            for target in following[identifier]:
                counts[target] -= 1
                if counts[target] == 0:
                    ready.append(target)
        return {port: set().union(*(values.get((reference, source_port), set()) for reference, source_port in refs))
                for port, refs in graph.metadata.get("return_bindings", {}).items()}

    outputs = flatten(root, {})
    return {"operations": operations, "outputs": outputs}
