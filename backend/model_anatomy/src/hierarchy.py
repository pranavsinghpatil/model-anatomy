"""Finite architecture graphs whose nodes are actual registered PyTorch modules."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from .ir import Edge, Evidence, Graph, Node
from .module_topology import ModuleTopology
from .patterns import SemanticAnalyzer


class HierarchyBuilder:
    """Preserve real instances and stop at basic modules, never expressions."""

    def __init__(self, model: torch.nn.Module, raw: dict[str, Any], analyzer: SemanticAnalyzer):
        self.model, self.raw, self.analyzer = model, raw, analyzer
        self.topology = ModuleTopology(model, raw, analyzer)
        self.represented: set[str] = set()
        self.covered_by_leaf: dict[str, str] = {}

    @staticmethod
    def identity(prefix: str, path: str) -> str:
        return prefix + hashlib.sha256(path.encode()).hexdigest()[:20]

    def build(self) -> Graph:
        self.topology.inspect()
        self.represented.add("")
        if self.analyzer.is_leaf(""):
            return Graph("root", [self.module_node("")], [], {"module_path": "", "topology": "single_module"})
        root = self.build_graph("")
        root.id = "root"
        return root

    def module_node(self, path: str) -> Node:
        record = self.raw["modules"][path]
        classification = self.analyzer.classifications[path]
        leaf = self.analyzer.is_leaf(path)
        self.represented.add(path)
        if leaf:
            for child in self.raw["modules"]:
                if child.startswith(path + ".") and child != path:
                    self.covered_by_leaf[child] = path
        parameters = {item["canonical_parameter"]: item["parameter_count"]
                      for name, item in self.raw["parameters"].items()
                      if not path or name.startswith(path + ".")}
        label = record["type"]
        local_name = path.rsplit(".", 1)[-1]
        if local_name.isdigit():
            label += f" [{local_name}]"
        elif local_name and local_name.lower() not in label.lower():
            label = f"{local_name} · {label}"
        metadata: dict[str, Any] = {
            "attributes": record["attributes"], "leaf_boundary": leaf,
            "classification_basis": classification.evidence,
            "parameter_refs": [name for name in self.raw["parameters"] if not path or name.startswith(path + ".")],
            "calls": self.topology.calls.get(path, []),
            "invocation_status": "source_observed" if path in self.topology.calls else "registration_only",
        }
        if record["container"] in {"module_list", "sequential"}:
            children = record["children"]
            types = sorted({self.raw["modules"][child]["type"] for child in children})
            label = f"{local_name or record['type']} · {' / '.join(types)} × {len(children)}" if children else label
            metadata.update(instance_count=len(children), instance_paths=children,
                grouping_basis="Actual registered container; all instances retained, no computation equivalence asserted",
                execution_order="sequential_contract" if record["container"] == "sequential" else "source_evidence_only",
                instance_attributes={child: self.raw["modules"][child]["attributes"] for child in children})
        evidence = [Evidence("native_module_tree", "direct", path or "<root>", {
            "qualified_type": record["qualified_type"], "attributes": record["attributes"],
            "basis": "Registered module identity from meta construction; not runtime execution",
        })]
        return Node(self.identity("m_", path), label, classification.kind,
                    module_path=path, module_type=record["qualified_type"],
                    parameter_count=sum(parameters.values()), metadata=metadata,
                    evidence=evidence, child_graph=None if leaf else self.build_graph(path),
                    confidence=classification.confidence, source_modules=[path])

    def build_graph(self, path: str) -> Graph:
        children = self.raw["modules"][path]["children"]
        nodes = [self.module_node(child) for child in children]
        ids = {node.module_path: node.id for node in nodes}
        edges: dict[tuple[str, str, str], Edge] = {}

        def endpoint(full_path: str) -> str | None:
            return next((ids[child] for child in children if full_path == child or full_path.startswith(child + ".")), None)

        for relationship in self.topology.edges:
            source_path, target_path = relationship["source"], relationship["target"]
            target = endpoint(target_path)
            if target is None:
                continue
            source = endpoint(source_path)
            if source is None and source_path.startswith(f"@{path}:"):
                argument = source_path.split(":", 1)[1]
                source = self.identity("input_", source_path)
                if not any(node.id == source for node in nodes):
                    nodes.insert(0, Node(source, argument, "input", metadata={"argument": argument},
                        evidence=[Evidence("native_forward_signature", "direct", path or "<root>", {"argument": argument})]))
            if source is None or source == target:
                continue
            relation = "conditional_dependency" if relationship.get("conditions") or relationship.get("source_conditions") else "dependency"
            key = source, target, relation
            proof = Evidence("module_call_source", "derived", relationship.get("method", relationship.get("owner", "")),
                             relationship)
            if key in edges:
                if proof not in edges[key].evidence:
                    edges[key].evidence.append(proof)
            else:
                edges[key] = Edge(source, target, None, relation=relation, evidence=[proof],
                                  label="conditional" if relation == "conditional_dependency" else "module dependency")
        report = self.topology.reports.get(path)
        return Graph(self.identity("g_", path), nodes, list(edges.values()), {
            "module_path": path, "topology": "source_backed_module_dependencies" if edges else "registered_module_inventory",
            "scope": "Basic modules are leaves; edges omit intermediate expressions and do not assert tensor identity",
            "registration_order_is_execution_order": False,
            "source_topology": report,
        })
