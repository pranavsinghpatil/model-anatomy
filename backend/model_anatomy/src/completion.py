"""Separate profile computation completeness, artifact inventory and semantic annotation."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

from .ir import Graph


def completion_report(root: Graph, raw: dict[str, Any], interpretation: dict[str, Any],
                      source: dict[str, Any], shapes: dict[str, Any]) -> dict[str, Any]:
    active = {graph.metadata.get("module_path", "") for graph in root.walk()}
    active.update(item["module_path"] for report in raw["static_flow_reports"].values() for item in report.get("call_sites", []))
    for path in list(active):
        while path:
            path = path.rpartition(".")[0]
            active.add(path)
    excluded_evidence: dict[str, list[dict[str, Any]]] = {}
    for item in source["excluded_call_sites"]:
        excluded_evidence.setdefault(item["module_path"], []).append(item)
    inventory_only = {node.module_path: node.metadata["inventory_only"]
                      for graph in root.walk() for node in graph.nodes if node.metadata.get("inventory_only")}
    inventory = {}
    annotations = raw["semantic_diagnostics"]["classifications"]
    for path in raw["modules"]:
        exclusions = [item for parent, items in excluded_evidence.items()
                      if path == parent or path.startswith(parent + ".") for item in items]
        inventory[path] = {"reachability": "active" if path in active else "excluded_by_recorded_profile" if exclusions else "not_invoked_in_recorded_profile" if path in inventory_only else "not_established",
                           "profile_evidence": exclusions, "inventory_evidence": inventory_only.get(path), "semantic_role": annotations[path]["kind"]}
    active_unknown = [path for path in active if path in annotations and annotations[path]["kind"] == "unknown"]
    missing = [name for name, item in raw["parameters"].items() if not item["checkpoint_matched"]]
    unmapped = [name for name, item in raw["checkpoint_extras"].items() if item["kind"] == "unmapped"]
    checks = {
        "parameter_mapping": not missing and not unmapped and not raw["warnings"],
        "source_interpretation": interpretation["status"] == "complete" and source["unexplained_source_operation_count"] == 0,
        "source_invocations": not source["missing_source_calls"] and not source["structural_invocations_not_established"],
        "operation_shapes": not shapes["counts"].get("unknown", 0) and not shapes["counts"].get("invalid", 0) and not shapes["unvalidated_operations"],
        "profile_recorded": all("analysis_scope" in graph.metadata or graph.metadata.get("topology") == "nn.Sequential contract" for graph in root.walk()),
    }
    causes, propagation = {}, []
    for graph in reversed(list(root.walk())):
        node_causes: dict[str, set[str]] = {}
        nodes = {node.id: node for node in graph.nodes}
        dependencies: dict[str, list[str]] = {identifier: [] for identifier in nodes}
        following: dict[str, list[str]] = {identifier: [] for identifier in nodes}
        for edge in graph.edges:
            if edge.relation == "dataflow":
                dependencies[edge.target].append(edge.source)
                following[edge.source].append(edge.target)
        remaining = {identifier: len(parents) for identifier, parents in dependencies.items()}
        ready = deque(identifier for identifier, count in remaining.items() if count == 0)
        while ready:
            node = nodes[ready.popleft()]
            inherited = set().union(*(node_causes[parent] for parent in dependencies[node.id]))
            if node.child_graph:
                inherited.update(node.child_graph.metadata.get("unknown_cause_ids", []))
            detail = node.metadata.get("unsupported")
            check = node.metadata.get("shape_check", {})
            if detail or check.get("status") in {"unknown", "invalid"}:
                if detail or not inherited or check.get("status") == "invalid":
                    location = node.metadata.get("source_location", {})
                    reason = detail.get("reason") if detail else check.get("reason")
                    identity = {"source": location, "reason": reason, "rule": detail.get("construct") if detail else check.get("rule")}
                    identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
                    causes[identifier] = dict(identity, id=identifier, graph_id=graph.id, node_id=node.id)
                    inherited.add(identifier)
            node_causes[node.id] = inherited
            if inherited:
                node.metadata["unknown_cause_ids"] = sorted(inherited)
                propagation.append({"graph_id": graph.id, "node_id": node.id, "cause_ids": sorted(inherited)})
            for target in following[node.id]:
                remaining[target] -= 1
                if remaining[target] == 0:
                    ready.append(target)
        graph.metadata["unknown_cause_ids"] = sorted(set().union(*node_causes.values()))
    return {
        "version": 1, "status": "complete" if all(checks.values()) else "partial", "checks": checks,
        "scope": "Canonical recorded input profile; excludes proven inactive branches, without claiming all runtime variants",
        "runtime_verified": False, "conditional_input_constraints": shapes["required_input_constraints"],
        "module_inventory": inventory,
        "semantic_annotations": {"registered_unknown_paths": raw["semantic_diagnostics"]["unknown_components"],
            "active_unknown_paths": sorted(active_unknown),
            "active_coverage": 1 - len(active_unknown) / len(active) if active else None,
            "scope": "Role annotation is independent of computation interpretation"},
        "unknown_causes": {"roots": list(causes.values()), "propagation": propagation},
    }
