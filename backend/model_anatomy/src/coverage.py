"""Source and invocation completeness checks, independent of shape/classification metrics."""

from __future__ import annotations

from typing import Any

from .ir import Graph


def source_completeness(root: Graph, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Measure inspected source sites, never runtime call counts or universal reachability."""
    expected: dict[tuple, dict[str, Any]] = {}
    established: set[tuple] = set()
    excluded: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    warnings: list[str] = []

    def key(call: dict[str, Any]) -> tuple:
        return (call.get("source"), call.get("source_sha256"), call.get("line"), call.get("column"))

    for path, report in reports.items():
        for call in report.get("source_call_sites", []):
            if call.get("excluded_by_profile"):
                excluded.append(call)
            else:
                expected[key(call)] = call
        for call in report.get("call_sites", []):
            expected[key(call)] = call
            established.add(key(call))
        operations.extend(dict(item, owner_path=path) for item in report.get("unexplained_operations", []))
    missing = [call for identifier, call in expected.items() if identifier not in established]
    reachable = {call["module_path"] for call in expected.values()}
    called = {expected[identifier]["module_path"] for identifier in established}
    missing_invocations = []
    for graph in root.walk():
        context = graph.metadata.get("module_path", graph.id) or "<root>"
        messages = list(graph.metadata.get("warnings", []))
        if graph.metadata.get("warning"):
            messages.append(graph.metadata["warning"])
        warnings.extend(f"{context}: {message}" for message in messages)
        for node in graph.nodes:
            if node.metadata.get("invocation_not_established") and not node.metadata.get("excluded_by_profile"):
                missing_invocations.append({"graph_id": graph.id, "node_id": node.id,
                                            "module_path": node.module_path,
                                            "reason": "Registered structure has no established invocation in this source graph"})
    if missing:
        warnings.append(f"{len(missing)} source module call sites have no established invocation")
    if operations:
        warnings.append(f"{len(operations)} source operations remain unexplained")
    if missing_invocations:
        warnings.append(f"{len(missing_invocations)} structural nodes have no established invocation")
    return {
        "scope": "Unique module call sites in inspected source under the recorded profile; unresolved paths remain possible",
        "runtime_verified": False,
        "representative_only": True,
        "limitations": [
            "Source unavailable or opaque dispatch may hide additional calls; these ratios do not prove whole-model execution coverage",
            "Collapsed instances are represented by inspected templates; semantic equivalence is a separate check",
            "Unexecuted canonical profile branches are excluded only with recorded static condition evidence",
        ],
        "source_call_coverage": len(established) / len(expected) if expected else None,
        "expected_source_calls": len(expected),
        "established_source_calls": len(established),
        "missing_source_calls": missing,
        "reachable_module_coverage": len(called & reachable) / len(reachable) if reachable else None,
        "reachable_module_paths": sorted(reachable),
        "established_module_paths": sorted(called),
        "excluded_call_sites": excluded,
        "unexplained_source_operations": operations,
        "unexplained_source_operation_count": len(operations),
        "structural_invocations_not_established": missing_invocations,
        "warnings": list(dict.fromkeys(warnings)),
    }
