"""Annotate interpreted operation graphs; semantic labels never select extraction paths."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .attention import build_attention
from .ir import Evidence, Graph
from .patterns import SemanticClassification
from .normalized_flow import normalized_flow


def annotate_graphs(builder: Any, root: Graph) -> dict[str, Any]:
    roles: dict[str, list[SemanticClassification]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for graph in reversed(list(root.walk())):
        path = graph.metadata.get("module_path", "")
        nodes = {node.id: node for node in graph.nodes}
        parents: dict[str, list[str]] = {identifier: [] for identifier in nodes}
        for edge in graph.edges:
            if edge.relation == "dataflow":
                parents[edge.target].append(edge.source)
        returned, pending = set(), [reference for refs in graph.metadata.get("return_bindings", {}).values()
                                   for reference, _ in refs]
        while pending:
            identifier = pending.pop()
            if identifier not in returned:
                returned.add(identifier)
                pending.extend(parents.get(identifier, []))
        def primitive(node: Any, kind: str) -> bool:
            return node.kind == kind or node.metadata.get("source_semantic_kind") == kind

        matrices = [node for node in nodes.values() if node.metadata.get("shape_check", {}).get("rule") in {"matmul_contract", "bmm_contract"}]
        normalized = [node for node in nodes.values() if node.metadata.get("native_callable") == "torch.nn.functional.softmax"]
        kind, reasons = "unknown", []
        if len(matrices) >= 2 and normalized:
            build_attention(builder, path, graph)
            proof = graph.metadata["attention_proof"]
            if proof["attention_pattern"] == "source_observed":
                kind, reasons = "attention", ["Recorded score-matrix, last-axis normalization and value-aggregation output lineage"]
            elif not proof["failure_reasons"]:
                # A semantic pattern miss is distinct from an unparsed computation.
                graph.metadata["warnings"] = [message for message in graph.metadata.get("warnings", [])
                    if message != "Attention semantics are unproven; selected source graph retained without a semantic template"]
        normalized = normalized_flow(graph)
        operations = normalized["operations"]
        def lineage(token: str) -> set[str]:
            seen, pending = set(), [token]
            while pending:
                item = pending.pop()
                if item not in seen:
                    seen.add(item)
                    pending.extend(operations.get(item, {}).get("dependencies", []))
            return seen
        projections = [token for token, item in operations.items() if item["node"].metadata.get("shape_check", {}).get("rule") == "native_projection"
                       and item["node"].metadata["shape_check"]["status"] in {"proven", "conditional"}]
        activations = [token for token, item in operations.items() if primitive(item["node"], "activation")
                       and item["node"].metadata.get("native_callable") != "torch.nn.functional.softmax"]
        outputs = set().union(*(lineage(token) for tokens in normalized["outputs"].values() for token in tokens))
        has_attention_math = any(item["node"].metadata.get("shape_check", {}).get("rule") in {"matmul_contract", "bmm_contract"} for item in operations.values())
        if kind == "unknown" and not has_attention_math:
            for final in projections:
                if final not in outputs:
                    continue
                candidates = lineage(final) - {final}
                if any(activation in candidates and any(first in lineage(activation) for first in projections if first != final)
                       for activation in activations):
                    kind, reasons = "feed_forward", ["Actual affine-to-activation-to-affine dependency reaches the returned output"]
                    break
        if kind == "unknown" and len(projections) == 1 and projections[0] in outputs and not has_attention_math:
            kind, reasons = "projection_block", ["A native affine projection is composed with interpreted source operations"]
            for token in lineage(projections[0]):
                selection = operations.get(token, {}).get("node")
                descriptors = selection.metadata.get("indexing", {}).get("indices", []) if selection else []
                if len(descriptors) == 2 and descriptors[0].get("kind") == "slice" and all(descriptors[0].get(key) is None for key in ("start", "stop", "step")) and descriptors[1].get("kind") == "integer":
                    kind, reasons = "selection_projection", ["Actual second-axis selection feeds the native affine output projection"]
        if kind == "unknown" and len(normalized["outputs"]) > 1 and not has_attention_math:
            terminals = [next(iter(tokens)) for tokens in normalized["outputs"].values() if len(tokens) == 1]
            if (len(terminals) == len(normalized["outputs"]) and len(set(terminals)) == len(terminals)
                    and all(token in projections for token in terminals)
                    and all(other not in lineage(token) for token in terminals for other in terminals if other != token)):
                kind, reasons = "output_head_group", ["Distinct returned affine outputs have independent source branches"]
        nested = {node.child_graph.metadata["source_semantics"]["kind"] if node.child_graph and node.child_graph.metadata["source_semantics"]["kind"] != "unknown" else node.kind
                  for node in nodes.values() if node.id in returned}
        if kind == "unknown" and "attention" in nested and not activations:
            kind, reasons = "attention", ["An interpreted attention child feeds the returned projection/shape wrapper"]
        if "attention" in nested and nested & {"feed_forward", "gated_mlp"} and nested & {"layer_norm", "rms_norm", "normalization"} and "residual" in nested:
            kind, reasons = "transformer_block", ["Source-established attention, feed-forward, normalization and skip dependencies"]
        if kind == "unknown" and "transformer_block" in nested:
            kind, reasons = "transformer_stack", ["Actual source calls compose interpreted transformer blocks"]
        failures = list(graph.metadata.get("warnings", []))
        if graph.metadata.get("warning"):
            failures.append(graph.metadata["warning"])
        for node in nodes.values():
            if node.metadata.get("excluded_by_profile") or node.metadata.get("inventory_only"):
                node.metadata["unknown_category"] = "profile_excluded"
                continue
            if node.metadata.get("unsupported") or node.metadata.get("invocation_not_established"):
                failures.append(f"{node.id}: source operation or invocation is unproven")
            check = node.metadata.get("shape_check")
            if check and check["status"] in {"unknown", "invalid"}:
                failures.append(f"{node.id}: {check.get('reason', 'operation contract is unproven')}")
            if node.child_graph and summaries[node.child_graph.id]["status"] != "complete":
                failures.append(f"{node.id}: child source interpretation is incomplete")
            if node.kind == "unknown" and not node.child_graph and not check and not node.metadata.get("source_proof"):
                failures.append(f"{node.id}: computation is opaque")
        interpretation = {"status": "partial" if failures else "complete", "failure_reasons": sorted(set(failures)),
                          "scope": "Interpreted source and recorded input profile; not runtime execution"}
        summaries[graph.id] = interpretation
        graph.metadata["source_interpretation"] = interpretation
        if failures:
            kind, reasons = "unknown", ["Source interpretation remains incomplete; candidate semantic patterns are not asserted"]
        classification = SemanticClassification(kind, 0.98 if kind != "unknown" else 0.0,
                                               reasons or ["Computation is recorded independently; a semantic role is not established"])
        graph.metadata["source_semantics"] = asdict(classification)
        roles.setdefault(path, []).append(classification)
        for node in nodes.values():
            if node.child_graph:
                child = node.child_graph.metadata["source_semantics"]
                node.metadata["source_interpretation"] = summaries[node.child_graph.id]
                node.metadata["source_semantic_kind"] = child["kind"]
                if child["kind"] != "unknown":
                    node.kind, node.confidence = child["kind"], child["confidence"]
                    node.label = {"attention": "Attention", "feed_forward": "Feed Forward / MLP", "transformer_block": "Transformer Block",
                                  "transformer_stack": "Transformer Stack", "projection_block": "Projection Block",
                                  "selection_projection": "Select / Project", "output_head_group": "Output Heads"}.get(node.kind, node.label)
                    node.metadata["classification"] = child
                elif node.kind == "unknown":
                    node.metadata["unknown_category"] = "unclassified_role" if summaries[node.child_graph.id]["status"] == "complete" else "unsupported_source"
                node.evidence.append(Evidence("python_source", "derived", "source_graph_interpretation",
                    {"graph_id": node.child_graph.id, "interpretation": summaries[node.child_graph.id], "semantic_role": child}))
    for path, candidates in roles.items():
        established = {candidate.kind for candidate in candidates if candidate.kind != "unknown"}
        if len(established) == 1 and all(candidate.kind != "unknown" for candidate in candidates) and path:
            builder.analyzer.classifications[path] = candidates[0]
    return {"rule": "source_graph_interpretation", "graphs": summaries,
            "status": summaries[root.id]["status"], "scope": summaries[root.id]["scope"]}
