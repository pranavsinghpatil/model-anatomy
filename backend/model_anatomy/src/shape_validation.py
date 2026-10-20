"""Audit tensor contracts independently of graph coverage and semantic classification."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from .ir import Evidence, Graph
from .shape_rules import known, module_contract, broadcast, matmul, transform, selection, concatenate, tensor_compare, tensor_mask_method


def validate_shapes(root: Graph, model: torch.nn.Module) -> dict[str, Any]:
    """Check native module contracts against actual edges without executing any module."""
    checks: list[dict[str, Any]] = []
    patterns: list[dict[str, Any]] = []
    unknown_edges: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    checks_by_node: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def record(graph: Graph, node_id: str, check: dict[str, Any], basis: str) -> None:
        item = dict(check, graph_id=graph.id, node_id=node_id, basis=basis)
        checks.append(item)
        checks_by_node.setdefault((graph.id, node_id), []).append(item)

    for graph in root.walk():
        nodes = {node.id: node for node in graph.nodes}
        incoming = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            if edge.relation == "dataflow":
                incoming[edge.target].append(edge)
        if graph.metadata.get("return_contract"):
            patterns.append(dict(graph.metadata["return_contract"], graph_id=graph.id))
        for node in graph.nodes:
            if (node.child_graph is None and node.kind not in {"input", "output", "parameter", "buffer"}
                    and not node.metadata.get("invocation_not_established")):
                operations.append({"graph_id": graph.id, "node_id": node.id, "kind": node.kind})
            if node.metadata.get("shape_pattern_boundary"):
                patterns.append(dict(node.metadata["shape_pattern_boundary"], graph_id=graph.id, node_id=node.id))
            if node.metadata.get("shape_check"):
                check = node.metadata["shape_check"]
                record(graph, node.id, check, "Bounded AST operation under recorded caller inputs")
                node.evidence.append(Evidence("static_shape_contract", "derived", check["rule"],
                    {**check, "source": node.metadata.get("source_location"),
                     "source_fingerprint": node.metadata.get("source_location", {}).get("source_sha256")
                        or next((item.details.get("source_fingerprint") for item in node.evidence if item.details.get("source_fingerprint")), None),
                     "runtime_verified": False}))
                specs = [edge.tensor for edge in incoming[node.id] if edge.tensor is not None]
                recomputed = None
                if check["rule"] in {"matmul_contract", "bmm_contract"} and len(specs) == 2:
                    recomputed = matmul(*specs, bmm=check["rule"] == "bmm_contract")
                elif check["rule"].startswith("elementwise_"):
                    recomputed = broadcast(specs, check["rule"])
                elif check["rule"] == "native_gather":
                    source = [edge.tensor for edge in incoming[node.id] if edge.target_port == "input" and edge.tensor]
                    indices = [edge.tensor for edge in incoming[node.id] if edge.target_port == "indices" and edge.tensor]
                    if len(source) == len(indices) == 1:
                        from .shape_rules import gather
                        recomputed = gather(source[0], indices[0], node.metadata.get("axis"))
                elif check["rule"].startswith("tensor_scalar_") and len(specs) == 1:
                    from .shape_rules import tensor_scalar
                    recomputed = tensor_scalar(specs[0], node.metadata["operator"], node.metadata["scalar"],
                                               node.metadata["scalar_side"])
                elif check["rule"] == "tensor_comparison" and specs:
                    recomputed = tensor_compare(specs)
                elif check["rule"] in {"tensor_expand_as", "tensor_masked_fill"} and len(specs) >= 2:
                    recomputed = tensor_mask_method(specs[0], node.metadata["method"], specs[1],
                                                   specs[2] if node.metadata.get("fill_is_tensor") and len(specs) == 3 else None)
                elif node.kind == "reshape" and node.metadata.get("method") not in {"to", "type", "type_as", "float", "half", "expand_as"} and specs and not node.metadata.get("keyword_arguments"):
                    recomputed = transform(specs[0], node.metadata["method"], node.metadata.get("dimensions", []))
                elif check["rule"] == "tensor_indexing" and specs:
                    descriptors = node.metadata["indexing"]["indices"]
                    if len(descriptors) == 1 and descriptors[0].get("kind") == "integer_tensor":
                        indexed = [edge.tensor for edge in incoming[node.id] if edge.target_port == "input" and edge.tensor is not None]
                        indices = [edge.tensor for edge in incoming[node.id] if edge.target_port == "indices" and edge.tensor is not None]
                        if len(indexed) == len(indices) == 1:
                            recomputed = selection(indexed[0], [{"kind": "integer_tensor", "shape": indices[0].shape, "dtype": indices[0].dtype}])
                        else:
                            recomputed = selection(specs[0], [{"kind": "unknown"}])
                    else:
                        recomputed = selection(specs[0], descriptors)
                elif check["rule"] in {"concatenate_dimensions", "stack_dimensions"}:
                    recomputed = concatenate(specs, node.metadata.get("dimension"), stack=check["rule"] == "stack_dimensions")
                if recomputed is not None:
                    record(graph, node.id, recomputed.record(), "Independent operation contract from actual input edges")
                    if recomputed.status in {"proven", "conditional"} and node.outputs:
                        output = node.outputs[0]
                        if output.shape != recomputed.spec.shape or output.dtype != recomputed.spec.dtype:
                            record(graph, node.id, {"rule": "operation_output_agreement", "status": "invalid",
                                "reason": "Exported output disagrees with independently checked operation",
                                "expected_shape": recomputed.spec.shape, "actual_shape": output.shape}, "Operation output")
                if check["rule"] == "native_tensor_split" and specs and known(specs[0].shape):
                    axis, sizes = node.metadata["axis"], node.metadata["sizes"]
                    valid = sum(sizes) == specs[0].shape[axis]
                    for index, size in enumerate(sizes):
                        expected = specs[0].shape[:axis] + [size] + specs[0].shape[axis + 1:]
                        declared = node.metadata["output_ports"].get(str(index), {})
                        valid = valid and declared.get("shape") == expected and declared.get("dtype") == specs[0].dtype
                    record(graph, node.id, {"rule": "split_output_agreement", "status": "proven" if valid else "invalid",
                        "reason": "" if valid else "Split outputs disagree with input extent or sections"}, "Independent split contract")
            elif node.metadata.get("source_proof"):
                constraints = [item["condition"] for item in node.metadata["source_proof"]["conditions"] if item.get("basis") == "required_input_constraint"]
                record(graph, node.id, {"rule": "bounded_python_source", "status": "conditional" if constraints else "proven",
                    "constraints": constraints,
                    "runtime_verified": False, "sources": node.metadata["source_proof"]["sources"],
                    "conditions": node.metadata["source_proof"]["conditions"]}, "Bounded external helper AST")
            if node.metadata.get("repetition_shape_contract"):
                patterns.append(dict(node.metadata["repetition_shape_contract"], graph_id=graph.id, node_id=node.id))
                if node.metadata["repetition_shape_contract"]["status"] == "unknown":
                    record(graph, node.id, node.metadata["repetition_shape_contract"], "Unresolved repeated shape recurrence")
            if node.module_path is not None and incoming[node.id]:
                module = model.get_submodule(node.module_path) if node.module_path else model
                check = module_contract(module, incoming[node.id][0].tensor)
                if check is not None:
                    record(graph, node.id, check.record(), "Independent native weight/input edge contract")
                    if check.status in {"proven", "conditional"} and node.outputs:
                        output = node.outputs[0]
                        if output.shape != check.spec.shape or output.dtype != check.spec.dtype:
                            record(graph, node.id, {"rule": "native_output_agreement", "status": "invalid",
                                "reason": "Exported output disagrees with independently checked native contract",
                                "expected_shape": check.spec.shape, "actual_shape": output.shape,
                                "expected_dtype": check.spec.dtype, "actual_dtype": output.dtype}, "Native module output")
            if node.child_graph is not None:
                inputs = {item.metadata.get("argument_name"): item for item in node.child_graph.nodes if item.kind == "input"}
                for edge in incoming[node.id]:
                    role = node.metadata.get("argument_roles", {}).get(edge.target_port, edge.target_port)
                    child = inputs.get(role)
                    if child is not None and child.outputs and edge.tensor is not None:
                        expected, actual = child.outputs[0], edge.tensor
                        established = known(expected.shape) and known(actual.shape)
                        actual_shape = list(actual.shape) if actual.shape is not None else None
                        constraints = []
                        for proof in child.metadata.get("input_constraints", []):
                            if (actual_shape is not None and proof.get("rule") == "normal_path_input_precondition"
                                    and type(proof.get("axis")) is int and -len(actual_shape) <= proof["axis"] < len(actual_shape)
                                    and actual_shape[proof["axis"]] == proof.get("symbol")):
                                actual_shape = [proof["value"] if dimension == proof["symbol"] else dimension for dimension in actual_shape]
                                constraints.append(proof["constraint"])
                        same = expected.shape == actual_shape and expected.dtype == actual.dtype
                        record(graph, node.id, {"rule": "module_input_boundary", "status": "conditional" if established and same and constraints else "proven" if established and same else "invalid" if established else "unknown",
                            "reason": "" if same else "Child input contract differs from caller input or is unproven",
                            "constraints": constraints, "argument": role, "caller_shape": actual.shape, "child_shape": expected.shape}, "Caller-to-child boundary")
        for edge in graph.edges:
            if edge.relation != "dataflow":
                continue
            source = nodes[edge.source]
            explicit = source.metadata.get("output_ports")
            if explicit is not None:
                declared = explicit.get(edge.source_port)
                if declared is None:
                    unproven = source.metadata.get("output_structure", {}).get("kind") == "unknown"
                    record(graph, source.id, {"rule": "source_port_exists", "status": "unknown" if unproven else "invalid",
                        "reason": "Output container is unresolved" if unproven else "Data edge references an absent typed output port", "port": edge.source_port}, "Typed source return")
                elif edge.tensor is not None and (declared["shape"] != edge.tensor.shape or declared["dtype"] != edge.tensor.dtype):
                    record(graph, source.id, {"rule": "source_port_agreement", "status": "invalid",
                        "reason": "Data edge disagrees with its typed output port", "port": edge.source_port}, "Typed source return")
            if edge.tensor is None or not known(edge.tensor.shape):
                unknown_edges.append({"graph_id": graph.id, "source": edge.source, "source_port": edge.source_port,
                                      "target": edge.target, "target_port": edge.target_port})
            proof = source.metadata.get("shape_check")
            if proof:
                edge.evidence.append(Evidence("static_shape_contract", "derived", proof["rule"],
                    {"status": proof["status"], "constraints": proof.get("constraints", []),
                     "source": source.metadata.get("source_location"), "runtime_verified": False}))
    counts = Counter(check["status"] for check in checks)
    constraints = sorted({condition for check in checks for condition in check.get("constraints", [])})
    unchecked = []
    for operation in operations:
        related = checks_by_node.get((operation["graph_id"], operation["node_id"]), [])
        if not related or any(check["status"] in {"invalid", "unknown"} for check in related):
            unchecked.append(dict(operation, reason="No established bounded operation contract, or a check remains invalid/unknown"))
    warnings = []
    for status in ("invalid", "unknown"):
        if counts[status]:
            warnings.append(f"{counts[status]} tensor shape contract checks are {status}; inspect analysis_report.shape_validation")
    return {
        "scope": "Static rank, dimension and native dtype contracts under recorded input profile; no tensor execution",
        "runtime_verified": False, "checks": checks,
        "counts": {status: counts[status] for status in ("proven", "conditional", "invalid", "unknown")},
        "required_input_constraints": constraints,
        "scoped_input_constraints": [{"graph_id": check["graph_id"], "node_id": check["node_id"],
            "constraints": check["constraints"]} for check in checks if check.get("constraints")],
        "operation_validation_coverage": (len(operations) - len(unchecked)) / len(operations) if operations else None,
        "unvalidated_operations": unchecked,
        "unknown_edges": unknown_edges,
        "semantic_pattern_boundaries": patterns,
        "limitations": ["Input bounds and symbolic nonzero requirements are obligations, not runtime observations",
                       "Attention/RoPE semantic equivalence and repeated-template equivalence remain separate checks",
                       "Known edge shapes do not imply that every operation or branch has been validated"],
        "warnings": warnings,
    }
