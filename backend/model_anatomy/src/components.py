"""Discover scoped native components without guessing custom wrapper computation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .artifacts import CheckpointMetadata, MetadataClient, ModelReference
from .errors import ArtifactError, NativeShapeMismatchError, UnsupportedModelError
from .ir import ModelIR

MAX_COMPONENT_CONFIGS = 16
MAX_COMPONENT_CANDIDATES = 16


def discover_components(
    reference: ModelReference, revision: str, files: set[str], checkpoint: CheckpointMetadata,
    client: MetadataClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Require a config directory to identify an exact checkpoint namespace."""
    from .architecture_selection import compatibility
    from .facts import inspect_native_model
    from .registry import _auto_model_candidates, instantiate_candidate, load_native_config

    configs = sorted(name for name in files if name.endswith("/config.json"))
    records, verified = [], []
    if len(configs) > MAX_COMPONENT_CONFIGS:
        return [{"status": "component_budget_exceeded", "config_count": len(configs),
                 "message": "Component config discovery requires an explicit smaller scope"}], []
    for config_file in configs:
        directory = config_file.removesuffix("/config.json")
        parts = directory.split("/")
        if any(not part or part in {".", ".."} or "\\" in part or "\x00" in part for part in parts):
            raise ArtifactError("Invalid component config path")
        namespace = ".".join(parts) + "."
        selected = {name.removeprefix(namespace): replace(tensor, name=name.removeprefix(namespace))
                    for name, tensor in checkpoint.tensors.items() if name.startswith(namespace)}
        record = {"path": directory, "config_file": config_file, "checkpoint_namespace": namespace,
                  "status": "namespace_unavailable", "tensor_count": len(selected),
                  "namespace_basis": "Exact config directory path and checkpoint key prefix; no inferred calls",
                  "candidates": []}
        records.append(record)
        if not selected:
            continue
        data = client.json(reference.url(config_file, revision))
        record["config_canonical_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        record["declared_architectures"] = data.get("architectures")
        try:
            config = load_native_config(data)
        except UnsupportedModelError as exc:
            record.update(status="native_unavailable", message=str(exc)[:1024])
            continue
        scoped = CheckpointMetadata(selected, checkpoint.shards,
                                    {name: checkpoint.tensor_files[namespace + name] for name in selected},
                                    checkpoint.index, checkpoint.format)
        candidates = {item[2]: item for item in _auto_model_candidates(config)}
        if len(candidates) > MAX_COMPONENT_CANDIDATES:
            record["status"] = "candidate_budget_exceeded"
            continue
        matches = []
        search_complete = True
        for name, candidate in sorted(candidates.items()):
            try:
                model, selection = instantiate_candidate(config, candidate)
                raw = inspect_native_model(model, scoped)
                proof = compatibility(model, raw)
                record["candidates"].append({"model_class": name, "compatibility": proof})
                if proof["status"] == "exact_metadata_match":
                    matches.append({"model": model, "selection": selection, "config": model.config,
                                    "config_data": data, "checkpoint": scoped, "record": record})
            except UnsupportedModelError as exc:
                search_complete = False
                record["candidates"].append({"model_class": name, "error": str(exc)[:1024]})
            except NativeShapeMismatchError as exc:
                record["candidates"].append({"model_class": name, "error": str(exc)[:1024]})
            except ArtifactError as exc:
                search_complete = False
                record["candidates"].append({"model_class": name, "error": str(exc)[:1024], "status": "unverified"})
        record.update(search_complete=search_complete, matching_classes=[type(item["model"]).__name__ for item in matches])
        if len(matches) == 1 and search_complete:
            record.update(status="verified_native_component", selected_model_class=type(matches[0]["model"]).__name__)
            verified.extend(matches)
        else:
            record["status"] = "ambiguous" if len(matches) > 1 else "search_incomplete" if not search_complete else "checkpoint_incompatible"
    # Overlapping scopes would count the same tensor more than once.
    overlapping = {item["record"]["checkpoint_namespace"] for item in verified
                   for other in verified if item is not other and
                   (item["record"]["checkpoint_namespace"].startswith(other["record"]["checkpoint_namespace"]))}
    if overlapping:
        for item in verified:
            item["record"]["status"] = "overlapping_components_require_selection"
        return records, []
    return records, verified


def build_component_report(
    inventory: ModelIR, verified: list[dict[str, Any]], *, reference: ModelReference,
    revision: str, files: set[str], client: MetadataClient, audit_start: int,
    profile: dict[str, Any], progress: Callable[[str], None],
) -> ModelIR:
    """Expose independent verified component graphs and an explicit unknown wrapper."""
    from .ir import Evidence, Graph, Node
    from .pipeline import build_native_report

    graphs, raw_modules, raw_parameters, raw_buffers = [], {}, {}, {}
    component_reports = {}
    owned = set()
    for item in verified:
        record = item["record"]
        namespace = record["checkpoint_namespace"]
        scope = {key: record[key] for key in ("path", "config_file", "checkpoint_namespace", "status",
                                              "config_canonical_sha256", "selected_model_class")}
        report = build_native_report(
            reference=reference, revision=revision, files=files, client=client, audit_start=audit_start,
            config_file=record["config_file"], config_data=item["config_data"], config=item["config"],
            checkpoint=item["checkpoint"], model=item["model"], selection=item["selection"],
            profile=dict(profile), progress=progress,
        )
        identity = "component_" + hashlib.sha256(namespace.encode()).hexdigest()[:12]
        def qualify(name: str, prefix: str = namespace) -> str:
            return prefix + name if name else prefix.removesuffix(".")
        for graph in report.root_graph.walk():
            graph.id = identity + "_" + graph.id
            graph.metadata["component_scope"] = scope
            if "module_path" in graph.metadata:
                graph.metadata["module_path"] = qualify(graph.metadata["module_path"])
            for node in graph.nodes:
                if node.module_path is not None:
                    node.module_path = qualify(node.module_path)
                node.source_modules = [qualify(path) for path in node.source_modules]
                if "parameter_refs" in node.metadata:
                    node.metadata["parameter_refs"] = [qualify(path) for path in node.metadata["parameter_refs"]]
                if "buffer_ref" in node.metadata:
                    node.metadata["buffer_ref"] = qualify(node.metadata["buffer_ref"])
                if node.metadata.get("tensor_name") in report.tensors:
                    node.metadata["tensor_name"] = namespace + node.metadata["tensor_name"]

        for path, module in report.raw_structure["modules"].items():
            raw_modules[qualify(path)] = {**module, "path": qualify(path),
                "parent": qualify(module["parent"]) if module["parent"] is not None else None,
                "children": [qualify(p) for p in module["children"]],
                "parameters": [qualify(p) for p in module["parameters"]],
                "buffers": [qualify(p) for p in module["buffers"]]}
        for path, parameter in report.raw_structure["parameters"].items():
            raw_parameters[qualify(path)] = {**parameter, "name": qualify(path),
                "canonical_parameter": qualify(parameter["canonical_parameter"]),
                "aliases": [qualify(p) for p in parameter["aliases"]],
                "tensor_names": [namespace + p for p in parameter["tensor_names"]],
                "tensor_name": namespace + parameter["tensor_name"] if parameter["tensor_name"] else None}
        raw_buffers.update({qualify(path): buffer for path, buffer in report.raw_structure["buffers"].items()})
        for name, tensor in report.tensors.items():
            original = namespace + name
            owned.add(original)
            inventory.tensors[original] = {**tensor, "name": original,
                "parameter_refs": [qualify(path) for path in tensor["parameter_refs"]],
                "component_scope": namespace.removesuffix(".")}
        graphs.append(Node(identity, f"Verified component: {record['path']}", "unknown",
            module_path=namespace.removesuffix("."), module_type=report.architecture,
            child_graph=report.root_graph,
            metadata={"component_scope": scope, "invocation_not_established": True,
                      "unknown_category": "wrapper_invocation_unproven"},
            evidence=[Evidence("component_checkpoint_binding", "derived", record["config_file"], scope)]))
        component_reports[namespace] = report.metadata["analysis_report"]
        inventory.evidence.extend(report.evidence)
    inventory.root_graph = Graph("component_overview", [
        Node("unknown_wrapper", "Custom wrapper computation is unavailable", "unknown",
             metadata={"unknown_category": "unsupported_source", "component_connections_inferred": False}),
        *graphs,
    ], [], metadata={"topology": "verified_components_without_wrapper_dataflow",
                     "scope": "Independent component architecture only; no inter-component calls inferred"})
    inventory.architecture = "Verified components; wrapper unresolved"
    inventory.metadata.pop("inspection", None)
    inventory.metadata.update(unique_parameter_count=None, logical_parameter_count=None,
        component_configs={item["record"]["config_file"]: item["config_data"] for item in verified},
        component_parameter_counts={namespace: sum(p["parameter_count"] for path, p in raw_parameters.items()
            if path.startswith(namespace) and path == p["canonical_parameter"]) for namespace in component_reports},
        analysis_report={"mode": "static_components", "forward_executed": False, "runtime_trace_available": False,
            "component_reports": component_reports, "unknown_nodes": 1 + len(graphs),
            "unmapped_checkpoint_tensors": sorted(set(inventory.tensors) - owned),
            "warnings": ["Verified component graphs do not establish custom wrapper computation or calls"]},
        limitations=["Full model implementation is unavailable; components are scoped to exact config directories and checkpoint namespaces",
                     "No model-level unique parameter count or dataflow between components is asserted"])
    inventory.raw_structure = {"modules": raw_modules, "parameters": raw_parameters, "buffers": raw_buffers,
                               "component_discovery": profile["component_discovery"]}
    inventory.root_graph.validate()
    return inventory
