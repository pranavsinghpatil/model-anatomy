"""Static native-Transformers reconstruction from configuration and checkpoint headers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import requests

from .artifact_inspection import build_artifact_inspection, has_adapter_tensors
from .artifacts import (
    AmbiguousCheckpointError,
    InspectionBudgetError,
    MetadataClient,
    parse_reference,
)
from .errors import ArtifactError, UnsupportedModelError
from .ir import Evidence, ModelIR
from .profile import config_filename, native_dimensions, repository_profile
from .version import ENGINE_VERSION


def _unmapped_checkpoint_tensors(raw: dict) -> list[str]:
    return [
        name
        for name, item in raw["checkpoint_extras"].items()
        if item["kind"] == "unmapped"
    ]


def _align_architecture_with_checkpoint(config, checkpoint, model, selection, raw):
    """Use a native class that owns checkpoint tensors the declared class leaves out."""
    unmapped = _unmapped_checkpoint_tensors(raw)
    if not unmapped or not checkpoint.tensors:
        return model, selection, raw
    from .facts import inspect_native_model
    from .registry import construct_native, native_auto_candidates

    best = None
    for _auto_name, auto_class, model_name in native_auto_candidates(config):
        if model_name == type(model).__name__:
            continue
        try:
            alternate = construct_native(config, auto_class)
            alternate_raw = inspect_native_model(alternate, checkpoint)
        except (UnsupportedModelError, ArtifactError):
            continue
        alternate_unmapped = _unmapped_checkpoint_tensors(alternate_raw)
        if len(alternate_unmapped) >= len(unmapped):
            continue
        matched = sum(
            parameter["parameter_count"]
            for name, parameter in alternate_raw["parameters"].items()
            if name == parameter["canonical_parameter"]
            and parameter["checkpoint_matched"]
        )
        if (
            best is None
            or len(alternate_unmapped) < best[0]
            or (len(alternate_unmapped) == best[0] and matched > best[1])
        ):
            best = (
                len(alternate_unmapped),
                matched,
                alternate,
                alternate_raw,
                auto_class,
                model_name,
            )
    if best is None:
        return model, selection, raw
    _count, _matched, model, raw, auto_class, model_name = best
    declared = selection.get("model_class")
    selection = {
        **selection,
        "auto_class": auto_class.__name__,
        "model_class": model_name,
        "checkpoint_alignment": (
            f"The declared architecture {declared} leaves {len(unmapped)} checkpoint tensors without a module. "
            f"The graph uses {model_name}, which owns those tensors."
        ),
    }
    return model, selection, raw


def reconstruct(
    value: str,
    *,
    client: MetadataClient | None = None,
    revision: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    allow_artifact_inspection: bool = False,
) -> ModelIR:
    """Construct metadata and a semantic hierarchy; never execute forward or load weights."""
    from .native_provider import require_supported_provider

    require_supported_provider()
    reference = parse_reference(value)
    if client is None:
        with requests.Session() as session:
            return reconstruct(
                value,
                client=MetadataClient(session),
                revision=revision,
                on_progress=on_progress,
                allow_artifact_inspection=allow_artifact_inspection,
            )
    audit_start = len(client.audit)
    progress = on_progress or (lambda stage: None)
    progress("resolving")
    if revision is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            from .errors import ArtifactError

            raise ArtifactError("An immutable 40-character revision is required")
        if (reference.model_id, revision) not in client.repository_files:
            pinned = client.resolve(replace(reference, revision=revision))
            if pinned != revision:
                from .errors import ArtifactError

                raise ArtifactError(
                    "Resolved revision disagrees with the pinned revision"
                )
    else:
        revision = client.resolve(reference)
    progress("reading_metadata")
    files = client.repository_files[(reference.model_id, revision)]
    if reference.explicit_filename and reference.filename not in files:
        from .errors import ArtifactError

        raise ArtifactError(
            f"Requested Safetensors artifact is missing: {reference.filename}"
        )
    config_file = config_filename(reference)
    config_data = (
        client.json(reference.url(config_file, revision))
        if config_file in files
        else None
    )
    manifest = (
        client.json(reference.url("model_index.json", revision))
        if allow_artifact_inspection and "model_index.json" in files
        else None
    )
    profile = repository_profile(
        reference,
        files,
        config_data,
        client.repository_metadata.get((reference.model_id, revision), {}),
        manifest,
    )
    inspect_artifacts = partial(build_artifact_inspection, model_profile=profile)
    if manifest is not None and not reference.explicit_filename:
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=None,
            reason_code="pipeline_components_require_selection",
        )
    if config_data is None and not allow_artifact_inspection:
        from .errors import ArtifactError

        raise ArtifactError("Repository has no config.json")

    has_safetensors = any(
        name.endswith((".safetensors", ".safetensors.index.json")) for name in files
    )
    has_gguf = any(name.endswith(".gguf") for name in files)
    if (
        reference.explicit_filename
        and reference.filename.endswith(".gguf")
        and not allow_artifact_inspection
    ):
        from .errors import ArtifactError

        raise ArtifactError(
            "GGUF header inspection is available only through artifact inspection"
        )
    adapter_config = None
    directory = (
        reference.filename.rpartition("/")[0] if reference.explicit_filename else ""
    )
    adapter_file = (
        f"{directory}/adapter_config.json" if directory else "adapter_config.json"
    )
    if allow_artifact_inspection and adapter_file in files:
        adapter_config = client.json(reference.url(adapter_file, revision))

    config = None
    native_error = None
    if config_data is not None:
        try:
            from .registry import load_native_config

            config = load_native_config(config_data)
        except UnsupportedModelError as exc:
            if not allow_artifact_inspection:
                raise
            native_error = exc
            profile["architecture_blockers"].append(str(exc)[:1024])
            profile["native_resolution"] = getattr(
                exc,
                "diagnostic",
                {
                    "status": "blocked",
                    "stage": "config_resolution",
                    "code": "native_config_unavailable",
                    "message": str(exc)[:1024],
                },
            )

    if not has_safetensors and not allow_artifact_inspection:
        from .errors import ArtifactError

        raise ArtifactError(
            "Repository has no Safetensors checkpoint; it is outside the target scope"
        )
    if not has_safetensors and (has_gguf or reference.filename.endswith(".gguf")):
        try:
            checkpoint = client.checkpoint(reference, revision)
        except AmbiguousCheckpointError:
            return inspect_artifacts(
                reference,
                revision,
                files,
                client.audit,
                audit_start=audit_start,
                config_data=config_data,
                checkpoint=None,
                reason_code="checkpoint_selection_required",
            )
        except InspectionBudgetError as exc:
            return inspect_artifacts(
                reference,
                revision,
                files,
                client.audit,
                audit_start=audit_start,
                config_data=config_data,
                checkpoint=None,
                reason_code="metadata_budget_exceeded",
                declared_checkpoint_index=exc.index_summary,
            )
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=checkpoint,
            reason_code="gguf_header_only",
        )
    if not has_safetensors and (
        adapter_config is not None or config is None or native_error is not None
    ):
        reason = (
            "adapter_requires_base_model"
            if adapter_config is not None
            else (
                "architecture_config_unavailable"
                if config_data is None
                else "native_architecture_unavailable"
            )
        )
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=None,
            reason_code=reason,
        )
    if not has_safetensors:
        from .artifacts import CheckpointMetadata

        checkpoint = CheckpointMetadata(
            tensors={}, shards={}, tensor_files={}, format="absent"
        )
    else:
        try:
            checkpoint = client.checkpoint(reference, revision)
        except AmbiguousCheckpointError:
            if not allow_artifact_inspection:
                raise
            return inspect_artifacts(
                reference,
                revision,
                files,
                client.audit,
                audit_start=audit_start,
                config_data=config_data,
                checkpoint=None,
                reason_code=(
                    "adapter_requires_base_model"
                    if adapter_config is not None
                    else "checkpoint_selection_required"
                ),
            )
        except InspectionBudgetError as exc:
            if not allow_artifact_inspection:
                raise
            artifact_ir = inspect_artifacts(
                reference,
                revision,
                files,
                client.audit,
                audit_start=audit_start,
                config_data=config_data,
                checkpoint=None,
                reason_code="metadata_budget_exceeded",
                declared_checkpoint_index=exc.index_summary,
            )
            if config_data is not None and exc.index_summary is not None:
                from .publisher_topology import build_reviewed_topology

                return (
                    build_reviewed_topology(
                        reference, revision, files, config_data, artifact_ir
                    )
                    or artifact_ir
                )
            return artifact_ir

    if checkpoint.format == "gguf":
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=checkpoint,
            reason_code="gguf_header_only",
        )

    frameworks = sorted(
        {
            shard.metadata["format"]
            for shard in checkpoint.shards.values()
            if "format" in shard.metadata
        }
    )
    non_native_frameworks = [framework for framework in frameworks if framework != "pt"]
    if non_native_frameworks:
        message = (
            "Safetensors metadata declares non-PyTorch framework(s): "
            + ", ".join(non_native_frameworks)
        )
        if not allow_artifact_inspection:
            raise UnsupportedModelError(message)
        profile["architecture_blockers"].append(message)
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=checkpoint,
            reason_code="checkpoint_framework_unavailable",
        )

    reason = None
    if allow_artifact_inspection:
        reason = (
            "adapter_requires_base_model"
            if adapter_config is not None or has_adapter_tensors(checkpoint)
            else (
                "architecture_config_unavailable"
                if config_data is None
                else (
                    "native_architecture_unavailable"
                    if native_error is not None
                    else None
                )
            )
        )

    def inspect_unavailable(reason: str) -> ModelIR:
        if (
            reason
            in {"native_architecture_unavailable", "architecture_config_unavailable"}
            and not reference.explicit_filename
        ):
            from .components import build_component_report, discover_components

            records, verified = discover_components(
                reference, revision, files, checkpoint, client
            )
            profile["component_discovery"] = records
            if verified:
                inventory = inspect_artifacts(
                    reference,
                    revision,
                    files,
                    client.audit,
                    audit_start=audit_start,
                    config_data=config_data,
                    checkpoint=checkpoint,
                    reason_code=reason,
                )
                return build_component_report(
                    inventory,
                    verified,
                    reference=reference,
                    revision=revision,
                    files=files,
                    client=client,
                    audit_start=audit_start,
                    profile=profile,
                    progress=progress,
                )
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=checkpoint,
            reason_code=reason,
        )

    if reason is not None:
        return inspect_unavailable(reason)

    progress("building_structure")
    try:
        from .registry import instantiate_native

        model, selection = instantiate_native(config)
    except UnsupportedModelError as exc:
        if not allow_artifact_inspection:
            raise
        profile["architecture_blockers"].append(str(exc)[:1024])
        profile["native_resolution"] = getattr(
            exc,
            "diagnostic",
            {
                "status": "blocked",
                "stage": "meta_construction",
                "code": "native_construction_unavailable",
                "message": str(exc)[:1024],
            },
        )
        return inspect_unavailable("native_architecture_unavailable")
    return build_native_report(
        reference=reference,
        revision=revision,
        files=files,
        client=client,
        audit_start=audit_start,
        config_file=config_file,
        config_data=config_data,
        config=config,
        checkpoint=checkpoint,
        model=model,
        selection=selection,
        profile=profile,
        progress=progress,
    )


def build_native_report(
    *,
    reference: Any,
    revision: str,
    files: set[str],
    client: MetadataClient,
    audit_start: int,
    config_file: str,
    config_data: dict[str, Any],
    config: Any,
    checkpoint: Any,
    model: Any,
    selection: dict[str, Any],
    profile: dict[str, Any],
    progress: Callable[[str], None],
) -> ModelIR:
    """Build one module architecture path for whole models and scoped components."""
    from .facts import inspect_native_model
    from .hierarchy import HierarchyBuilder
    from .inputs import describe_inputs
    from .patterns import SemanticAnalyzer

    config_url = reference.url(config_file, revision)
    raw = inspect_native_model(model, checkpoint)
    from .architecture_selection import select_checkpoint_architecture

    model, selection, raw = select_checkpoint_architecture(
        config, checkpoint, model, selection, raw
    )
    from .architecture_selection import compatibility
    from .checkpoint_scope import checkpoint_differences

    state_compatibility = compatibility(model, raw)
    state_differences = checkpoint_differences(raw, state_compatibility)
    profile["native_resolution"] = selection.get("native_resolution")
    analyzer = SemanticAnalyzer(model, raw)
    progress("analyzing_components")
    builder = HierarchyBuilder(model, raw, analyzer)
    root = builder.build()
    inactive_state = {
        name: item for name, item in raw['checkpoint_extras'].items()
        if item['kind'] == 'inactive_component_state'
    }
    if inactive_state:
        root.metadata['checkpoint_components_outside_selected_graph'] = inactive_state
    raw["semantic_diagnostics"] = analyzer.diagnostics()
    raw["module_topology"] = builder.topology.reports
    raw["model_selection"] = selection
    parameter_refs = {}
    for name, parameter in raw["parameters"].items():
        for tensor_name in parameter["tensor_names"]:
            parameter_refs.setdefault(tensor_name, []).append(name)
    tensors = {}
    for name, tensor in sorted(checkpoint.tensors.items()):
        tensors[name] = {
            **tensor.public_dict(),
            "artifact": reference.url(checkpoint.tensor_files[name], revision),
            "parameter_refs": parameter_refs.get(name, []),
            "role": (
                "parameter"
                if name in parameter_refs
                else raw["checkpoint_extras"][name]["kind"]
            ),
            "evidence_kind": "direct",
            "role_evidence_kind": (
                "reconstructed"
                if name in parameter_refs
                else raw["checkpoint_extras"][name]["evidence_kind"]
            ),
            "binding_evidence": (
                {'layout_record_paths': [path for path in parameter_refs[name]
                                         if raw['parameters'][path].get('checkpoint_layout')]}
                if name in parameter_refs else raw['checkpoint_extras'][name]
            ),
        }
    shard_audit = {
        name: {
            "url": reference.url(name, revision),
            "header_sha256": shard.header_sha256,
            "header_length": shard.header_length,
            "file_size": shard.file_size,
        }
        for name, shard in checkpoint.shards.items()
    }
    artifact_data = {
        "repository_files": sorted(files),
        "config_url": config_url,
        "config_canonical_sha256": hashlib.sha256(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest(),
        "shards": shard_audit,
        "sharded": checkpoint.index is not None,
        "safetensors_bytes_read": sum(r["bytes_read"] for r in checkpoint.requests),
        "weight_payload_bytes_read": 0,
        "requests": client.audit[audit_start:],
    }
    if checkpoint.index is not None:
        artifact_data["index"] = {
            "url": next(
                r["url"]
                for r in client.audit[audit_start:]
                if r["url"].endswith(".safetensors.index.json")
            ),
            "canonical_sha256": hashlib.sha256(
                json.dumps(checkpoint.index, sort_keys=True).encode()
            ).hexdigest(),
            "metadata": checkpoint.index.get("metadata", {}),
            "tensor_count": len(checkpoint.index["weight_map"]),
        }
    if len(shard_audit) == 1:
        shard = next(iter(shard_audit.values()))
        artifact_data.update(
            safetensors_url=shard["url"],
            header_length=shard["header_length"],
            header_sha256=shard["header_sha256"],
            file_size=shard["file_size"],
        )
    from .completion import completion_report

    completion = completion_report(
        raw,
        builder.represented,
        builder.covered_by_leaf,
        selection,
        builder.topology.reports,
    )
    warnings = list(selection["warnings"]) + list(raw["warnings"])
    diagnostics = [
        dict(item, module_path=path)
        for path, report in builder.topology.reports.items()
        for item in report["boundaries"]
    ]
    for path, report in builder.topology.reports.items():
        blocking = [item for item in report["boundaries"] if item.get("blocking", True)]
        if blocking:
            warnings.append(
                f"{path or '<root>'}: module topology has {len(blocking)} architecture gaps"
            )
    unknown_tensors = [
        name
        for name, item in raw["checkpoint_extras"].items()
        if item["kind"] == "unmapped"
    ]
    if unknown_tensors:
        warnings.append(
            f"{len(unknown_tensors)} checkpoint tensors have no native module owner"
        )
    all_nodes = [
        node
        for graph in root.walk()
        for node in graph.nodes
        if node.module_path is not None
    ]
    unclassified = raw["semantic_diagnostics"]["unclassified_roles"]
    used_arguments = [
        node.metadata["argument"] for node in root.nodes if node.kind == "input"
    ]
    metadata = {
        "model_profile": profile,
        "revision": revision,
        "requested_revision": reference.revision,
        "source_config": config_data,
        "dimensions": native_dimensions(config),
        "unique_parameter_count": raw["unique_parameter_count"],
        "logical_parameter_count": raw["logical_parameter_count"],
        "checkpoint_tensor_element_count": sum(
            t.parameter_count for t in checkpoint.tensors.values()
        ),
        "checkpoint_buffer_element_count": sum(
            t.parameter_count
            for n, t in checkpoint.tensors.items()
            if raw["checkpoint_extras"].get(n, {}).get("kind")
            in {"buffer", "legacy_buffer"}
        ),
        "versions": raw["versions"],
        "engine_version": ENGINE_VERSION,
        "artifacts": artifact_data,
        "inputs": describe_inputs(model, used_arguments),
        "parameter_count_policy": "Model parameter object identity is unique; checkpoint duplicates and hierarchy references are reported separately",
        "analysis_report": {
            "architecture_selection": selection["architecture_selection"],
            "checkpoint_differences": state_differences,
            "completion": completion,
            "mode": "static_module_architecture",
            "forward_executed": False,
            "runtime_trace_available": False,
            "semantic_coverage": (
                sum(node.kind != "module" for node in all_nodes) / len(all_nodes)
                if all_nodes
                else 1
            ),
            "module_semantic_coverage": 1 - len(unclassified) / len(raw["modules"]),
            "unknown_nodes": 0,
            "unclassified_module_roles": unclassified,
            "unmapped_checkpoint_tensors": unknown_tensors,
            "checkpoint_parameter_coverage": (
                sum(
                    p["parameter_count"]
                    for name, p in raw["parameters"].items()
                    if name == p["canonical_parameter"] and p["checkpoint_matched"]
                )
                / raw["unique_parameter_count"]
                if raw["unique_parameter_count"]
                else 1
            ),
            "module_inventory_coverage": (
                len(builder.represented | builder.covered_by_leaf.keys())
                / len(raw["modules"])
            ),
            "module_topology": builder.topology.reports,
            "diagnostics": diagnostics,
            "warnings": list(dict.fromkeys(warnings)),
        },
        "limitations": [
            "Graphs describe actual native modules. Basic modules are leaves; tensor expressions are outside this representation",
            "Edges are source-backed module dependencies with intermediate expressions omitted, not runtime tensor traces",
            "ModuleList registration does not establish execution order; every actual layer instance is retained",
            "Checkpoint shapes and storage dtypes are direct header evidence; intermediate activation shapes are not inferred",
            "Forward execution, weight downloads and publisher remote code are disabled",
            "Module registration establishes presence, not reachability for every possible runtime input",
        ],
    }
    if selection["architecture_selection"]["status"] == "checkpoint_selected":
        metadata["limitations"].append(
            f"Native architecture {type(model).__name__} is a unique checkpoint-compatible interpretation; "
            f"the publisher config declares {selection['architecture_selection']['declared_architectures']}. "
            "Metadata compatibility does not prove publisher task intent or runtime numerical equivalence"
        )
    if selection.get("native_resolution", {}).get("publisher_auto_map"):
        metadata["limitations"].append(
            "Publisher declares custom code; this graph describes the installed native implementation. "
            "Checkpoint metadata agreement does not prove equivalence to the publisher's implementation"
        )
    if (
        selection["architecture_selection"]["status"]
        == "declared_native_with_ignored_tensors"
    ):
        metadata["limitations"].append(
            "Native loading rules ignore recorded checkpoint keys without module owners; architecture completeness "
            "covers the declared native implementation, not complete checkpoint ownership or publisher equivalence"
        )
    evidence = [
        Evidence("hf_config", "direct", config_url, {"revision": revision}),
        Evidence(
            "safetensors_header",
            "direct",
            reference.model_id,
            {"shards": shard_audit, "payload_bytes_read": 0},
        ),
        Evidence("native_auto_mapping", "direct", selection["auto_class"], selection),
        Evidence(
            "checkpoint_architecture_selection",
            "derived",
            "unique_complete_native_checkpoint_match",
            selection["architecture_selection"],
        ),
        Evidence(
            "parameter_identity",
            "direct",
            "named_parameters(remove_duplicate=False)",
            {"alias_groups": raw["parameter_alias_groups"]},
        ),
    ]
    ir = ModelIR(
        reference.model_id, type(model).__name__, root, metadata, tensors, raw, evidence
    )
    progress("validating")
    ir.root_graph.validate()
    return ir
