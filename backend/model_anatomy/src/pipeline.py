"""Static native-Transformers reconstruction from configuration and checkpoint headers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import requests

from .artifact_inspection import build_artifact_inspection, has_adapter_tensors
from .artifacts import AmbiguousCheckpointError, InspectionBudgetError, MetadataClient, parse_reference
from .errors import UnsupportedModelError
from .ir import Evidence, ModelIR
from .profile import config_filename, native_dimensions, repository_profile
from .version import ENGINE_VERSION


def reconstruct(
    value: str,
    *,
    client: MetadataClient | None = None,
    revision: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    allow_artifact_inspection: bool = False,
) -> ModelIR:
    """Construct metadata and a semantic hierarchy; never execute forward or load weights."""
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

                raise ArtifactError("Resolved revision disagrees with the pinned revision")
    else:
        revision = client.resolve(reference)
    progress("reading_metadata")
    files = client.repository_files[(reference.model_id, revision)]
    if reference.explicit_filename and reference.filename not in files:
        from .errors import ArtifactError

        raise ArtifactError(f"Requested Safetensors artifact is missing: {reference.filename}")
    config_file = config_filename(reference)
    config_data = client.json(reference.url(config_file, revision)) if config_file in files else None
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

    has_safetensors = any(name.endswith((".safetensors", ".safetensors.index.json")) for name in files)
    has_gguf = any(name.endswith(".gguf") for name in files)
    if reference.explicit_filename and reference.filename.endswith(".gguf") and not allow_artifact_inspection:
        from .errors import ArtifactError

        raise ArtifactError("GGUF header inspection is available only through artifact inspection")
    adapter_config = None
    directory = reference.filename.rpartition("/")[0] if reference.explicit_filename else ""
    adapter_file = f"{directory}/adapter_config.json" if directory else "adapter_config.json"
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

    if not has_safetensors:
        if not allow_artifact_inspection:
            from .errors import ArtifactError

            raise ArtifactError("Repository has no Safetensors checkpoint; it is outside the target scope")
        if has_gguf or reference.filename.endswith(".gguf"):
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
            except InspectionBudgetError:
                return inspect_artifacts(
                    reference,
                    revision,
                    files,
                    client.audit,
                    audit_start=audit_start,
                    config_data=config_data,
                    checkpoint=None,
                    reason_code="metadata_budget_exceeded",
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
        reason = (
            "adapter_requires_base_model"
            if adapter_config is not None
            else "architecture_config_unavailable"
            if config_data is None
            else "safetensors_unavailable"
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
    except InspectionBudgetError:
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
            reason_code="metadata_budget_exceeded",
        )

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
        {shard.metadata["format"] for shard in checkpoint.shards.values() if "format" in shard.metadata}
    )
    non_native_frameworks = [framework for framework in frameworks if framework != "pt"]
    if non_native_frameworks:
        message = "Safetensors metadata declares non-PyTorch framework(s): " + ", ".join(
            non_native_frameworks
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
            else "architecture_config_unavailable"
            if config_data is None
            else "native_architecture_unavailable"
            if native_error is not None
            else None
        )
    if reason is not None:
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

    progress("building_structure")
    try:
        from .registry import instantiate_native

        model, selection = instantiate_native(config)
    except UnsupportedModelError as exc:
        if not allow_artifact_inspection:
            raise
        profile["architecture_blockers"].append(str(exc)[:1024])
        return inspect_artifacts(
            reference,
            revision,
            files,
            client.audit,
            audit_start=audit_start,
            config_data=config_data,
            checkpoint=checkpoint,
            reason_code="native_architecture_unavailable",
        )
    from .adapters import adapter_for
    from .facts import inspect_native_model
    from .hierarchy import HierarchyBuilder
    from .inputs import InputResolver
    from .patterns import SemanticAnalyzer

    config_url = reference.url(config_file, revision)
    raw = inspect_native_model(model, checkpoint)
    adapter = adapter_for(config.model_type)
    if adapter:
        adapter.refine_facts(raw, checkpoint, config_data)
    analyzer = SemanticAnalyzer(model, raw, adapter)
    progress("analyzing_components")
    builder = HierarchyBuilder(model, raw, analyzer)
    root = builder.build()
    raw["semantic_diagnostics"] = analyzer.diagnostics()
    raw["static_flow_reports"] = builder.flow_reports
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
            "role": "parameter" if name in parameter_refs else raw["checkpoint_extras"][name]["kind"],
            "evidence_kind": "direct",
            "role_evidence_kind": "reconstructed"
            if name in parameter_refs
            else raw["checkpoint_extras"][name]["evidence_kind"],
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
                r["url"] for r in client.audit[audit_start:] if r["url"].endswith(".safetensors.index.json")
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
    warnings = list(selection["warnings"]) + list(raw["warnings"])
    for path, report in builder.flow_reports.items():
        warnings.extend(f"{path or '<root>'}: {w}" for w in report["warnings"])
    unknown_tensors = [name for name, item in raw["checkpoint_extras"].items() if item["kind"] == "unmapped"]
    if unknown_tensors:
        warnings.append(
            f"{len(unknown_tensors)} checkpoint tensors have no native module owner; preserved as unmapped"
        )
    all_nodes = [
        node
        for graph in root.walk()
        for node in graph.nodes
        if node.kind not in {"input", "output", "parameter"}
    ]
    unknown = [node for node in all_nodes if node.kind == "unknown"]
    edges = [edge for graph in root.walk() for edge in graph.edges if edge.relation == "dataflow"]
    unknown_modules = raw["semantic_diagnostics"]["unknown_components"]
    metadata = {
        "model_profile": profile,
        "revision": revision,
        "requested_revision": reference.revision,
        "source_config": config_data,
        "dimensions": native_dimensions(config),
        "unique_parameter_count": raw["unique_parameter_count"],
        "logical_parameter_count": raw["logical_parameter_count"],
        "checkpoint_tensor_element_count": sum(t.parameter_count for t in checkpoint.tensors.values()),
        "checkpoint_buffer_element_count": sum(
            t.parameter_count
            for n, t in checkpoint.tensors.items()
            if raw["checkpoint_extras"].get(n, {}).get("kind") in {"buffer", "legacy_buffer"}
        ),
        "versions": raw["versions"],
        "engine_version": ENGINE_VERSION,
        "artifacts": artifact_data,
        "inputs": InputResolver(model).describe(),
        "parameter_count_policy": "Model parameter object identity is unique; checkpoint duplicates and hierarchy references are reported separately",
        "analysis_report": {
            "mode": "static",
            "analysis_scope": {
                "profile": "canonical_inference",
                "model_mode": "eval",
                "cache": "disabled",
                "training_loss": "excluded",
                "inputs": "canonical modality inputs",
            },
            "forward_executed": False,
            "runtime_trace_available": False,
            "semantic_coverage": 1 - len(unknown) / len(all_nodes) if all_nodes else 0,
            "classified_nodes": len(all_nodes) - len(unknown),
            "unknown_nodes": len(unknown),
            "module_semantic_coverage": 1 - len(unknown_modules) / len(raw["modules"]),
            "unknown_module_paths": unknown_modules,
            "unmapped_checkpoint_tensors": unknown_tensors,
            "checkpoint_parameter_coverage": sum(
                p["parameter_count"]
                for name, p in raw["parameters"].items()
                if name == p["canonical_parameter"] and p["checkpoint_matched"]
            )
            / raw["unique_parameter_count"]
            if raw["unique_parameter_count"]
            else 1,
            "tensor_shape_coverage": sum(e.tensor is not None and e.tensor.shape is not None for e in edges)
            / len(edges)
            if edges
            else 0,
            "adapters_used": [adapter.name] if adapter else [],
            "static_flow_diagnostics": [
                {"module_path": path, **item}
                for path, report in builder.flow_reports.items()
                for item in report.get("diagnostics", [])
                if isinstance(item, dict)
            ],
            "profile_assumptions": [
                {"module_path": path, **assumption}
                for path, report in builder.flow_reports.items()
                for assumption in report.get("profile_assumptions", [])
                if isinstance(assumption, dict)
            ],
            "warnings": list(dict.fromkeys(warnings)),
        },
        "limitations": [
            "Coverage describes canonical eval inference with cache and training loss disabled; other input, cache and training profiles are not established",
            "Static source analysis supports a bounded Python subset; unresolved branches and unsupported loops are reported",
            "Semantic classifications and symbolic shapes are inspectable reconstructions, not runtime observations",
            "Floating edge dtypes describe the native meta model profile; runtime casts are unverified, while checkpoint dtypes are direct header evidence",
            "Secondary tuple output shapes are unknown unless their role is established; opaque transformations retain dependency evidence only",
            "Containment edges are explicitly distinguished from dataflow; registration order is never assumed to be execution order",
            "Forward execution, weight downloads, remote code, and CPU parameter allocation are disabled",
            "Quantized shape conversions, opaque kernels, and optional native dependencies may require additional handling",
            "Collapsed child graphs show a representative instance; all source paths and raw parameter records are retained",
        ],
    }
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
            "parameter_identity",
            "direct",
            "named_parameters(remove_duplicate=False)",
            {"alias_groups": raw["parameter_alias_groups"]},
        ),
    ]
    ir = ModelIR(reference.model_id, type(model).__name__, root, metadata, tensors, raw, evidence)
    progress("validating")
    ir.root_graph.validate()
    return ir
