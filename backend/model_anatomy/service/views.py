"""Read-only views and exports of the engine's stored IR 1.1, with no model imports."""

import io
import json
import zipfile
from dataclasses import asdict, replace
from html import escape

from model_anatomy.src.ir import ModelIR
from model_anatomy.src.render import mermaid

from .errors import ServiceError


def load_ir(data: bytes | str) -> ModelIR:
    try:
        ir = ModelIR.from_json(data.decode("utf-8") if isinstance(data, bytes) else data)
        if ir.schema_version != "1.1":
            raise ValueError("Only IR 1.1 is supported")
        return ir
    except (ValueError, KeyError, TypeError) as exc:
        raise ServiceError(503, "invalid_artifact", "Stored IR failed validation") from exc


def graph(ir: ModelIR, graph_id: str):
    if ir.metadata.get("inspection", {}).get("kind") == "artifact_inspection":
        raise ServiceError(
            409,
            "architecture_unavailable",
            "This report contains verified artifacts, not an architecture graph",
        )
    for item in ir.root_graph.walk():
        if item.id == graph_id:
            return item
    raise ServiceError(404, "not_found", "Graph not found")


def artifact_view(ir: ModelIR, offset: int = 0, limit: int = 50) -> dict:
    inspection = ir.metadata.get("inspection", {})
    files = inspection.get("files", ir.metadata.get("artifacts", {}).get("repository_files", []))
    names = sorted(ir.tensors)
    tensors = [dict(ir.tensors[name], name=name) for name in names[offset : offset + limit]]
    return {
        "kind": inspection.get("kind", "architecture"),
        "reason_code": inspection.get("reason_code"),
        "message": inspection.get("message"),
        "revision": ir.metadata["revision"],
        "files": files[offset : offset + limit],
        "file_count": len(files),
        "tensors": tensors,
        "tensor_count": len(names),
        "checkpoint_tensor_element_count": ir.metadata.get("checkpoint_tensor_element_count"),
        "checkpoint_available": inspection.get("checkpoint_available", bool(tensors)),
        "checkpoint_format": ir.metadata.get("artifacts", {}).get("checkpoint_format"),
        "selected_file_tensor_element_count": ir.metadata.get("artifacts", {}).get(
            "selected_file_tensor_element_count"
        ),
        "selected_file_split_metadata": ir.metadata.get("artifacts", {}).get("selected_file_split_metadata"),
        "source_config": ir.metadata.get("source_config"),
        "model_profile": ir.metadata.get("model_profile"),
        "evidence": [asdict(item) for item in ir.evidence],
        "offset": offset,
        "limit": limit,
    }


def public_node(node) -> dict:
    # Exclude the nested graph before serialization to keep pagination responses bounded.
    result = {
        key: value for key, value in asdict(replace(node, child_graph=None)).items() if key != "child_graph"
    }
    result["child_graph_id"] = node.child_graph.id if node.child_graph else None
    return result


def graph_view(ir: ModelIR, graph_id: str) -> dict:
    item = graph(ir, graph_id)
    return {
        "id": item.id,
        "nodes": [public_node(n) for n in item.nodes],
        "edges": [asdict(e) for e in item.edges],
        "metadata": item.metadata,
    }


def node_view(ir: ModelIR, graph_id: str, node_id: str, offset=0, limit=50) -> dict:
    selected = graph(ir, graph_id)
    node = next((n for n in selected.nodes if n.id == node_id), None)
    if node is None:
        raise ServiceError(404, "not_found", "Node not found")
    prefixes = set(node.source_modules)
    if node.module_path is not None:
        prefixes.add(node.module_path)
    raw = ir.raw_structure
    parameters = raw.get("parameters", {})
    buffers = raw.get("buffers", {})
    names = set()
    for path, parameter in {**parameters, **buffers}.items():
        if any(path.startswith(prefix + ".") or path == prefix or prefix == "" for prefix in prefixes):
            names.update(parameter.get("tensor_names", []))
            if parameter.get("tensor_name"):
                names.add(parameter["tensor_name"])
    # Parameter nodes can identify checkpoint names directly in their metadata.
    tensor_name = node.metadata.get("tensor_name")
    if tensor_name in ir.tensors:
        names.add(tensor_name)
    tensors = [dict(item, name=name) for name, item in sorted(ir.tensors.items()) if name in names]
    warnings = list(ir.metadata.get("analysis_report", {}).get("warnings", []))
    module = raw.get("modules", {}).get(node.module_path)
    return {
        "graph_id": graph_id,
        "node": public_node(node),
        "tensors": tensors[offset : offset + limit],
        "tensor_count": len(tensors),
        "evidence": [asdict(e) for e in node.evidence],
        "module": module,
        "warnings": warnings,
    }


def export_files(ir: ModelIR) -> dict[str, bytes]:
    def encode(data):
        return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

    inspection = ir.metadata.get("inspection", {})
    artifacts_only = inspection.get("kind") == "artifact_inspection"
    rendered = "" if artifacts_only else mermaid(ir.root_graph)
    markdown = f"# {ir.model_id}\n\nRevision: `{ir.metadata.get('revision')}`\n\n"
    if artifacts_only:
        markdown += f"## Architecture unavailable\n\n{inspection['message']}\n\n"
        markdown += f"Verified tensor records: {len(ir.tensors)}. Tensor elements are not a model parameter count.\n\n"
        markdown += (
            "## Repository files\n\n"
            + "\n".join(
                "- <code>" + escape(name).replace("\r", "&#13;").replace("\n", "&#10;") + "</code>"
                for name in inspection.get("files", [])
            )
            + "\n"
        )
    else:
        markdown += f"```mermaid\n{rendered}```\n"
    limitations = ir.metadata.get("limitations", [])
    if limitations:
        markdown += "\n## Limitations\n\n" + "\n".join(f"- {value}" for value in limitations) + "\n"
    files = {
        "model.json": ir.to_json().encode(),
        "model.md": markdown.encode(),
        "diagnostics.json": encode(
            {
                "analysis_report": ir.metadata.get("analysis_report"),
                "semantic": ir.raw_structure.get("semantic_diagnostics"),
                "module_topology": ir.raw_structure.get("module_topology"),
                "inspection": inspection or None,
            }
        ),
        "artifact_audit.json": encode(ir.metadata.get("artifacts", {})),
    }
    if not artifacts_only:
        files["root.mmd"] = rendered.encode()
    for item in [] if artifacts_only else ir.root_graph.walk():
        # Archive names are assigned independently of user-controlled graph IDs.
        index = len(files)
        files[f"graphs/{index}.mmd"] = mermaid(item).encode()
    return files


def export(ir: ModelIR, format: str) -> tuple[bytes, str, str]:
    if format == "mermaid" and ir.metadata.get("inspection", {}).get("kind") == "artifact_inspection":
        raise ServiceError(
            409,
            "architecture_unavailable",
            "No architecture diagram is available for this artifact inspection",
        )
    files = export_files(ir)
    formats = {
        "json": ("model.json", "application/json"),
        "mermaid": ("root.mmd", "text/plain"),
        "markdown": ("model.md", "text/markdown"),
        "diagnostics": ("diagnostics.json", "application/json"),
        "audit": ("artifact_audit.json", "application/json"),
    }
    if format in formats:
        name, media = formats[format]
        return files[name], media, name
    if format == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return buffer.getvalue(), "application/zip", "model-anatomy.zip"
    raise ServiceError(422, "invalid_export", "Expected json, mermaid, markdown, diagnostics, audit, or zip")
