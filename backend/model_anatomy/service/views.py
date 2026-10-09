"""Read-only views and exports of the engine's stored IR 1.1, with no model imports."""

import io
import json
import zipfile
from dataclasses import asdict, replace

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
    for item in ir.root_graph.walk():
        if item.id == graph_id:
            return item
    raise ServiceError(404, "not_found", "Graph not found")


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

    rendered = mermaid(ir.root_graph)
    markdown = f"# {ir.model_id}\n\nRevision: `{ir.metadata.get('revision')}`\n\n```mermaid\n{rendered}```\n"
    limitations = ir.metadata.get("limitations", [])
    if limitations:
        markdown += "\n## Limitations\n\n" + "\n".join(f"- {value}" for value in limitations) + "\n"
    files = {
        "model.json": ir.to_json().encode(),
        "root.mmd": rendered.encode(),
        "model.md": markdown.encode(),
        "diagnostics.json": encode(
            {
                "analysis_report": ir.metadata.get("analysis_report"),
                "semantic": ir.raw_structure.get("semantic_diagnostics"),
                "static_flow": ir.raw_structure.get("static_flow_reports"),
            }
        ),
        "artifact_audit.json": encode(ir.metadata.get("artifacts", {})),
    }
    for item in ir.root_graph.walk():
        # Archive names are assigned independently of user-controlled graph IDs.
        index = len(files)
        files[f"graphs/{index}.mmd"] = mermaid(item).encode()
    return files


def export(ir: ModelIR, format: str) -> tuple[bytes, str, str]:
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
