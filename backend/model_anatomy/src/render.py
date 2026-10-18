"""Debug renderers consuming only the public IR."""

from __future__ import annotations

import hashlib
import re
from html import escape
from pathlib import Path

from .errors import AnatomyError, ArtifactError
from .ir import Graph, ModelIR
from .output_transaction import publish_files


def _label(value: str) -> str:
    return escape(value, quote=True).replace("|", "&#124;").replace("\n", " ")


def mermaid(graph: Graph) -> str:
    """Render one selected graph, leaving nested graphs independently inspectable."""
    graph.validate()
    ids = {node.id: f"n{index}" for index, node in enumerate(graph.nodes)}
    lines = ["flowchart TD"]
    for node in graph.nodes:
        label = node.label + (f" x{node.repeat}" if node.repeat > 1 else "")
        if node.repetition and node.repetition.mode != "sequential":
            label += f" ({node.repetition.mode})"
        lines.append(f'    {ids[node.id]}["{_label(label)}"]')
    for edge in graph.edges:
        shape = (
            ", ".join(str(d) for d in edge.tensor.shape)
            if edge.tensor and edge.tensor.shape is not None
            else "unknown shape"
        )
        label = f"{edge.label}: " if edge.label else ""
        label += f"[{shape}]" if edge.relation == "dataflow" else edge.relation
        arrow = "-->" if edge.relation == "dataflow" else "-.->"
        lines.append(f'    {ids[edge.source]} {arrow}|"{_label(label)}"| {ids[edge.target]}')
    return "\n".join(lines) + "\n"


def write_outputs(
    ir: ModelIR, output_directory: Path, config_directory: Path, preview_directory: Path
) -> list[Path]:
    """Write formal JSON/Mermaid artifacts and an English Markdown preview."""
    import json

    model_path = output_directory / "model.json"
    old_graph_ids = set()
    previous = None
    if model_path.exists():
        try:
            previous = ModelIR.from_json(model_path.read_text(encoding="utf-8"))
        except (AnatomyError, ValueError, KeyError, TypeError):
            pass
    if previous is not None:
        if previous.model_id != ir.model_id:
            raise ArtifactError("Output directory already belongs to a different model")
        old_graph_ids = {graph.id for graph in previous.root_graph.walk()}
    serialized = ir.to_json()
    if ModelIR.from_json(serialized).to_dict() != ir.to_dict():
        raise ValueError("IR round-trip changed the model representation")
    contents = {model_path: serialized}
    config_path = config_directory / "config.json"
    contents[config_path] = json.dumps(ir.metadata["source_config"], indent=2, sort_keys=True) + "\n"
    audit_path = output_directory / "artifact_audit.json"
    contents[audit_path] = json.dumps(ir.metadata["artifacts"], indent=2, sort_keys=True) + "\n"
    paths = [model_path, config_path, audit_path]
    diagnostics_path = output_directory / "diagnostics.json"
    contents[diagnostics_path] = (
        json.dumps(
            {
                "analysis_report": ir.metadata.get("analysis_report"),
                "module_tree": ir.raw_structure.get("modules"),
                **ir.raw_structure.get("semantic_diagnostics", {}),
                "static_flow": ir.raw_structure.get("static_flow_reports"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths.append(diagnostics_path)
    preview = [
        f"# {ir.architecture} Hierarchy Preview",
        "",
        f"Model: `{ir.model_id}` | Revision: `{ir.metadata['revision']}`",
        "",
        "These graphs are generated from the same Hierarchical Model IR. Solid edges represent static source or pattern dataflow; dotted edges represent structural relationships. No forward pass was executed.",
        "",
        "Child graphs of repeated nodes show a representative instance. All instance paths and parameters remain in the raw structure. Unknown shapes and components are explicitly marked.",
        "",
    ]
    for graph in ir.root_graph.walk():
        rendered = mermaid(graph)
        path = output_directory / f"{graph.id}.mmd"
        contents[path] = rendered
        paths.append(path)
        preview.extend([f"## {graph.id}", "", "```mermaid", rendered.rstrip(), "```", ""])
    obsolete = {
        output_directory / f"{obsolete_id}.mmd"
        for obsolete_id in old_graph_ids - {graph.id for graph in ir.root_graph.walk()}
        if obsolete_id.replace("_", "").isalnum()
    }
    identity = f"{ir.model_id}@{ir.metadata['revision']}"
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    stem = re.sub(r"[^A-Z0-9_]", "_", ir.model_id.upper().replace("/", "__"))[:120]
    preview_path = preview_directory / f"{stem}_{suffix}_PREVIEW.md"
    contents[preview_path] = "\n".join(preview)
    paths.append(preview_path)
    publish_files(contents, obsolete)
    return paths
