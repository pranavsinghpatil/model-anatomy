"""Debug renderers consuming only the public IR."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .errors import AnatomyError
from .ir import Graph, ModelIR


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
    if model_path.exists():
        try:
            previous = ModelIR.from_json(model_path.read_text(encoding="utf-8"))
            if previous.model_id == ir.model_id:
                old_graph_ids = {graph.id for graph in previous.root_graph.walk()}
        except (AnatomyError, ValueError, KeyError, TypeError):
            pass
    serialized = ir.to_json()
    if ModelIR.from_json(serialized).to_dict() != ir.to_dict():
        raise ValueError("IR round-trip changed the model representation")
    output_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)
    preview_directory.mkdir(parents=True, exist_ok=True)
    model_path.write_text(serialized, encoding="utf-8")
    config_path = config_directory / "config.json"
    config_path.write_text(
        json.dumps(ir.metadata["source_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path = output_directory / "artifact_audit.json"
    audit_path.write_text(
        json.dumps(ir.metadata["artifacts"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths = [model_path, config_path, audit_path]
    diagnostics_path = output_directory / "diagnostics.json"
    diagnostics_path.write_text(
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
        + "\n",
        encoding="utf-8",
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
        path.write_text(rendered, encoding="utf-8")
        paths.append(path)
        preview.extend([f"## {graph.id}", "", "```mermaid", rendered.rstrip(), "```", ""])
    for obsolete_id in old_graph_ids - {graph.id for graph in ir.root_graph.walk()}:
        if obsolete_id.replace("_", "").isalnum():
            (output_directory / f"{obsolete_id}.mmd").unlink(missing_ok=True)
    preview_name = (
        "GPT2_PREVIEW.md"
        if ir.metadata.get("source_config", {}).get("model_type") == "gpt2"
        else output_directory.name.upper().replace("-", "_") + "_PREVIEW.md"
    )
    preview_path = preview_directory / preview_name
    preview_path.write_text("\n".join(preview), encoding="utf-8")
    paths.append(preview_path)
    return paths
