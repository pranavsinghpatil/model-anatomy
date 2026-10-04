"""Debug renderers consuming only the public IR."""

from __future__ import annotations

from html import escape
from pathlib import Path

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
        lines.append(f'    {ids[node.id]}["{_label(label)}"]')
    for edge in graph.edges:
        shape = ", ".join(str(d) for d in edge.tensor.shape)
        label = f"{edge.label}: " if edge.label else ""
        label += f"[{shape}]"
        lines.append(f'    {ids[edge.source]} -->|"{_label(label)}"| {ids[edge.target]}')
    return "\n".join(lines) + "\n"


def write_outputs(
    ir: ModelIR, output_directory: Path, config_directory: Path, preview_directory: Path
) -> list[Path]:
    """Write formal JSON/Mermaid artifacts and a bilingual Markdown preview."""
    import json

    serialized = ir.to_json()
    if ModelIR.from_json(serialized).to_dict() != ir.to_dict():
        raise ValueError("IR round-trip changed the model representation")
    output_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)
    preview_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "model.json"
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
    preview = [
        "# GPT-2 Hierarchy Preview / 层级图预览",
        "",
        f"Model: `{ir.model_id}` · revision: `{ir.metadata['revision']}`",
        "",
        "这些图由同一份 Hierarchical Model IR 生成。边和 symbolic shapes 来自 GPT-2 semantic rules；未执行 runtime forward。",
        "",
        "重复节点的 child graph 是 block 0 的 template；其他实例的参数保留在 Raw Structure。",
        "",
    ]
    for graph in ir.root_graph.walk():
        rendered = mermaid(graph)
        path = output_directory / f"{graph.id}.mmd"
        path.write_text(rendered, encoding="utf-8")
        paths.append(path)
        preview.extend([f"## {graph.id}", "", "```mermaid", rendered.rstrip(), "```", ""])
    preview_path = preview_directory / "GPT2_PREVIEW.md"
    preview_path.write_text("\n".join(preview), encoding="utf-8")
    paths.append(preview_path)
    return paths
