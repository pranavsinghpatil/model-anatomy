import type { GraphEdge, GraphNode } from "../api/contract";
import { shape } from "../format";

export function nodeTitle(node: GraphNode) {
  // Reported paths distinguish repeated native class labels without inventing
  // semantic classification or dataflow from their names.
  if (
    node.module_path &&
    ["module", "unknown", "unclassified"].includes(node.kind)
  ) {
    const part = node.module_path.split(".").filter(Boolean).at(-1);
    if (part && /^\d+$/.test(part)) {
      return `${node.label || node.module_type || "Module"} [${part}]`;
    }
    return part ?? node.label;
  }
  return node.label;
}
const relationNames: Record<string, string> = {
  dataflow: "Data flow",
  conditional_dependency: "Conditional dependency",
  parameter_ownership: "Parameter ownership",
  dependency: "Dependency",
  module_dependency: "Module dependency",
  source_order: "Source order",
  structural: "Structural relationship",
};

export function relationLabel(relation: string) {
  return (
    relationNames[relation] ??
    relation
      .replace(/[_-]+/g, " ")
      .replace(/^\w/, (letter) => letter.toUpperCase())
  );
}

export function edgeLabel(edge: GraphEdge, detail = false) {
  const name = edge.label?.trim() || edge.tensor?.semantic_name?.trim();
  const parts = [name || relationLabel(edge.relation)];
  if (edge.tensor?.shape) parts.push(shape(edge.tensor.shape));
  if (detail) {
    if (name) parts.push(relationLabel(edge.relation));
    const source = edge.source_port || "Unspecified port";
    const target = edge.target_port || "Unspecified port";
    parts.push(`${source} → ${target}`);
  }
  return parts.join(" · ");
}

export function nodeCategory(kind: string) {
  const value = kind.toLowerCase();
  if (value === "input" || value === "output") return "boundary";
  if (value.includes("attention")) return "attention";
  if (value.includes("position") || value.includes("rotary")) return "position";
  if (value.includes("norm")) return "normalization";
  if (
    ["mlp", "feed_forward", "linear", "projection", "head"].some((part) =>
      value.includes(part),
    )
  )
    return "projection";
  if (["block", "model", "layer"].some((part) => value.includes(part)))
    return "structure";
  if (value === "embedding") return "embedding";
  if (["backbone", "container", "module"].includes(value)) return "structure";
  return "other";
}
export function nodeSize(node: GraphNode, density: "compact" | "detailed") {
  const boundary = node.kind === "input" || node.kind === "output";
  return {
    width: boundary ? 224 : 264,
    height: boundary
      ? density === "compact"
        ? 60
        : 96
      : density === "detailed"
        ? node.child_graph_id
          ? 212
          : 176
        : node.child_graph_id
          ? 120
          : 84,
  };
}
