import type { GraphNode } from "../api/contract";

export function nodeTitle(node: GraphNode) {
  // Reported paths distinguish repeated native class labels without inventing
  // semantic classification or dataflow from their names.
  if (node.module_path && ["module", "unknown", "unclassified"].includes(node.kind)) {
    const part = node.module_path.split(".").filter(Boolean).at(-1);
    if (part && /^\d+$/.test(part)) {
      return `${node.label || node.module_type || "Module"} [${part}]`;
    }
    return part ?? node.label;
  }
  return node.label;
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
