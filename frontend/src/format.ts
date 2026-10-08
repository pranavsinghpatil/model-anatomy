import type { Dimension } from "./api/contract";

export function count(
  value: number | null | undefined,
  compact = false,
): string {
  if (value == null || !Number.isFinite(value)) return "Unknown";
  return new Intl.NumberFormat(
    "en-US",
    compact ? { notation: "compact", maximumFractionDigits: 2 } : {},
  ).format(value);
}
export function shape(value: Dimension[] | null | undefined): string {
  if (value == null) return "Unknown shape";
  if (!value.length) return "Scalar";
  return `[${value.join(", ")}]`;
}
export function valueText(value: unknown): string {
  if (value == null) return "Unknown";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
export function time(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Unknown time"
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
export const scopedNodeId = (graphId: string, nodeId: string) =>
  JSON.stringify([graphId, nodeId]);
