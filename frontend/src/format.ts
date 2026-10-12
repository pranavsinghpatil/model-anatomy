import type { Dimension } from "./api/contract";

export type ReportDimension = "hidden" | "heads" | "layers" | "vocabulary";

export function reportDimensionKeys(analysis: {
  input_names?: string[];
  dimensions: { vocabulary?: number | null };
}): ReportDimension[] {
  const hasTextInputs =
    analysis.input_names?.some((name) =>
      ["input_ids", "decoder_input_ids"].includes(name),
    ) ?? false;
  const hasLegacyVocabulary =
    typeof analysis.dimensions.vocabulary === "number";
  return [
    "hidden",
    "heads",
    "layers",
    ...(hasTextInputs || hasLegacyVocabulary ? (["vocabulary"] as const) : []),
  ];
}

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
