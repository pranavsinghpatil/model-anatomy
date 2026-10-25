import type { DiscoveryCatalogResult } from "../api/client";

export function statusLabel(status: string) {
  if (status === "reported_complete")
    return "Reported complete · module anatomy";
  if (status === "reported_partial") return "Reported partial · module anatomy";
  if (status === "out_of_scope") return "Outside scope · no Safetensors/access";
  if (status === "reported_failure")
    return "Reported failure · teammate validation";
  if (status === "completed") return "Complete static report";
  if (status === "partial") return "Partial graph";
  if (status === "inspected") return "Artifacts inspected";
  const failures: Record<string, string> = {
    http_access: "HF request failed",
    network: "Network request failed",
    artifact_error: "Artifact processing failed",
    unsupported: "Reconstruction unavailable",
    invalid_report: "Report validation failed",
    failure: "Inspection failed",
    memorylimit: "Memory limit reached",
    timeout: "Time limit reached",
  };
  return failures[status] ?? status.replaceAll("_", " ");
}

export function successful(item: DiscoveryCatalogResult) {
  if (item.source === "external_validation") return false;
  return ["completed", "partial", "inspected"].includes(item.status);
}

export function provenanceLabel(item: DiscoveryCatalogResult) {
  if (item.source === "external_validation") return "Teammate CSV";
  if (item.source === "benchmark") return "Benchmark attempt";
  if (!successful(item)) return "Inspection attempt";
  return item.stored_report_id ? "Saved inspection" : "Inspection outcome";
}

export function outcomeLabel(item: DiscoveryCatalogResult) {
  if (item.source === "external_validation") return statusLabel(item.status);
  return item.source === "inspection" && !successful(item)
    ? `Recorded failure · ${statusLabel(item.status)}`
    : statusLabel(item.status);
}

export function dateTime(value: string | null | undefined) {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function publicationFileUrl(
  repository: string | null | undefined,
  commit: string | null | undefined,
  path: string | null | undefined,
) {
  if (!repository || !commit || !path) return null;
  const encodedPath = path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `${repository.replace(/\/$/, "")}/blob/${encodeURIComponent(commit)}/${encodedPath}`;
}

export function hfUrl(item: DiscoveryCatalogResult) {
  if (item.reference) return item.reference;
  const repo = `https://huggingface.co/${encodeURIComponent(item.model_id).replaceAll("%2F", "/")}`;
  return item.revision
    ? `${repo}/tree/${encodeURIComponent(item.revision)}`
    : repo;
}

export function recordKey(item: DiscoveryCatalogResult) {
  return [
    item.model_id,
    item.revision ?? "revision-unknown",
    item.artifact ?? "artifact-unspecified",
    item.selection_id ?? "selection-unspecified",
    item.explicit_artifact ? "explicit" : "implicit",
    item.source,
  ].join("|");
}

export function scopeStatus(item: DiscoveryCatalogResult) {
  if (item.scope_status) return item.scope_status;
  if (item.status === "out_of_scope") return "out_of_scope";
  if (
    item.external_observations?.some(
      (observation) => observation.status === "out_of_scope",
    )
  )
    return "out_of_scope";
  if (
    (item.source === "external_validation" &&
      ["reported_complete", "reported_partial", "reported_failure"].includes(
        item.status,
      )) ||
    item.external_observations?.some((observation) =>
      ["reported_complete", "reported_partial", "reported_failure"].includes(
        observation.status ?? "",
      ),
    )
  )
    return "in_scope";
  return "unknown";
}

export function hasOutOfScopeExternalHistory(item: DiscoveryCatalogResult) {
  return (
    item.external_observations?.some(
      (observation) => observation.status === "out_of_scope",
    ) ?? false
  );
}

export function scopeReasonLabel(reason: string | null | undefined) {
  if (reason === "no_safetensors" || reason === "safetensors_unavailable")
    return "Safetensors unavailable";
  if (reason === "access_restricted") return "Access restricted";
  return reason ? reason.replaceAll("_", " ") : "Unknown";
}
