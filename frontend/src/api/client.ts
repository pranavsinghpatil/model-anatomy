import type {
  AnalysisDetail,
  ArtifactInspectionResponse,
  AnalysesResponse,
  ArchitectureGraph,
  Health,
  ExportFormat,
  NodeDetail,
  CreateAnalysisRequest,
  QuestionRequest,
  QuestionAnswer,
  CatalogResponse,
} from "./contract";

const prefix = "/api/v1";
const segment = encodeURIComponent;
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export type BenchmarkCatalogResult = {
  model_id: string;
  category: string;
  revision: string;
  status: string;
  architecture: string | null;
  reference: string | null;
  declared_task: string | null;
  started_at: string;
  finished_at: string;
  engine_version: string;
  stored_report_id: string | null;
  [key: string]: unknown;
};

export type BenchmarkCatalogResponse = {
  schema_version: 1;
  generated_at: string;
  engine_version: string;
  manifest_sha256: string;
  results: BenchmarkCatalogResult[];
  baseline_cases?: number;
  recheck_attempts?: number;
};

export type DiscoveryObservation = {
  status?: string;
  engine_version?: string | null;
  finished_at?: string | null;
  case_identity?: string;
  revision?: string | null;
  reference?: string | null;
  artifact?: string | null;
  stored_report_id?: string | null;
};

export type DiscoveryCatalogResult = {
  model_id: string;
  category: string;
  revision: string | null;
  reference: string | null;
  stored_report_id: string | null;
  status: string;
  architecture: string | null;
  declared_task: string | null;
  started_at: string | null;
  finished_at: string | null;
  engine_version: string | null;
  warning_count: number | null;
  graph_count: number | null;
  tensor_count: number | null;
  parameter_counts?: { unique?: number | null } | null;
  coverage?: Record<string, number | null> | null;
  dimensions?: Record<string, number | number[] | null> | null;
  artifact?: string | null;
  explicit_artifact?: boolean | null;
  artifact_verified?: boolean | null;
  selection_id?: string | null;
  category_source?: string | null;
  declared_task_source?: string | null;
  revision_source?: string | null;
  source: "inspection" | "benchmark";
  run_kind?: "baseline" | "recheck";
  previous_observations?: DiscoveryObservation[];
  benchmark_observations?: DiscoveryObservation[];
  warnings?: string[];
  diagnostic_notes?: string[];
  reason_codes?: string[];
  error?: { message?: string } | null;
};

export type DiscoveryCatalogResponse = {
  results: DiscoveryCatalogResult[];
  total: number;
  status_counts: Record<string, number>;
  generated_at: string;
  benchmark_publication_id: string | null;
};

async function errorFrom(response: Response): Promise<ApiError> {
  const payload = await response.json().catch(() => null);
  const error = payload?.error;
  return new ApiError(
    response.status,
    error?.code || "http_error",
    error?.message || `Request failed (${response.status})`,
    error?.request_id,
  );
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  options: { prefix?: string; acceptedStatuses?: number[] } = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${options.prefix ?? prefix}${path}`, {
      ...init,
      credentials: "include",
      headers: {
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiError(
      0,
      "network_error",
      "Cannot reach the analysis service.",
    );
  }
  if (!response.ok && !options.acceptedStatuses?.includes(response.status))
    throw await errorFrom(response);
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(
      response.status,
      "invalid_response",
      "The service returned an invalid response.",
    );
  }
}

export const api = {
  health: async (signal?: AbortSignal): Promise<Health> => {
    const health = await request<Health>(
      "/health",
      { signal },
      {
        prefix: "",
        acceptedStatuses: [503],
      },
    );
    if (
      !health ||
      !["ok", "degraded"].includes(health.status) ||
      health.api_version !== "v1" ||
      !health.checks ||
      [
        health.checks.database,
        health.checks.storage,
        health.checks.worker,
        health.analysis_available,
        health.ai_available,
      ].some((value) => typeof value !== "boolean") ||
      !["local", "supabase"].includes(health.storage_backend) ||
      !(health.reason === null || typeof health.reason === "string")
    )
      throw new ApiError(
        0,
        "invalid_response",
        "The service returned an invalid health response.",
      );
    return health;
  },
  analyses: (signal?: AbortSignal) =>
    request<AnalysesResponse>("/analyses", { signal }),
  catalog: (q: string, offset = 0, limit = 24, signal?: AbortSignal) => {
    const params = new URLSearchParams({
      q,
      offset: String(offset),
      limit: String(limit),
    });
    return request<CatalogResponse>(`/catalog?${params}`, { signal });
  },
  benchmarkCatalog: (signal?: AbortSignal) =>
    request<BenchmarkCatalogResponse>("/benchmarks/catalog", { signal }),
  discoveryCatalog: async (signal?: AbortSignal) => {
    const payload = await request<DiscoveryCatalogResponse>(
      "/catalog/discovery",
      { signal },
    );
    if (
      !payload ||
      !Array.isArray(payload.results) ||
      typeof payload.total !== "number" ||
      !payload.status_counts ||
      typeof payload.status_counts !== "object" ||
      typeof payload.generated_at !== "string" ||
      !(
        payload.benchmark_publication_id === null ||
        typeof payload.benchmark_publication_id === "string"
      )
    )
      throw new ApiError(
        0,
        "invalid_response",
        "The model catalog returned an invalid response.",
      );
    return payload;
  },
  inspectCatalog: (id: string) =>
    request<AnalysisDetail>(`/catalog/${segment(id)}/inspect`, {
      method: "POST",
    }),
  analyze: (model: string) =>
    request<AnalysisDetail>("/analyses", {
      method: "POST",
      body: JSON.stringify({
        model,
      } satisfies CreateAnalysisRequest),
    }),
  analysis: (id: string, signal?: AbortSignal) =>
    request<AnalysisDetail>(`/analyses/${segment(id)}`, { signal }),
  artifacts: (id: string, offset = 0, limit = 50, signal?: AbortSignal) =>
    request<ArtifactInspectionResponse>(
      `/analyses/${segment(id)}/artifacts?offset=${offset}&limit=${limit}`,
      { signal },
    ),
  cancel: (id: string) =>
    request<AnalysisDetail>(`/analyses/${segment(id)}`, { method: "DELETE" }),
  graph: (id: string, graphId: string, signal?: AbortSignal) =>
    request<ArchitectureGraph>(
      `/analyses/${segment(id)}/graphs/${segment(graphId)}`,
      { signal },
    ),
  node: (
    id: string,
    graphId: string,
    nodeId: string,
    offset = 0,
    signal?: AbortSignal,
  ) =>
    request<NodeDetail>(
      `/analyses/${segment(id)}/nodes/${segment(graphId)}/${segment(nodeId)}?offset=${offset}&limit=50`,
      { signal },
    ),
  question: (id: string, question: string, graphId?: string, nodeId?: string) =>
    request<QuestionAnswer>(`/analyses/${segment(id)}/questions`, {
      method: "POST",
      body: JSON.stringify({
        question,
        ...(graphId ? { graph_id: graphId } : {}),
        ...(nodeId ? { node_id: nodeId } : {}),
      } satisfies QuestionRequest),
    }),
  export: async (id: string, format: ExportFormat) => {
    const response = await fetch(
      `${prefix}/analyses/${segment(id)}/exports/${format}`,
      { credentials: "include" },
    ).catch(() => {
      throw new ApiError(
        0,
        "network_error",
        "Cannot reach the export service.",
      );
    });
    if (!response.ok) throw await errorFrom(response);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const extension = {
      json: "json",
      mermaid: "mmd",
      markdown: "md",
      diagnostics: "json",
      audit: "json",
      zip: "zip",
    }[format];
    link.download = `model-anatomy-${id.replace(/[^a-zA-Z0-9_-]/g, "_")}-${format}.${extension}`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  },
};

export const pending = (status?: string) =>
  status === "queued" || status === "running";
export const graphReady = (status?: string) =>
  status === "completed" || status === "partial";
export const retryQuery = (count: number, error: Error) =>
  count < 1 &&
  (!(error instanceof ApiError) || error.status === 0 || error.status >= 500);
