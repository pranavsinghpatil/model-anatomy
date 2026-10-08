import type {
  AnalysisDetail,
  AnalysesResponse,
  ArchitectureGraph,
  Health,
  ExportFormat,
  NodeDetail,
  ExamplesResponse,
  CreateAnalysisRequest,
  QuestionRequest,
  QuestionAnswer,
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
  examples: (signal?: AbortSignal) =>
    request<ExamplesResponse>("/examples", { signal }),
  analyses: (signal?: AbortSignal) =>
    request<AnalysesResponse>("/analyses", { signal }),
  analyze: (model: string) =>
    request<AnalysisDetail>("/analyses", {
      method: "POST",
      body: JSON.stringify({ model } satisfies CreateAnalysisRequest),
    }),
  analysis: (id: string, signal?: AbortSignal) =>
    request<AnalysisDetail>(`/analyses/${segment(id)}`, { signal }),
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
