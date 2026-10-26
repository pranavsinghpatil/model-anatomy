import { ApiError } from "../api/client";

export type FeedbackContext =
  | "request"
  | "inspection"
  | "submit"
  | "export"
  | "cancel"
  | "question"
  | "citation"
  | "catalog"
  | "graph";
export type FeedbackMessage = {
  title: string;
  description: string;
  technical?: string;
};

const contextTitles: Record<FeedbackContext, string> = {
  request: "Could not load this information",
  inspection: "Inspection could not be opened",
  submit: "Inspection could not be started",
  export: "Export could not be created",
  cancel: "Inspection could not be cancelled",
  question: "Question could not be answered",
  citation: "Source could not be opened",
  catalog: "Catalog is unavailable",
  graph: "Architecture view is unavailable",
};
const codeDescriptions: Record<string, string> = {
  invalid_model:
    "Use a Hugging Face model ID (namespace/model) or a https://huggingface.co/ URL.",
  validation_error: "Review the submitted information and try again.",
  worker_unavailable:
    "New inspections are paused because the worker is unavailable. Saved inspections remain open.",
  queue_full: "The inspection queue is full. Try again later.",
  rate_limited: "The service allowance has been reached.",
  expired: "This inspection has expired. Start a new inspection to continue.",
  not_found: "The requested item could not be found.",
  network_error:
    "The service could not be reached. Check your connection and try again.",
};
export function isModelValidationError(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    ["invalid_model", "validation_error"].includes(error.code)
  );
}
export function feedbackFor(
  error: unknown,
  context: FeedbackContext = "request",
): FeedbackMessage {
  const apiError = error instanceof ApiError ? error : undefined;
  const title =
    apiError?.code === "rate_limited"
      ? "Allowance reached"
      : context === "inspection" && apiError?.code === "not_found"
        ? "Inspection not found"
        : apiError?.code === "expired" || apiError?.status === 410
          ? "Inspection expired"
          : contextTitles[context];
  let description = apiError
    ? (codeDescriptions[apiError.code] ?? apiError.message)
    : error instanceof Error
      ? error.message
      : "An unexpected error occurred. Try again.";
  if (
    apiError?.message &&
    ["rate_limited", "queue_full"].includes(apiError.code)
  ) {
    description = apiError.message;
  }
  if (context === "submit" && apiError?.code === "network_error") {
    description =
      "We could not confirm whether the inspection started. Check Your inspections before trying again.";
  }
  if (context === "inspection" && apiError?.code === "not_found") {
    description =
      "This inspection is missing or no longer available. Start a new inspection to continue.";
  }
  const technical = apiError
    ? [
        apiError.code,
        apiError.requestId ? `Request ID ${apiError.requestId}` : "",
      ]
        .filter(Boolean)
        .join(" · ")
    : undefined;
  return { title, description, technical };
}
