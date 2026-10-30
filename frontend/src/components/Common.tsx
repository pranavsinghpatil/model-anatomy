import { AlertCircle, LoaderCircle, RefreshCw } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { ApiError } from "../api/client";
import { feedbackFor, type FeedbackContext } from "../state/feedback-messages";
import { CopyButton } from "./ui";

export function IconButton({
  label,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className="icon-button"
      aria-label={label}
      title={label}
      {...props}
    >
      {children}
    </button>
  );
}

const statusLabels: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  cancelling: "Cancelling",
  completed: "Completed",
  partial: "Partially complete",
  failed: "Failed",
  cancelled: "Cancelled",
  timed_out: "Timed out",
  unsupported: "Unsupported",
  inspected: "Artifacts inspected",
};

export function Status({ status }: { status: string }) {
  const label = statusLabels[status] ?? status.replaceAll("_", " ");
  return (
    <span
      className={`status status-${status}`}
      role="status"
      aria-label={`Inspection status: ${label}`}
    >
      <span className="status-dot" />
      {label}
    </span>
  );
}

export function Loading({ text = "Loading" }: { text?: string }) {
  return (
    <div className="loading" role="status">
      <LoaderCircle size={16} className="spin" />
      {text}
    </div>
  );
}

export function ErrorNotice({
  error,
  context = "request",
  retry,
  actions,
  id,
}: {
  error: Error;
  context?: FeedbackContext;
  retry?: () => void;
  actions?: ReactNode;
  id?: string;
}) {
  const feedback = feedbackFor(error, context);
  return (
    <div className="error-notice" role="alert" id={id}>
      <AlertCircle size={16} />
      <div className="error-notice-content">
        <strong>{feedback.title}</strong>
        <p>{feedback.description}</p>
        {error instanceof ApiError && feedback.technical && (
          <details className="error-technical">
            <summary>Technical details</summary>
            <code>{feedback.technical}</code>
            {error.requestId && (
              <CopyButton value={error.requestId} label="Copy request ID" />
            )}
          </details>
        )}
      </div>
      <div className="error-notice-actions">
        {actions}
        {retry && (
          <IconButton label="Retry request" onClick={retry}>
            <RefreshCw size={15} />
          </IconButton>
        )}
      </div>
    </div>
  );
}

export function Warnings({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <details className="warnings">
      <summary>
        <AlertCircle size={14} />
        {warnings.length} {warnings.length === 1 ? "warning" : "warnings"}
      </summary>
      <ul>
        {warnings.map((warning, i) => (
          <li key={i}>{warning}</li>
        ))}
      </ul>
    </details>
  );
}
