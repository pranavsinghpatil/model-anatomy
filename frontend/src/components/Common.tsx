import { AlertCircle, LoaderCircle, RefreshCw } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { ApiError } from "../api/client";

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
export function Status({ status }: { status: string }) {
  return (
    <span
      className={`status status-${status}`}
      role="status"
      aria-label={`Analysis status: ${status.replaceAll("_", " ")}`}
    >
      <span className="status-dot" />
      {status.replaceAll("_", " ")}
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
  retry,
}: {
  error: Error;
  retry?: () => void;
}) {
  return (
    <div className="error-notice" role="alert">
      <AlertCircle size={16} />
      <div>
        <strong>
          {error instanceof ApiError && error.status === 410
            ? "Analysis expired"
            : "Request unavailable"}
        </strong>
        <p>{error.message}</p>
        {error instanceof ApiError && (
          <small>
            {error.code}
            {error.requestId ? ` / ${error.requestId}` : ""}
          </small>
        )}
      </div>
      {retry && (
        <IconButton label="Retry request" onClick={retry}>
          <RefreshCw size={15} />
        </IconButton>
      )}
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
