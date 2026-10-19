import { useEffect, useMemo, useState } from "react";
import { ArrowRight, ExternalLink, Search } from "lucide-react";
import { useNavigate } from "react-router-dom";
import "./benchmark-catalog.css";

type BenchmarkResult = {
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
  warning_count?: number;
  warnings?: string[];
  reason_codes?: string[];
  error?: { message?: string };
  graph_count?: number;
  tensor_count?: number;
  parameter_counts?: { unique?: number | null };
  coverage?: Record<string, number | null>;
};

type BenchmarkReport = {
  schema_version: 1;
  generated_at: string;
  engine_version: string;
  manifest_sha256: string;
  results: BenchmarkResult[];
};

const pageSize = 24;
const number = (value: number | null | undefined) =>
  value == null ? "Unknown" : value.toLocaleString();

function statusLabel(status: string) {
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

function dateTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export default function BenchmarkCatalog() {
  const navigate = useNavigate();
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [outcome, setOutcome] = useState("all");
  const [page, setPage] = useState(0);

  const load = async (signal?: AbortSignal) => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/benchmark-catalog.json", { signal });
      if (!response.ok)
        throw new Error(`Report request failed (${response.status}).`);
      const payload = (await response.json()) as BenchmarkReport;
      if (payload.schema_version !== 1 || !Array.isArray(payload.results))
        throw new Error(
          "The benchmark report has an unsupported or invalid format.",
        );
      setReport(payload);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not load the benchmark report.",
      );
      setReport(null);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, []);

  const categories = useMemo(
    () =>
      [...new Set(report?.results.map((item) => item.category) ?? [])].sort(),
    [report],
  );
  const outcomes = useMemo(
    () => [...new Set(report?.results.map((item) => item.status) ?? [])].sort(),
    [report],
  );
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filtered = useMemo(() => {
    return (report?.results ?? []).filter((item) => {
      const matchesQuery =
        !normalizedQuery ||
        [
          item.model_id,
          item.category,
          item.architecture ?? "",
          item.declared_task ?? "",
        ]
          .join(" ")
          .toLocaleLowerCase()
          .includes(normalizedQuery);
      return (
        matchesQuery &&
        (category === "all" || item.category === category) &&
        (outcome === "all" || item.status === outcome)
      );
    });
  }, [report, normalizedQuery, category, outcome]);
  const pageCount = Math.ceil(filtered.length / pageSize);
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);

  useEffect(() => setPage(0), [normalizedQuery, category, outcome]);

  const startInspection = (item: BenchmarkResult) => {
    const repoPath = encodeURIComponent(item.model_id).replaceAll("%2F", "/");
    const reference =
      item.reference ||
      `https://huggingface.co/${repoPath}/tree/${encodeURIComponent(item.revision)}`;
    navigate(`/app?model=${encodeURIComponent(reference)}`);
  };

  const counts = (value: string, key: "category" | "status") =>
    (report?.results ?? []).filter((item) => item[key] === value).length;
  const statusCounts = outcomes.map((status) => ({
    status,
    count: counts(status, "status"),
  }));

  if (loading)
    return (
      <section className="benchmark-state" role="status">
        Loading measured benchmark results…
      </section>
    );
  if (error)
    return (
      <section className="benchmark-state" role="alert">
        <h2>Could not load benchmark results</h2>
        <p>{error}</p>
        <button
          type="button"
          className="catalog-empty-btn"
          onClick={() => void load()}
        >
          Retry
        </button>
      </section>
    );
  if (!report) return null;

  return (
    <section className="benchmark-content" aria-labelledby="benchmark-heading">
      <div className="benchmark-report-meta">
        <div>
          <h2 id="benchmark-heading">Measured compatibility attempts</h2>
          <p>
            {report.results.length} pinned repositories · engine{" "}
            {report.engine_version} · generated {dateTime(report.generated_at)}
          </p>
        </div>
        <code title="Pinned input manifest SHA-256">
          Manifest {report.manifest_sha256}
        </code>
      </div>

      <div className="benchmark-summary" aria-label="Outcome totals">
        <div>
          <strong>{report.results.length}</strong>
          <span>Attempts</span>
        </div>
        {statusCounts.map(({ status, count }) => (
          <div key={status}>
            <strong>{count}</strong>
            <span>{statusLabel(status)}</span>
          </div>
        ))}
      </div>

      <div className="benchmark-filters" aria-label="Filter benchmark results">
        <label className="catalog-search-box">
          <Search size={16} className="search-icon" aria-hidden="true" />
          <input
            type="search"
            className="catalog-search-input"
            aria-label="Search benchmark results"
            placeholder="Search model, architecture, category, or task…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <label>
          Task category
          <select
            aria-label="Filter by task category"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="all">
              All categories ({report.results.length})
            </option>
            {categories.map((value) => (
              <option key={value} value={value}>
                {value} ({counts(value, "category")})
              </option>
            ))}
          </select>
        </label>
        <label>
          Outcome
          <select
            aria-label="Filter by outcome"
            value={outcome}
            onChange={(event) => setOutcome(event.target.value)}
          >
            <option value="all">All outcomes ({report.results.length})</option>
            {outcomes.map((value) => (
              <option key={value} value={value}>
                {statusLabel(value)} ({counts(value, "status")})
              </option>
            ))}
          </select>
        </label>
      </div>

      <p className="benchmark-result-count" aria-live="polite">
        Showing {filtered.length ? page * pageSize + 1 : 0}–
        {Math.min((page + 1) * pageSize, filtered.length)} of {filtered.length}{" "}
        matching results
      </p>
      {filtered.length === 0 ? (
        <div className="benchmark-state">
          <h3>No measured results match these filters</h3>
          <p>Change the search or filters to view recorded attempts.</p>
        </div>
      ) : (
        <div className="catalog-cards-grid benchmark-grid">
          {visible.map((item) => {
            const repoPath = encodeURIComponent(item.model_id).replaceAll(
              "%2F",
              "/",
            );
            const hfHref =
              item.reference ||
              `https://huggingface.co/${repoPath}/tree/${encodeURIComponent(item.revision)}`;
            const warnings = item.warnings ?? [];
            return (
              <article
                key={`${item.model_id}@${item.revision}`}
                className="catalog-detail-card benchmark-card"
              >
                <div className="card-top-row">
                  <span className={`benchmark-status status-${item.status}`}>
                    {statusLabel(item.status)}
                  </span>
                  <span className="card-family-tag">{item.category}</span>
                </div>
                <h3 className="card-model-title">{item.model_id}</h3>
                <p className="benchmark-task">
                  Declared task: {item.declared_task ?? "Not declared"}
                </p>
                <div className="benchmark-facts">
                  <div>
                    <span>Architecture</span>
                    <strong>{item.architecture ?? "Not reconstructed"}</strong>
                  </div>
                  <div>
                    <span>Graphs</span>
                    <strong>{number(item.graph_count)}</strong>
                  </div>
                  <div>
                    <span>Tensors</span>
                    <strong>{number(item.tensor_count)}</strong>
                  </div>
                  <div>
                    <span>Unique parameters</span>
                    <strong>{number(item.parameter_counts?.unique)}</strong>
                  </div>
                </div>
                <div className="benchmark-revision">
                  <span>Pinned revision</span>
                  <code>{item.revision}</code>
                </div>
                <div className="benchmark-card-actions">
                  <button
                    type="button"
                    className="card-inspect-btn"
                    onClick={() => startInspection(item)}
                  >
                    <span>Inspect pinned model</span>
                    <ArrowRight size={14} />
                  </button>
                  <a
                    className="card-hf-btn"
                    href={hfHref}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`View ${item.model_id} pinned revision on Hugging Face`}
                  >
                    <ExternalLink size={14} />
                  </a>
                </div>
                <details className="benchmark-details">
                  <summary>Evidence and details</summary>
                  <dl>
                    <dt>Revision</dt>
                    <dd>{item.revision}</dd>
                    <dt>Reference</dt>
                    <dd>
                      <a href={hfHref} target="_blank" rel="noreferrer">
                        {hfHref}
                      </a>
                    </dd>
                    <dt>Started</dt>
                    <dd>{dateTime(item.started_at)}</dd>
                    <dt>Finished</dt>
                    <dd>{dateTime(item.finished_at)}</dd>
                    <dt>Engine</dt>
                    <dd>{item.engine_version}</dd>
                    <dt>Warnings</dt>
                    <dd>{item.warning_count ?? warnings.length}</dd>
                    <dt>Reason codes</dt>
                    <dd>{item.reason_codes?.join(", ") || "None recorded"}</dd>
                    {item.error?.message && (
                      <>
                        <dt>Recorded error</dt>
                        <dd>{item.error.message}</dd>
                      </>
                    )}
                  </dl>
                  {warnings.length > 0 && (
                    <div>
                      <h4>Warnings</h4>
                      <ul>
                        {warnings.map((warning, index) => (
                          <li key={`${index}-${warning}`}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {item.coverage && (
                    <div className="benchmark-coverage">
                      <h4>Coverage evidence</h4>
                      {Object.entries(item.coverage).map(([key, value]) => (
                        <div key={key}>
                          <span>{key.replaceAll("_", " ")}</span>
                          <strong>
                            {value == null
                              ? "Unknown"
                              : `${(value * 100).toFixed(1)}%`}
                          </strong>
                        </div>
                      ))}
                    </div>
                  )}
                </details>
              </article>
            );
          })}
        </div>
      )}
      {pageCount > 1 && (
        <nav className="catalog-pagination" aria-label="Benchmark result pages">
          <button
            type="button"
            className="catalog-empty-btn"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </button>
          <span aria-live="polite">
            Page {page + 1} of {pageCount}
          </span>
          <button
            type="button"
            className="catalog-empty-btn"
            disabled={page + 1 >= pageCount}
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </nav>
      )}
      <p className="benchmark-disclaimer">
        These results describe static metadata and bounded reconstruction
        attempts. They make no claim about inference speed or model accuracy.
      </p>
    </section>
  );
}
