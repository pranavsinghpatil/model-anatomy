import { Fragment, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, ExternalLink, Search } from "lucide-react";
import {
  ApiError,
  api,
  retryQuery,
  type DiscoveryCatalogResult,
  type DiscoveryCatalogResponse,
} from "../api/client";
import { DitherBackground } from "../components/DitherBackground";
import { EntryHeader, PageFooter } from "./Entry";
import "./benchmark-catalog.css";
import "./entry.css";

const pageSize = 24;
const number = (value: number | null | undefined) =>
  value == null ? "Unknown" : value.toLocaleString();

function statusLabel(status: string) {
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

function successful(item: DiscoveryCatalogResult) {
  if (item.source === "external_validation") return false;
  return ["completed", "partial", "inspected"].includes(item.status);
}

function provenanceLabel(item: DiscoveryCatalogResult) {
  if (item.source === "external_validation") return "Teammate CSV";
  if (item.source === "benchmark") return "Benchmark attempt";
  if (!successful(item)) return "Inspection attempt";
  return item.stored_report_id ? "Saved inspection" : "Inspection outcome";
}

function outcomeLabel(item: DiscoveryCatalogResult) {
  if (item.source === "external_validation") return statusLabel(item.status);
  return item.source === "inspection" && !successful(item)
    ? `Recorded failure · ${statusLabel(item.status)}`
    : statusLabel(item.status);
}

function dateTime(value: string | null | undefined) {
  if (!value) return "Unknown";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function publicationFileUrl(
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

function hfUrl(item: DiscoveryCatalogResult) {
  if (item.reference) return item.reference;
  const repo = `https://huggingface.co/${encodeURIComponent(item.model_id).replaceAll("%2F", "/")}`;
  return item.revision
    ? `${repo}/tree/${encodeURIComponent(item.revision)}`
    : repo;
}

function recordKey(item: DiscoveryCatalogResult) {
  return [
    item.model_id,
    item.revision ?? "revision-unknown",
    item.artifact ?? "artifact-unspecified",
    item.selection_id ?? "selection-unspecified",
    item.explicit_artifact ? "explicit" : "implicit",
    item.source,
  ].join("|");
}

function scopeStatus(item: DiscoveryCatalogResult) {
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

function hasOutOfScopeExternalHistory(item: DiscoveryCatalogResult) {
  return (
    item.external_observations?.some(
      (observation) => observation.status === "out_of_scope",
    ) ?? false
  );
}

function scopeReasonLabel(reason: string | null | undefined) {
  if (reason === "no_safetensors" || reason === "safetensors_unavailable")
    return "Safetensors unavailable";
  if (reason === "access_restricted") return "Access restricted";
  return reason ? reason.replaceAll("_", " ") : "Unknown";
}

export default function Catalog() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [outcome, setOutcome] = useState("all");
  const [scope, setScope] = useState("all");
  const [source, setSource] = useState("all");
  const [page, setPage] = useState(0);
  const [openingId, setOpeningId] = useState<string | null>(null);
  const [actionError, setActionError] = useState("");

  const catalog = useQuery({
    queryKey: ["catalog", "discovery"],
    queryFn: ({ signal }) => api.discoveryCatalog(signal),
    retry: retryQuery,
    staleTime: 0,
    refetchOnWindowFocus: true,
    refetchInterval: 15_000,
    refetchIntervalInBackground: false,
  });
  const inspect = useMutation({
    mutationFn: (reportId: string) => api.inspectCatalog(reportId),
    onSuccess: async (analysis) => {
      await queryClient.invalidateQueries({ queryKey: ["catalog"] });
      navigate(
        `/app/analyses/${encodeURIComponent(analysis.id)}?view=${analysis.status === "inspected" ? "artifacts" : "architecture"}`,
      );
    },
    onError: (error) =>
      setActionError(
        error instanceof ApiError
          ? error.message
          : "Could not open the saved inspection.",
      ),
  });

  const results = catalog.data?.results ?? [];
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const categories = useMemo(
    () => [...new Set(results.map((item) => item.category))].sort(),
    [results],
  );
  const filtered = useMemo(
    () =>
      results.filter((item) => {
        const matchesQuery =
          !normalizedQuery ||
          [
            item.model_id,
            item.category,
            item.architecture ?? "",
            item.declared_task ?? "",
            item.artifact ?? "",
            item.revision ?? "",
          ]
            .join(" ")
            .toLocaleLowerCase()
            .includes(normalizedQuery);
        return (
          matchesQuery &&
          (category === "all" || item.category === category) &&
          (outcome === "all" || item.status === outcome) &&
          (scope === "all" ||
            scopeStatus(item) === scope ||
            (scope === "out_of_scope" && hasOutOfScopeExternalHistory(item)) ||
            (scope === "unknown" && scopeStatus(item) === "unknown")) &&
          (source === "all" ||
            (source === "external" &&
              (item.source === "external_validation" ||
                (item.external_observations?.length ?? 0) > 0)) ||
            (source === "native" && item.source !== "external_validation"))
        );
      }),
    [results, normalizedQuery, category, outcome, scope, source],
  );
  const pageCount = Math.ceil(filtered.length / pageSize);
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);

  useEffect(
    () => setPage(0),
    [normalizedQuery, category, outcome, scope, source],
  );

  const startInspection = (item: DiscoveryCatalogResult) => {
    navigate(`/app?model=${encodeURIComponent(hfUrl(item))}`);
  };
  const openSavedReport = (reportId: string) => {
    setActionError("");
    setOpeningId(reportId);
    inspect.mutate(reportId, { onSettled: () => setOpeningId(null) });
  };
  const openRecord = (item: DiscoveryCatalogResult) => {
    setActionError("");
    if (item.stored_report_id) {
      openSavedReport(item.stored_report_id);
    } else {
      startInspection(item);
    }
  };

  const statusCounts = catalog.data?.status_counts ?? {};
  const statuses = [
    ...new Set([
      ...Object.keys(statusCounts),
      ...results.map((item) => item.status),
    ]),
  ].sort();
  const scopeCounts = results.reduce(
    (counts, item) => {
      if (
        scopeStatus(item) === "out_of_scope" ||
        hasOutOfScopeExternalHistory(item)
      )
        counts.out_of_scope += 1;
      if (scopeStatus(item) === "in_scope") counts.in_scope += 1;
      if (scopeStatus(item) === "unknown") counts.unknown += 1;
      return counts;
    },
    { in_scope: 0, out_of_scope: 0, unknown: 0 },
  );
  const errorText =
    catalog.error instanceof ApiError
      ? catalog.error.message
      : "Cannot reach the catalog service.";

  return (
    <div className="entry-page">
      <DitherBackground />
      <EntryHeader />
      <main id="main" tabIndex={-1} className="catalog-page-main">
        <header className="catalog-hero-section">
          <div className="catalog-hero-inner">
            <div className="catalog-eyebrow-badge">
              <span className="entry-badge-dot" />
              <span>Model Compatibility Evidence</span>
            </div>
            <h1 className="catalog-page-title">
              Model <span className="hero-highlight-amber">Catalog</span>
            </h1>
            <p className="catalog-page-subtitle">
              Browse saved inspections and measured benchmark attempts for
              pinned Hugging Face model artifacts.
            </p>
          </div>
        </header>

        <section
          className="benchmark-content"
          aria-labelledby="catalog-heading"
        >
          <div className="benchmark-report-meta">
            <div>
              <h2 id="catalog-heading">Unified model evidence</h2>
              <p>
                {catalog.data
                  ? `${catalog.data.total} catalog records · updated ${dateTime(catalog.data.generated_at)}`
                  : "Inspection and benchmark records"}
              </p>
            </div>
            <button
              type="button"
              className="catalog-empty-btn"
              aria-label="Refresh model catalog"
              disabled={catalog.isFetching}
              onClick={() => void catalog.refetch()}
            >
              {catalog.isFetching ? "Refreshing…" : "Refresh"}
            </button>
          </div>

          {catalog.data && (
            <div
              className="benchmark-summary"
              aria-label="Global catalog record totals"
            >
              <div>
                <strong>{catalog.data.total}</strong>
                <span>Catalog records</span>
              </div>
              {statuses.map((status) => (
                <div key={status}>
                  <strong>
                    {statusCounts[status] ??
                      results.filter((item) => item.status === status).length}
                  </strong>
                  <span>{statusLabel(status)}</span>
                </div>
              ))}
            </div>
          )}
          {catalog.data?.external_publications?.map((publication, index) => {
            const counts = publication.source_counts;
            const publicationUrl = publicationFileUrl(
              publication.source_repository,
              publication.source_commit,
              publication.source_path,
            );
            const inScope = counts?.observation_statuses
              ? Object.entries(counts.observation_statuses)
                  .filter(([status]) => status !== "out_of_scope")
                  .reduce((sum, [, count]) => sum + count, 0)
              : null;
            return (
              <p
                className="benchmark-disclaimer"
                key={`${publication.csv_sha256 ?? index}`}
              >
                Teammate CSV validation: {number(counts?.input_rows)} input rows
                · {number(counts?.unique_repository_ids)} unique repositories ·{" "}
                {number(inScope)} in scope ·{" "}
                {number(counts?.observation_statuses?.out_of_scope)} outside
                scope.
                {publicationUrl && (
                  <>
                    {" "}
                    Source:{" "}
                    <a href={publicationUrl} target="_blank" rel="noreferrer">
                      pinned CSV source
                    </a>
                  </>
                )}
              </p>
            );
          })}

          {actionError && (
            <p className="benchmark-action-error" role="alert">
              {actionError}
            </p>
          )}
          {catalog.isError && catalog.data && (
            <p className="catalog-refresh-error" role="status">
              Refresh failed. Showing the last loaded catalog. {errorText}
              <button type="button" onClick={() => void catalog.refetch()}>
                Retry
              </button>
            </p>
          )}
          {catalog.isPending && (
            <section className="benchmark-state" role="status">
              Loading the model catalog…
            </section>
          )}
          {catalog.isError && !catalog.data && (
            <section className="benchmark-state" role="alert">
              <h2>Could not load the model catalog</h2>
              <p>{errorText}</p>
              <button
                type="button"
                className="catalog-empty-btn"
                onClick={() => void catalog.refetch()}
              >
                Retry
              </button>
            </section>
          )}

          {catalog.data && (
            <>
              <div
                className="benchmark-filters"
                aria-label="Filter catalog records"
              >
                <label className="catalog-search-box">
                  <Search
                    size={16}
                    className="search-icon"
                    aria-hidden="true"
                  />
                  <input
                    type="search"
                    className="catalog-search-input"
                    aria-label="Search model catalog"
                    placeholder="Search model, architecture, category, task, revision…"
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
                      All categories ({catalog.data.total})
                    </option>
                    {categories.map((value) => (
                      <option key={value} value={value}>
                        {value} (
                        {
                          results.filter((item) => item.category === value)
                            .length
                        }
                        )
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
                    <option value="all">
                      All outcomes ({catalog.data.total})
                    </option>
                    {statuses.map((value) => (
                      <option key={value} value={value}>
                        {statusLabel(value)} (
                        {statusCounts[value] ??
                          results.filter((item) => item.status === value)
                            .length}
                        )
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  CSV scope and history
                  <select
                    aria-label="Filter by external scope"
                    value={scope}
                    onChange={(event) => setScope(event.target.value)}
                  >
                    <option value="all">All scope ({results.length})</option>
                    <option value="in_scope">
                      In scope (CSV) ({scopeCounts.in_scope})
                    </option>
                    <option value="out_of_scope">
                      Outside scope (CSV/history) ({scopeCounts.out_of_scope})
                    </option>
                    <option value="unknown">
                      No recorded exclusion ({scopeCounts.unknown})
                    </option>
                  </select>
                </label>
                <label>
                  Evidence source
                  <select
                    aria-label="Filter by evidence source"
                    value={source}
                    onChange={(event) => setSource(event.target.value)}
                  >
                    <option value="all">All evidence ({results.length})</option>
                    <option value="native">
                      Saved and benchmark (
                      {
                        results.filter(
                          (item) => item.source !== "external_validation",
                        ).length
                      }
                      )
                    </option>
                    <option value="external">
                      Teammate CSV (
                      {
                        results.filter(
                          (item) =>
                            item.source === "external_validation" ||
                            (item.external_observations?.length ?? 0) > 0,
                        ).length
                      }
                      )
                    </option>
                  </select>
                </label>
              </div>

              <p className="benchmark-result-count" aria-live="polite">
                Showing {filtered.length ? page * pageSize + 1 : 0}–
                {Math.min((page + 1) * pageSize, filtered.length)} of{" "}
                {filtered.length} matching records
              </p>

              {visible.length === 0 ? (
                <div className="benchmark-state">
                  <h3>
                    {catalog.data.total === 0
                      ? "No model evidence is available yet"
                      : "No records match these filters"}
                  </h3>
                  <p>
                    {catalog.data.total === 0
                      ? "Inspect a Hugging Face model to add its verified artifact facts."
                      : "Change the search or filters to view catalog records."}
                  </p>
                  <Link to="/#submit-title" className="catalog-empty-btn">
                    <span>Inspect a model</span>
                    <ArrowRight size={14} />
                  </Link>
                </div>
              ) : (
                <div className="catalog-cards-grid benchmark-grid">
                  {visible.map((item) => {
                    const hfHref = hfUrl(item);
                    const artifactPath =
                      item.explicit_artifact && item.artifact_verified === false
                        ? null
                        : item.artifact === "model.safetensors" &&
                            !item.explicit_artifact
                          ? null
                          : (item.artifact ??
                            item.reference?.match(
                              /\/(?:blob|resolve)\/[^/]+\/(.+)$/,
                            )?.[1]);
                    const benchmarkObservations =
                      item.benchmark_observations ?? [];
                    const previousObservations =
                      item.previous_observations ?? [];
                    const externalObservations =
                      item.external_observations ?? [];
                    const isExternal = item.source === "external_validation";
                    const displayedExternalObservations =
                      externalObservations.length > 0
                        ? externalObservations
                        : isExternal
                          ? [
                              {
                                source_status: item.source_status,
                                status: item.status,
                                scope_reason: item.scope_reason,
                                report_mode: item.report_mode,
                                graph_count: item.graph_count,
                                node_count: item.node_count,
                                source_repository: item.source_repository,
                                source_commit: item.source_commit,
                                source_path: item.source_path,
                                source_rows: item.source_rows,
                              },
                            ]
                          : [];
                    return (
                      <article
                        key={recordKey(item)}
                        className="catalog-detail-card benchmark-card"
                      >
                        <div className="card-top-row">
                          <span
                            className={`benchmark-status status-${item.status}`}
                          >
                            {outcomeLabel(item)}
                          </span>
                          <span className="card-family-tag">
                            {provenanceLabel(item)}
                          </span>
                          {item.run_kind === "recheck" && (
                            <span className="card-family-tag">Recheck</span>
                          )}
                          <span className="card-family-tag">
                            {item.category}
                          </span>
                        </div>
                        <h3 className="card-model-title">{item.model_id}</h3>
                        <p className="benchmark-task">
                          Declared task: {item.declared_task ?? "Not declared"}
                        </p>
                        {artifactPath && (
                          <p className="benchmark-scope">
                            {item.explicit_artifact && item.artifact_verified
                              ? "Selected artifact:"
                              : item.explicit_artifact
                                ? "Requested artifact:"
                                : "Artifact:"}{" "}
                            <code>{artifactPath}</code>
                          </p>
                        )}
                        {!isExternal &&
                          item.explicit_artifact &&
                          item.artifact_verified === false && (
                            <p className="benchmark-scope">
                              Requested artifact withheld
                            </p>
                          )}
                        {!isExternal &&
                          !item.explicit_artifact &&
                          item.artifact === "model.safetensors" && (
                            <p className="benchmark-scope">
                              Automatic checkpoint selection
                            </p>
                          )}
                        {(scopeStatus(item) === "out_of_scope" ||
                          hasOutOfScopeExternalHistory(item)) && (
                          <p className="benchmark-scope">
                            CSV scope exclusion:{" "}
                            {scopeReasonLabel(
                              item.scope_status === "out_of_scope" ||
                                item.status === "out_of_scope"
                                ? item.scope_reason
                                : item.external_observations?.find(
                                    (observation) =>
                                      observation.status === "out_of_scope",
                                  )?.scope_reason,
                            )}
                          </p>
                        )}
                        <div className="benchmark-facts">
                          <div>
                            <span>Architecture</span>
                            <strong>{item.architecture ?? "Unknown"}</strong>
                          </div>
                          <div>
                            <span>
                              {isExternal ? "Reported graphs" : "Graphs"}
                            </span>
                            <strong>{number(item.graph_count)}</strong>
                          </div>
                          {isExternal && item.node_count != null && (
                            <div>
                              <span>Reported nodes</span>
                              <strong>{number(item.node_count)}</strong>
                            </div>
                          )}
                          {!isExternal && (
                            <div>
                              <span>Inspected tensors</span>
                              <strong>{number(item.tensor_count)}</strong>
                            </div>
                          )}
                          {!isExternal && (
                            <div>
                              <span>Unique parameters</span>
                              <strong>
                                {number(item.parameter_counts?.unique)}
                              </strong>
                            </div>
                          )}
                        </div>
                        <div className="benchmark-revision">
                          <span>
                            {item.revision_source === "requested"
                              ? "Requested revision"
                              : "Pinned revision"}
                          </span>
                          <code>
                            {item.revision ??
                              "Unknown — repository reference only"}
                          </code>
                        </div>
                        <div className="benchmark-card-actions">
                          <button
                            type="button"
                            className="card-inspect-btn"
                            disabled={
                              inspect.isPending ||
                              (isExternal &&
                                scopeStatus(item) === "out_of_scope")
                            }
                            onClick={() => openRecord(item)}
                            aria-label={
                              isExternal && scopeStatus(item) === "out_of_scope"
                                ? `Outside inspection scope for ${item.model_id}`
                                : item.stored_report_id
                                  ? openingId === item.stored_report_id
                                    ? `Opening saved report for ${item.model_id}`
                                    : `Open saved inspection for ${item.model_id}`
                                  : `${isExternal ? "Inspect with current engine" : item.explicit_artifact && item.artifact_verified === false ? "Inspect pinned repository" : successful(item) ? "Inspect this revision" : "Retry inspection"} for ${item.model_id}`
                            }
                          >
                            <span>
                              {isExternal &&
                              scopeStatus(item) === "out_of_scope"
                                ? "Outside inspection scope"
                                : openingId === item.stored_report_id &&
                                    item.stored_report_id
                                  ? "Opening saved report…"
                                  : item.stored_report_id
                                    ? "Open saved report"
                                    : isExternal
                                      ? "Inspect with current engine"
                                      : successful(item)
                                        ? "Inspect this revision"
                                        : item.explicit_artifact &&
                                            item.artifact_verified === false
                                          ? "Inspect pinned repository"
                                          : "Retry inspection"}
                            </span>
                            <ArrowRight size={14} />
                          </button>
                          {item.stored_report_id && !successful(item) && (
                            <button
                              type="button"
                              className="card-retry-btn"
                              onClick={() => startInspection(item)}
                            >
                              Retry latest reference
                            </button>
                          )}
                          <a
                            className="card-hf-btn"
                            href={hfHref}
                            target="_blank"
                            rel="noreferrer"
                            aria-label={`View ${item.model_id} on Hugging Face`}
                          >
                            <ExternalLink size={14} />
                          </a>
                        </div>
                        <details className="benchmark-details">
                          <summary>Provenance and history</summary>
                          <dl>
                            <dt>Evidence source</dt>
                            <dd>
                              {item.source === "benchmark"
                                ? "Measured static benchmark attempt"
                                : !successful(item) && item.stored_report_id
                                  ? "Recorded failure; previous saved report available"
                                  : !successful(item) &&
                                      item.previous_observations?.some(
                                        (observation) =>
                                          observation.stored_report_id,
                                      )
                                    ? "Recorded failure; previous saved report available"
                                    : item.stored_report_id
                                      ? "Validated saved inspection"
                                      : successful(item)
                                        ? "Inspection outcome"
                                        : "Recorded inspection attempt"}
                            </dd>
                            {item.category_source && (
                              <>
                                <dt>Category source</dt>
                                <dd>{item.category_source}</dd>
                              </>
                            )}
                            {item.declared_task_source && (
                              <>
                                <dt>Task source</dt>
                                <dd>{item.declared_task_source}</dd>
                              </>
                            )}
                            {item.revision_source && (
                              <>
                                <dt>Revision source</dt>
                                <dd>{item.revision_source}</dd>
                              </>
                            )}
                            {item.run_kind && (
                              <>
                                <dt>Run kind</dt>
                                <dd>
                                  {item.run_kind === "recheck"
                                    ? "Recheck"
                                    : "Baseline"}
                                </dd>
                              </>
                            )}
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
                            <dd>{item.engine_version ?? "Unknown"}</dd>
                            <dt>Warning count</dt>
                            <dd>{number(item.warning_count)}</dd>
                            <dt>Reason codes</dt>
                            <dd>
                              {item.reason_codes?.join(", ") || "None recorded"}
                            </dd>
                            {item.error?.message && (
                              <>
                                <dt>Recorded error</dt>
                                <dd>{item.error.message}</dd>
                              </>
                            )}
                            {item.dimensions &&
                              Object.entries(item.dimensions).map(
                                ([key, value]) => (
                                  <Fragment key={key}>
                                    <dt>{key.replaceAll("_", " ")}</dt>
                                    <dd>
                                      {Array.isArray(value)
                                        ? value.map(number).join(" / ")
                                        : number(value as number | null)}
                                    </dd>
                                  </Fragment>
                                ),
                              )}
                          </dl>
                          {item.coverage && (
                            <div className="benchmark-coverage">
                              <h4>Coverage evidence</h4>
                              {Object.entries(item.coverage).map(
                                ([key, value]) => (
                                  <div key={key}>
                                    <span>{key.replaceAll("_", " ")}</span>
                                    <strong>
                                      {value == null
                                        ? "Unknown"
                                        : `${(value * 100).toFixed(1)}%`}
                                    </strong>
                                  </div>
                                ),
                              )}
                            </div>
                          )}
                          {displayedExternalObservations.length > 0 && (
                            <div className="benchmark-diagnostic-notes">
                              <h4>External validation history</h4>
                              <ul>
                                {displayedExternalObservations.map(
                                  (observation, index) => {
                                    const repo =
                                      observation.source_repository ??
                                      item.source_repository;
                                    const commit =
                                      observation.source_commit ??
                                      item.source_commit;
                                    const reference = publicationFileUrl(
                                      repo,
                                      commit,
                                      observation.source_path ??
                                        item.source_path,
                                    );
                                    return (
                                      <li
                                        key={`${commit ?? "unknown"}-${index}`}
                                      >
                                        {statusLabel(
                                          observation.status ?? "unknown",
                                        )}
                                        {observation.report_mode
                                          ? ` · mode ${observation.report_mode}`
                                          : ""}
                                        {observation.graph_count != null
                                          ? ` · ${number(observation.graph_count)} reported graphs`
                                          : ""}
                                        {observation.node_count != null
                                          ? ` · ${number(observation.node_count)} reported nodes`
                                          : ""}
                                        {observation.status ===
                                          "out_of_scope" &&
                                        observation.scope_reason
                                          ? ` · ${scopeReasonLabel(observation.scope_reason)}`
                                          : ""}
                                        {reference ? (
                                          <>
                                            {" "}
                                            ·{" "}
                                            <a
                                              href={reference}
                                              target="_blank"
                                              rel="noreferrer"
                                            >
                                              pinned CSV source
                                            </a>
                                          </>
                                        ) : (
                                          ""
                                        )}
                                        {observation.source_rows?.length
                                          ? ` · CSV row${observation.source_rows.length > 1 ? "s" : ""} ${observation.source_rows.join(", ")}`
                                          : ""}
                                      </li>
                                    );
                                  },
                                )}
                              </ul>
                            </div>
                          )}
                          {benchmarkObservations.length > 0 && (
                            <div className="benchmark-diagnostic-notes">
                              <h4>Benchmark history</h4>
                              <ul>
                                {benchmarkObservations
                                  .slice(0, 5)
                                  .map((observation, index) => (
                                    <li
                                      key={`${observation.finished_at ?? "unknown"}-${index}`}
                                    >
                                      {statusLabel(
                                        observation.status ?? "unknown",
                                      )}{" "}
                                      · engine{" "}
                                      {observation.engine_version ?? "Unknown"}{" "}
                                      · finished{" "}
                                      {dateTime(observation.finished_at)}
                                      {observation.artifact
                                        ? ` · ${observation.artifact}`
                                        : ""}
                                    </li>
                                  ))}
                              </ul>
                              {benchmarkObservations.length > 5 && (
                                <p>
                                  {benchmarkObservations.length - 5} older
                                  benchmark observations omitted.
                                </p>
                              )}
                            </div>
                          )}
                          {previousObservations.length > 0 && (
                            <div className="benchmark-diagnostic-notes">
                              <h4>Previous observations</h4>
                              <ul>
                                {previousObservations
                                  .slice(0, 5)
                                  .map((observation, index) => (
                                    <li
                                      key={`${observation.finished_at ?? "unknown"}-${index}`}
                                    >
                                      {statusLabel(
                                        observation.status ?? "unknown",
                                      )}{" "}
                                      · engine{" "}
                                      {observation.engine_version ?? "Unknown"}{" "}
                                      · finished{" "}
                                      {dateTime(observation.finished_at)}
                                      {observation.artifact
                                        ? ` · ${observation.artifact}`
                                        : ""}
                                      {observation.stored_report_id && (
                                        <button
                                          type="button"
                                          className="history-open-btn"
                                          disabled={inspect.isPending}
                                          onClick={() =>
                                            openSavedReport(
                                              observation.stored_report_id!,
                                            )
                                          }
                                        >
                                          {openingId ===
                                          observation.stored_report_id
                                            ? "Opening report…"
                                            : "Open previous saved report"}
                                        </button>
                                      )}
                                    </li>
                                  ))}
                              </ul>
                              {previousObservations.length > 5 && (
                                <p>
                                  {previousObservations.length - 5} older
                                  observations omitted.
                                </p>
                              )}
                            </div>
                          )}
                          {item.warnings && item.warnings.length > 0 && (
                            <div>
                              <h4>Warnings</h4>
                              <ul>
                                {item.warnings.map((warning, index) => (
                                  <li key={`${index}-${warning}`}>{warning}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {(item.warning_count ?? 0) > 0 &&
                            !item.warnings?.length && (
                              <p>
                                Run a pinned inspection to view full diagnostic
                                details.
                              </p>
                            )}
                          {item.diagnostic_notes &&
                            item.diagnostic_notes.length > 0 && (
                              <div className="benchmark-diagnostic-notes">
                                <h4>Diagnostic summary</h4>
                                <ul>
                                  {item.diagnostic_notes.map((note, index) => (
                                    <li key={`${index}-${note}`}>{note}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                        </details>
                      </article>
                    );
                  })}
                </div>
              )}

              {pageCount > 1 && (
                <nav
                  className="catalog-pagination"
                  aria-label="Catalog result pages"
                >
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
              {catalog.data.benchmark_publication_id && (
                <p className="benchmark-disclaimer">
                  Benchmark publication {catalog.data.benchmark_publication_id}.
                  Results describe static metadata and reconstruction attempts;
                  they do not measure inference speed or task accuracy.
                </p>
              )}
            </>
          )}
        </section>
      </main>
      <PageFooter />
    </div>
  );
}
