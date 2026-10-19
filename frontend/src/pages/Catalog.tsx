import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Search,
  Copy,
  Check,
  ExternalLink,
  SlidersHorizontal,
} from "lucide-react";
import { ApiError, api, retryQuery } from "../api/client";
import type { CatalogItem } from "../api/contract";
import { DitherBackground } from "../components/DitherBackground";
import { EntryHeader, PageFooter } from "./Entry";
import "./entry.css";

const pageSize = 24;
function displayed(value: number | null | undefined) {
  return value == null ? "Unknown" : value.toLocaleString();
}
function statusLabel(status: CatalogItem["status"]) {
  return status === "completed"
    ? "Architecture reconstructed"
    : status === "partial"
      ? "Partial architecture"
      : "Artifacts inspected";
}

export default function Catalog() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const timer = window.setTimeout(
      () => setDebouncedQuery(searchQuery.trim()),
      300,
    );
    return () => window.clearTimeout(timer);
  }, [searchQuery]);
  useEffect(() => setOffset(0), [debouncedQuery]);

  const catalog = useQuery({
    queryKey: ["catalog", debouncedQuery, offset],
    queryFn: ({ signal }) =>
      api.catalog(debouncedQuery, offset, pageSize, signal),
    retry: retryQuery,
    refetchOnWindowFocus: true,
  });
  const inspect = useMutation({
    mutationFn: (id: string) => api.inspectCatalog(id),
    onSuccess: async (analysis) => {
      await queryClient.invalidateQueries({ queryKey: ["catalog"] });
      navigate(`/app/analyses/${encodeURIComponent(analysis.id)}/architecture`);
    },
  });

  const copyId = async (id: string) => {
    await navigator.clipboard.writeText(id);
    setCopiedId(id);
    window.setTimeout(() => setCopiedId(null), 2000);
  };
  const response = catalog.data ?? {
    models: [],
    total: 0,
    offset,
    limit: pageSize,
  };
  const models = response?.models ?? [];
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
              <span>Stored Model Inspections</span>
            </div>
            <h1 className="catalog-page-title">
              Model <span className="hero-highlight-amber">Catalog</span>
            </h1>
            <p className="catalog-page-subtitle">
              Models appear here after a real inspection is saved. Listed
              dimensions and status come from stored inspection records.
            </p>
          </div>
        </header>

        <section
          className="catalog-toolbar-section"
          aria-label="Catalog search"
        >
          <div className="catalog-toolbar-inner">
            <div className="catalog-search-box">
              <Search size={16} className="search-icon" aria-hidden="true" />
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search stored model IDs..."
                className="catalog-search-input"
                aria-label="Search stored model catalog"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="search-clear-btn"
                  aria-label="Clear search query"
                >
                  ×
                </button>
              )}
            </div>
          </div>
        </section>

        <section
          className="catalog-models-section"
          aria-labelledby="catalog-list-heading"
        >
          <div className="catalog-models-inner">
            <h2 id="catalog-list-heading" className="sr-only">
              Stored model inspections
            </h2>
            {catalog.isPending ? (
              <div className="catalog-empty-state" role="status">
                <p>Loading stored inspections…</p>
              </div>
            ) : null}
            {catalog.isError ? (
              <div className="catalog-empty-state" role="alert">
                <h3>Could not load the model catalog</h3>
                <p>{errorText}</p>
                <button
                  type="button"
                  className="catalog-empty-btn"
                  onClick={() => void catalog.refetch()}
                >
                  Retry
                </button>
              </div>
            ) : null}
            {catalog.isSuccess && models.length === 0 ? (
              <div className="catalog-empty-state">
                <SlidersHorizontal
                  size={32}
                  className="empty-icon"
                  aria-hidden="true"
                />
                <h3>
                  {response.total === 0
                    ? "No inspections are stored yet"
                    : `No stored models match “${debouncedQuery}”`}
                </h3>
                <p>
                  {response.total === 0
                    ? "Inspect a Hugging Face model to save its real artifact facts here."
                    : "Try another model ID, or inspect a model from the home page."}
                </p>
                <Link to="/#submit-title" className="catalog-empty-btn">
                  <span>Inspect a model</span>
                  <ArrowRight size={14} />
                </Link>
              </div>
            ) : null}
            {models.length > 0 ? (
              <>
                <p className="catalog-page-subtitle" aria-live="polite">
                  Showing {response.offset + 1}–
                  {response.offset + models.length} of {response.total} stored
                  inspections
                </p>
                <div className="catalog-cards-grid">
                  {models.map((item) => (
                    <article key={item.id} className="catalog-detail-card">
                      <div className="card-top-row">
                        <span className="card-family-tag">
                          {statusLabel(item.status)}
                        </span>
                        <span className="card-params-tag">
                          {item.parameter_count == null
                            ? "Parameter count unknown"
                            : `${displayed(item.parameter_count)} parameters`}
                        </span>
                      </div>
                      <div className="card-heading-group">
                        <h3 className="card-model-title">{item.model_id}</h3>
                        <div className="card-id-row">
                          <code className="card-repo-id">
                            Revision {item.revision}
                          </code>
                          <button
                            type="button"
                            className="copy-id-btn"
                            onClick={() => void copyId(item.model_id)}
                            title="Copy repository ID"
                            aria-label={`Copy repository ID for ${item.model_id}`}
                          >
                            {copiedId === item.model_id ? (
                              <Check size={13} className="copied-check" />
                            ) : (
                              <Copy size={13} />
                            )}
                          </button>
                        </div>
                      </div>
                      <div className="card-specs-matrix">
                        <div className="spec-metric">
                          <span className="metric-label">Layers</span>
                          <strong className="metric-val">
                            {displayed(item.dimensions.layers)}
                          </strong>
                        </div>
                        <div className="spec-metric">
                          <span className="metric-label">Heads</span>
                          <strong className="metric-val">
                            {Array.isArray(item.dimensions.heads)
                              ? item.dimensions.heads.map(displayed).join(" / ")
                              : displayed(item.dimensions.heads)}
                          </strong>
                        </div>
                        <div className="spec-metric">
                          <span className="metric-label">Hidden</span>
                          <strong className="metric-val">
                            {displayed(item.dimensions.hidden)}
                          </strong>
                        </div>
                        <div className="spec-metric">
                          <span className="metric-label">Vocabulary</span>
                          <strong className="metric-val">
                            {displayed(item.dimensions.vocabulary)}
                          </strong>
                        </div>
                      </div>
                      <div className="card-technical-details">
                        <div className="tech-row">
                          <span className="tech-label">Architecture:</span>
                          <span className="tech-value">
                            {item.architecture ?? "Unknown"}
                          </span>
                        </div>
                        <div className="tech-row">
                          <span className="tech-label">Inspection:</span>
                          <span className="tech-value">
                            {new Date(item.inspected_at).toLocaleString()}
                          </span>
                        </div>
                        <div className="tech-row">
                          <span className="tech-label">Artifact:</span>
                          <code className="tech-class">
                            {item.artifact}
                            {item.explicit_artifact ? " · selected" : ""}
                          </code>
                        </div>
                      </div>
                      <div className="card-footer-actions">
                        <button
                          type="button"
                          className="card-inspect-btn"
                          disabled={inspect.isPending}
                          onClick={() => inspect.mutate(item.id)}
                          aria-label={`Open stored inspection for ${item.model_id}`}
                        >
                          <span>
                            {inspect.isPending
                              ? "Opening inspection…"
                              : "Open Inspection"}
                          </span>
                          <ArrowRight size={14} />
                        </button>
                        <a
                          href={`https://huggingface.co/${encodeURIComponent(item.model_id).replaceAll("%2F", "/")}`}
                          target="_blank"
                          rel="noreferrer"
                          className="card-hf-btn"
                          title="View repository on Hugging Face"
                          aria-label={`Open ${item.model_id} on Hugging Face`}
                        >
                          <ExternalLink size={14} />
                        </a>
                      </div>
                    </article>
                  ))}
                </div>
                <div className="catalog-pagination" aria-label="Catalog pages">
                  <button
                    type="button"
                    className="catalog-empty-btn"
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - pageSize))}
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    className="catalog-empty-btn"
                    disabled={offset + models.length >= response.total}
                    onClick={() => setOffset(offset + pageSize)}
                  >
                    Next
                  </button>
                </div>
              </>
            ) : null}
            {inspect.isError && (
              <div role="alert" className="catalog-empty-state">
                <h3>Could not open this stored inspection</h3>
                <p>{inspect.error.message}</p>
                <button
                  type="button"
                  className="catalog-empty-btn"
                  onClick={() => inspect.reset()}
                >
                  Dismiss
                </button>
              </div>
            )}
          </div>
        </section>
      </main>
      <PageFooter />
    </div>
  );
}
