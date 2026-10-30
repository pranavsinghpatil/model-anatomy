import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  History,
  Network,
  ShieldCheck,
  Database,
} from "lucide-react";
import { analysisHref, api, ApiError } from "../api/client";
import { ErrorNotice, Status } from "../components/Common";
import { Button, ThemeControl } from "../components/ui";
import { ModelAnatomyMark } from "../components/Logo";
import { useRecent, useService } from "../state/queries";
import { isModelValidationError } from "../state/feedback-messages";
import { DitherBackground } from "../components/DitherBackground";
import "./entry.css";

const examples = [
  { id: "HuggingFaceTB/SmolLM2-135M", label: "SmolLM2" },
  { id: "google-bert/bert-base-uncased", label: "BERT base" },
  { id: "distilbert/distilbert-base-uncased", label: "DistilBERT" },
];

function Brand() {
  return (
    <Link className="entry-brand" to="/" aria-label="Model Anatomy home">
      <div className="brand-gem" aria-hidden="true">
        <ModelAnatomyMark size={18} />
      </div>
      <span className="entry-brand-name">Model Anatomy</span>
      <span className="brand-tag">STUDIO</span>
    </Link>
  );
}

export function PageFooter() {
  return (
    <footer className="entry-footer" role="contentinfo">
      <div className="entry-footer-inner">
        <div className="footer-bottom-bar">
          <div className="footer-copyright">
            <Link
              to="/"
              className="footer-brand"
              aria-label="Model Anatomy home"
            >
              <span className="entry-brand-name">Model Anatomy</span>
            </Link>
            <span className="footer-evidence-note">
              Inspection facts cite available model evidence.
            </span>
          </div>
          <div className="footer-actions">
            <Link to="/catalog">Catalog</Link>
            <a href="/#support">Support boundaries</a>
          </div>
        </div>
      </div>
    </footer>
  );
}

export function EntryHeader() {
  const location = useLocation();
  const isCatalog = location.pathname.startsWith("/catalog");

  return (
    <header className="entry-header">
      <div className="entry-header-inner">
        <div className="entry-header-left">
          <Brand />
          <span className="entry-header-divider" aria-hidden="true" />
          <nav className="entry-nav-links" aria-label="Sections">
            <Link
              to="/catalog"
              className={`entry-nav-link ${isCatalog ? "is-active" : ""}`}
              aria-current={isCatalog ? "page" : undefined}
            >
              <span>Catalog</span>
            </Link>
          </nav>
        </div>
        <nav className="entry-header-actions" aria-label="Page controls">
          <Link className="entry-history-link" to="/#recent">
            <History size={14} />
            <span>Your inspections</span>
          </Link>
          <a
            className="entry-header-cta"
            href={isCatalog ? "/#submit-title" : "#submit-title"}
          >
            <span>New inspection</span>
            <ArrowRight size={13} />
          </a>
          <ThemeControl />
        </nav>
      </div>
    </header>
  );
}

export function Start() {
  const location = useLocation();
  const active = useRef(true);
  const submissionPath = useRef(window.location.pathname);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);

  const [model, setModel] = useState(
    () => new URLSearchParams(window.location.search).get("model") ?? "",
  );
  useEffect(() => {
    if (!["#support", "#recent", "#submit-title"].includes(location.hash))
      return;
    const target = document.getElementById(location.hash.slice(1));
    if (!target) return;
    if (target instanceof HTMLDetailsElement) target.open = true;
    const focusTarget = target.matches("h2")
      ? target
      : target.querySelector<HTMLElement>("summary, h2");
    const frame = window.requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: "instant", block: "start" });
      if (focusTarget) {
        if (focusTarget.tagName !== "SUMMARY") focusTarget.tabIndex = -1;
        focusTarget.focus({ preventScroll: true });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [location.hash]);
  const navigate = useNavigate();
  const client = useQueryClient();
  const { health, service, capability } = useService();
  const recent = useRecent();
  const jobs = [...(recent.data?.analyses ?? [])].sort(
    (a, b) => Date.parse(b.created_at) - Date.parse(a.created_at),
  );
  const create = useMutation({
    mutationFn: api.analyze,
    onSuccess: (job) => {
      client.setQueryData(["analysis", job.id], job);
      void client.invalidateQueries({ queryKey: ["analyses"] });
      void client.invalidateQueries({ queryKey: ["catalog"] });
      if (
        !active.current ||
        window.location.pathname !== submissionPath.current
      )
        return;
      void navigate(analysisHref(job));
    },
  });

  return (
    <div className="entry-page">
      <DitherBackground />
      <EntryHeader />
      <main id="main" tabIndex={-1} className="entry-main">
        <section className="entry-launch" aria-labelledby="entry-title">
          <div className="entry-primary">
            <div className="entry-heading">
              <div className="entry-announcement-badge">
                <span className="entry-badge-dot" />
                <span>Tensor &amp; Architecture Inspection Studio</span>
                <span className="badge-ver">preview</span>
              </div>
              <h1 id="entry-title">
                See how a <br />
                <span className="hero-highlight-amber">model is built.</span>
              </h1>
              <p className="entry-lede">
                Explore model structure, checkpoint metadata and source evidence
                from Hugging Face.
              </p>
            </div>
          </div>

          <section className="entry-submit" aria-labelledby="submit-title">
            <div className="border-beam" aria-hidden="true" />

            <div className="entry-submit-heading">
              <h2 id="submit-title">Inspect a model</h2>
            </div>

            <div className="entry-field-heading">
              <label className="entry-field-label" htmlFor="model-input">
                Hugging Face model
              </label>
              <div
                className={`entry-live-service ${capability === "checking" ? "is-checking" : capability === "ready" ? "is-ready" : "is-unavailable"}`}
                role="status"
                aria-live="polite"
              >
                <span className="entry-live-dot" aria-hidden="true" />
                <span>
                  {capability === "checking"
                    ? "Checking inspection service"
                    : capability === "ready"
                      ? "New inspections available"
                      : capability === "stored-only"
                        ? "Saved inspections available; new runs unavailable"
                        : "Inspection storage unavailable"}
                </span>
              </div>
            </div>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (
                  model.trim() &&
                  capability !== "checking" &&
                  capability !== "unavailable" &&
                  !create.isPending
                )
                  create.mutate(model.trim());
              }}
            >
              <div className="entry-input-row">
                <input
                  id="model-input"
                  name="model"
                  required
                  maxLength={512}
                  placeholder="namespace/model or https://huggingface.co/…"
                  autoComplete="off"
                  spellCheck={false}
                  value={model}
                  aria-invalid={Boolean(
                    create.error && isModelValidationError(create.error),
                  )}
                  aria-describedby={
                    create.error && isModelValidationError(create.error)
                      ? "model-input-error"
                      : "model-input-help"
                  }
                  onChange={(event) => {
                    setModel(event.target.value);
                    if (create.error) create.reset();
                  }}
                />
                <Button
                  type="submit"
                  variant="primary"
                  disabled={
                    capability === "checking" ||
                    capability === "unavailable" ||
                    !model.trim() ||
                    create.isPending
                  }
                  aria-busy={create.isPending}
                >
                  {create.isPending
                    ? "Starting…"
                    : capability === "stored-only"
                      ? "Find saved inspection"
                      : "Inspect model"}
                  <ArrowRight size={16} />
                </Button>
              </div>
              <p className="entry-field-help" id="model-input-help">
                Public model repositories and pinned checkpoint file URLs. Model
                weights are not downloaded.
              </p>
            </form>

            <div
              className="entry-examples"
              role="group"
              aria-labelledby="entry-examples-title"
            >
              <span id="entry-examples-title">Try an example</span>
              <div>
                {examples.map((example) => (
                  <button
                    className="entry-example-chip"
                    aria-pressed={model === example.id}
                    type="button"
                    key={example.id}
                    title={example.id}
                    onClick={() => {
                      setModel(example.id);
                      if (create.error) create.reset();
                    }}
                    aria-label={`Fill model field with ${example.id}`}
                  >
                    <span className="chip-name">{example.label}</span>
                    <ArrowRight size={13} aria-hidden="true" />
                  </button>
                ))}
              </div>
            </div>

            {capability === "stored-only" && (
              <div
                className="entry-service entry-service-unavailable"
                role="status"
              >
                <div>
                  <strong>Saved reports remain available</strong>
                  <p>
                    New inspections are paused while the worker is offline. A
                    matching saved inspection can still open.
                  </p>
                </div>
                <Button type="button" onClick={() => void health.refetch()}>
                  Check again
                </Button>
              </div>
            )}
            {capability === "unavailable" && !health.isPending && (
              <div
                className="entry-service entry-service-unavailable"
                role="status"
              >
                <div>
                  <strong>Inspection service unavailable</strong>
                  <p>
                    {health.error
                      ? "We cannot reach the inspection service. Check your connection and try again."
                      : "The report database or storage is temporarily unavailable. Try checking again."}
                  </p>
                  {service?.reason && (
                    <details>
                      <summary>Technical details</summary>
                      <code>{service.reason}</code>
                    </details>
                  )}
                </div>
                <Button type="button" onClick={() => void health.refetch()}>
                  Check again
                </Button>
              </div>
            )}
            {create.error && (
              <ErrorNotice
                error={create.error}
                context="submit"
                id={
                  isModelValidationError(create.error)
                    ? "model-input-error"
                    : undefined
                }
                actions={
                  create.error instanceof ApiError &&
                  create.error.code === "worker_unavailable" ? (
                    <Link className="button" to="/catalog">
                      Browse saved inspections
                    </Link>
                  ) : (
                    <Link className="button" to="/#recent">
                      Your inspections
                    </Link>
                  )
                }
                retry={
                  !isModelValidationError(create.error) &&
                  !(
                    create.error instanceof ApiError &&
                    create.error.code === "network_error"
                  ) &&
                  capability !== "unavailable" &&
                  capability !== "checking"
                    ? () => create.mutate(model.trim())
                    : undefined
                }
              />
            )}
          </section>

          <section
            className="entry-recent"
            id="recent"
            aria-labelledby="recent-title"
          >
            <div className="entry-section-heading">
              <div>
                <h2 id="recent-title">Your inspections</h2>
              </div>
              <span className="entry-recent-count">
                {`${jobs.length} ${jobs.length === 1 ? "inspection" : "inspections"}`}
              </span>
            </div>
            <p className="entry-recent-copy">
              {jobs.length > 0
                ? "Open a recent inspection report from this browser. Saved reports can expire; each report shows its availability."
                : "No inspections yet. Enter a public model above to begin."}
            </p>
            {jobs.length > 0 && (
              <div className="entry-table-wrap">
                <table className="entry-table">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col">Revision</th>
                      <th scope="col">Submitted</th>
                      <th scope="col">Status</th>
                      <th scope="col">
                        <span className="sr-only">Action</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map((job) => (
                      <tr key={job.id}>
                        <td>
                          <Link
                            className="entry-model-link"
                            to={analysisHref(job)}
                          >
                            <span>
                              <strong>{job.model_id}</strong>
                              <small>
                                {job.architecture ?? "Inspection in progress"}
                                {job.expires_at &&
                                  ` · Available until ${new Date(job.expires_at).toLocaleDateString()}`}
                              </small>
                            </span>
                          </Link>
                        </td>
                        <td className="entry-revision">
                          {job.revision ? job.revision.slice(0, 12) : "Pending"}
                        </td>
                        <td className="entry-time">
                          <time dateTime={job.created_at}>
                            {new Date(job.created_at).toLocaleString([], {
                              month: "short",
                              day: "numeric",
                              hour: "numeric",
                              minute: "2-digit",
                            })}
                          </time>
                        </td>
                        <td>
                          <Status status={job.status} />
                        </td>
                        <td>
                          <Link
                            className="entry-continue"
                            to={analysisHref(job)}
                            aria-label={`Open inspection of ${job.model_id}`}
                          >
                            Open report <ArrowRight size={14} />
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* Optimized Architectural Telemetry Ribbon */}
          <div
            id="metrics"
            className="entry-stats-ribbon"
            role="region"
            aria-label="Key metrics"
          >
            <div className="stat-card">
              <span className="stat-number">Pinned</span>
              <span className="stat-label">Revision</span>
              <span className="stat-sub">Facts tied to a model revision</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">Artifact</span>
              <span className="stat-label">Metadata</span>
              <span className="stat-sub">
                Repository files and tensor headers
              </span>
            </div>
            <div className="stat-card">
              <span className="stat-number">Unknowns</span>
              <span className="stat-label">Stay explicit</span>
              <span className="stat-sub">Missing facts are not inferred</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">Saved</span>
              <span className="stat-label">Reports</span>
              <span className="stat-sub">Visitor-owned inspections</span>
            </div>
          </div>

          {/* Saved inspection catalog link */}
          <section
            id="blueprint"
            className="entry-interactive-teaser"
            aria-labelledby="blueprint-heading"
          >
            <div className="teaser-header">
              <div>
                <span className="teaser-eyebrow">
                  Stored inspection reports
                </span>
                <h2 id="blueprint-heading">
                  Explore architecture facts from{" "}
                  <span className="hero-highlight-amber">real inspections</span>
                  .
                </h2>
                <p className="entry-bento-desc">
                  The catalog lists models only after their inspection report
                  has been saved. Unknown dimensions remain unknown until
                  artifacts establish them.
                </p>
              </div>
              <Link to="/catalog" className="teaser-load-btn">
                <span>Browse stored inspections</span>
                <ArrowRight size={14} />
              </Link>
            </div>
          </section>
          {/* Streamlined Capabilities Highlights Grid */}
          <div
            id="capabilities"
            className="entry-bento-grid"
            role="region"
            aria-label="Capabilities"
          >
            <div className="entry-bento-card bento-feature-highlight">
              <div className="entry-bento-header">
                <span className="entry-bento-icon" aria-hidden="true">
                  <Network size={15} />
                </span>
                <span>Semantic Graph Reconstruction</span>
              </div>
              <p className="entry-bento-desc">
                Instantiated modules, attention topologies, and residual streams
                derived directly from genuine PyTorch code without LLM
                guesswork.
              </p>
              <div className="bento-badge-strip">
                <span className="bento-tag">Module AST</span>
                <span className="bento-tag">Attention Heads</span>
                <span className="bento-tag">Residual Routing</span>
              </div>
            </div>

            <div className="entry-bento-card">
              <div className="entry-bento-header">
                <span className="entry-bento-icon" aria-hidden="true">
                  <Database size={15} />
                </span>
                <span>SafeTensors Header Streaming</span>
              </div>
              <p className="entry-bento-desc">
                Repository file metadata and SafeTensors headers are shown when
                the selected model artifacts provide them.
              </p>
              <div className="bento-badge-strip">
                <span className="bento-tag">Repository Files</span>
                <span className="bento-tag">SafeTensors Metadata</span>
                <span className="bento-tag">Evidence Linked</span>
              </div>
            </div>

            <div className="entry-bento-card">
              <div className="entry-bento-header">
                <span className="entry-bento-icon" aria-hidden="true">
                  <ShieldCheck size={15} />
                </span>
                <span>Deterministic Evidence Audit</span>
              </div>
              <p className="entry-bento-desc">
                Facts include their available evidence and retain unknowns where
                reconstruction cannot establish a value.
              </p>
              <div className="bento-badge-strip">
                <span className="bento-tag">Explicit Provenance</span>
                <span className="bento-tag">Stored Report</span>
                <span className="bento-tag">Unknowns Preserved</span>
              </div>
            </div>
          </div>

          {/* Architecture Catalog Banner */}
        </section>

        <details className="entry-boundaries" id="support">
          <summary>
            <span>Support boundaries</span>
            <span className="entry-boundary-hint">
              Support and safety boundaries
            </span>
          </summary>
          <div className="entry-boundary-content">
            <p>
              Architecture reconstruction depends on the model’s native
              implementation and artifacts. Unsupported models may still return
              a verified artifact inventory, without an architecture graph.
            </p>
            <p>
              Inspection does not execute publisher code or download model
              weights. It reports known facts, derivations, and unresolved
              details separately. Results stay tied to a pinned revision and its
              source artifacts; support varies by model family and layout.
            </p>
          </div>
        </details>
      </main>
      <PageFooter />
    </div>
  );
}
