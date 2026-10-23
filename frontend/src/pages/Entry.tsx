import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  ArrowUp,
  History,
  Network,
  ShieldCheck,
  Database,
} from "lucide-react";
import { analysisHref, api } from "../api/client";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Button, ThemeControl } from "../components/ui";
import { ModelAnatomyMark } from "../components/Logo";
import { useRecent, useService } from "../state/queries";
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
        <div className="footer-top-grid">
          {/* Column 1: Brand & Architecture Mission */}
          <div className="footer-brand-col">
            <Link
              to="/"
              className="footer-brand"
              aria-label="Model Anatomy home"
            >
              <div className="brand-gem footer-gem" aria-hidden="true">
                <ModelAnatomyMark size={15} />
              </div>
              <span className="entry-brand-name">Model Anatomy</span>
              <span className="brand-tag">STUDIO</span>
            </Link>
            <p className="footer-mission">
              Reconstructing trustworthy, semantic, tensor-aware architecture
              graphs from open-weight Hugging Face models without downloading
              weights.
            </p>
            <div className="footer-tech-badges">
              <span className="tech-badge">PyTorch AST</span>
              <span className="tech-badge">SafeTensors Stream</span>
              <span className="tech-badge">Deterministic IR</span>
            </div>
          </div>

          {/* Column 2: Architecture Views */}
          <div className="footer-col">
            <h4 className="footer-col-title">Inspection</h4>
            <ul className="footer-nav-list">
              <li>
                <a href="/#blueprint">Architecture Graph</a>
              </li>
              <li>
                <a href="/#blueprint">SafeTensors Headers</a>
              </li>
              <li>
                <a href="/#blueprint">Parameter &amp; Weight BOM</a>
              </li>
              <li>
                <a href="/#blueprint">Module AST Hierarchy</a>
              </li>
            </ul>
          </div>

          {/* Column 3: Model Registry */}
          <div className="footer-col">
            <h4 className="footer-col-title">Model Registry</h4>
            <ul className="footer-nav-list">
              <li>
                <Link to="/catalog">Model Catalog</Link>
              </li>
              <li>
                <Link to="/catalog">Architecture inspections</Link>
              </li>
              <li>
                <Link to="/catalog">Artifact inspections</Link>
              </li>
              <li>
                <Link to="/catalog">Stored revisions</Link>
              </li>
            </ul>
          </div>

          {/* Column 4: Evidence & Provenance */}
          <div className="footer-col">
            <h4 className="footer-col-title">Provenance</h4>
            <ul className="footer-nav-list">
              <li>
                <Link to="/catalog">Stored models</Link>
              </li>
              <li>
                <a href="/#main">Support Boundaries</a>
              </li>
              <li>
                <span className="footer-static-note">
                  Zero Weight Downloads
                </span>
              </li>
              <li>
                <span className="footer-static-note">
                  Pinned Revision Audit
                </span>
              </li>
            </ul>
          </div>
        </div>

        {/* Footer Bottom Bar */}
        <div className="footer-bottom-bar">
          <div className="footer-copyright">
            <span>&copy; {new Date().getFullYear()} Model Anatomy Studio.</span>
            <span className="footer-sep">&middot;</span>
            <span className="footer-evidence-note">
              Artifact-backed inspection with explicit evidence.
            </span>
          </div>
          <div className="footer-actions">
            <a
              href="#main"
              className="footer-back-to-top"
              onClick={(e) => {
                e.preventDefault();
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              aria-label="Back to top of page"
            >
              <span>Back to top</span>
              <ArrowUp size={13} aria-hidden="true" />
            </a>
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
            <a
              href={isCatalog ? "/#blueprint" : "#blueprint"}
              className="entry-nav-link"
            >
              <span>Blueprint</span>
            </a>
            <a
              href={isCatalog ? "/#metrics" : "#metrics"}
              className="entry-nav-link"
            >
              <span>Metrics</span>
            </a>
            <a
              href={isCatalog ? "/#capabilities" : "#capabilities"}
              className="entry-nav-link"
            >
              <span>Capabilities</span>
            </a>
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
            <span>Your analyses</span>
          </Link>
          <a
            className="entry-header-cta"
            href={isCatalog ? "/#submit-title" : "#submit-title"}
          >
            <span>Start inspection</span>
            <ArrowRight size={13} />
          </a>
          <ThemeControl />
        </nav>
      </div>
    </header>
  );
}

export function Start() {
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
  const navigate = useNavigate();
  const client = useQueryClient();
  const { health, service, available } = useService();
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
                Schematic layer flow, SafeTensors headers, parameter
                allocations, and deterministic evidence — reconstructed live
                without downloading model weights.
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
                className={`entry-live-service ${health.isPending ? "is-checking" : available ? "is-ready" : "is-unavailable"}`}
                role="status"
                aria-live="polite"
              >
                <span className="entry-live-dot" aria-hidden="true" />
                <span>
                  {health.isPending
                    ? "Checking service availability"
                    : available
                      ? "Stored inspection service available"
                      : "Stored inspection service unavailable"}
                </span>
              </div>
            </div>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (model.trim() && available && !create.isPending)
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
                  onChange={(event) => setModel(event.target.value)}
                />
                <Button
                  type="submit"
                  variant="primary"
                  disabled={!available || !model.trim() || create.isPending}
                  aria-busy={create.isPending}
                >
                  {create.isPending ? "Starting…" : "Inspect model"}
                  <ArrowRight size={16} />
                </Button>
              </div>
              <p className="entry-field-help">
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
                    onClick={() => setModel(example.id)}
                    aria-label={`Fill model field with ${example.id}`}
                  >
                    <span className="chip-name">{example.label}</span>
                    <ArrowRight size={13} aria-hidden="true" />
                  </button>
                ))}
              </div>
            </div>

            {!available && !health.isPending && (
              <div
                className="entry-service entry-service-unavailable"
                role="status"
              >
                <div>
                  <strong>Stored inspections are unavailable</strong>
                  <p>
                    {service?.reason ??
                      health.error?.message ??
                      "The catalog database or artifact storage is not available."}{" "}
                    New submissions can still match a previously stored
                    inspection.
                  </p>
                </div>
                <Button type="button" onClick={() => void health.refetch()}>
                  Check again
                </Button>
              </div>
            )}
            {create.error && <ErrorNotice error={create.error} />}
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
          <section
            id="catalog"
            className="entry-catalog-banner"
            aria-labelledby="catalog-heading"
          >
            <div className="catalog-banner-glow" aria-hidden="true" />
            <div className="catalog-banner-content">
              <div className="catalog-banner-info">
                <div className="catalog-pill-badge">
                  <span className="entry-badge-dot" />
                  <span>Architecture Registry</span>
                  <span className="catalog-count-tag">Saved inspections</span>
                </div>
                <h2 id="catalog-heading" className="catalog-banner-title">
                  Explore stored model inspections
                </h2>
                <p className="catalog-banner-desc">
                  Browse real inspection records and open their saved
                  architecture or artifact reports.
                </p>
                <div className="catalog-banner-tags">
                  <span className="cat-chip">Saved model IDs</span>
                  <span className="cat-chip">Stored dimensions</span>
                  <span className="cat-chip">Inspection status</span>
                </div>
              </div>
              <div className="catalog-banner-actions">
                <Link to="/catalog" className="catalog-cta-btn">
                  <span>Browse Full Catalog</span>
                  <ArrowRight size={15} />
                </Link>
                <span className="catalog-cta-sub">Browse database records</span>
              </div>
            </div>
          </section>
        </section>

        <details className="entry-boundaries">
          <summary>
            <span>What can this inspect?</span>
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

        <section
          className="entry-recent"
          id="recent"
          aria-labelledby="recent-title"
        >
          <div className="entry-section-heading">
            <div>
              <h2 id="recent-title">Recent analyses</h2>
            </div>
            <span className="entry-recent-count">
              {`${jobs.length} ${jobs.length === 1 ? "analysis" : "analyses"}`}
            </span>
          </div>
          <p className="entry-recent-copy">
            {jobs.length > 0
              ? "Your submitted analyses. Select one to continue."
              : "No analyses in this session yet. Submit a public model above. Results stay on this browser."}
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
                              {job.architecture ?? "Architecture pending"}
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
                          aria-label={`Continue analysis of ${job.model_id}`}
                        >
                          Continue <ArrowRight size={14} />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
      <PageFooter />
    </div>
  );
}
