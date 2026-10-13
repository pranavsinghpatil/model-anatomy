import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownRight,
  ArrowRight,
  CircleHelp,
  FileSearch,
  GitBranch,
  History,
  Network,
  ShieldCheck,
} from "lucide-react";
import { api } from "../api/client";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Button, ModeControl, ThemeControl } from "../components/ui";
import { usePreferences } from "../state/preferences";
import { useRecent, useService } from "../state/queries";
import "./entry.css";

const examples = [
  "HuggingFaceTB/SmolLM2-135M",
  "google-bert/bert-base-uncased",
  "distilbert/distilbert-base-uncased",
];

function Brand() {
  return (
    <Link className="entry-brand" to="/" aria-label="Model Anatomy home">
      <span className="entry-brand-mark" aria-hidden="true">
        <Network size={19} strokeWidth={1.8} />
      </span>
      <span className="entry-brand-name">Model Anatomy</span>
    </Link>
  );
}

function FooterBrand() {
  return (
    <span className="entry-footer-brand">
      <span className="entry-brand-mark" aria-hidden="true">
        <Network size={14} />
      </span>
      <span className="entry-brand-name">Model Anatomy</span>
    </span>
  );
}

function EntryHeader() {
  return (
    <header className="entry-header">
      <Brand />
      <nav className="entry-header-actions" aria-label="Page controls">
        <a className="entry-history-link" href="#recent">
          <History size={15} />
          <span>Your analyses</span>
        </a>
        <ThemeControl />
      </nav>
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
  const { mode } = usePreferences();
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
      // A late response can update visitor history, but must not override a
      // route change made while the request was running.
      if (
        !active.current ||
        window.location.pathname !== submissionPath.current
      )
        return;
      const view = job.status === "inspected" ? "artifacts" : "architecture";
      void navigate(`/app/analyses/${encodeURIComponent(job.id)}?view=${view}`);
    },
  });

  return (
    <div className="entry-page">
      <EntryHeader />
      <main id="main" tabIndex={-1} className="entry-main">
        <section className="entry-launch" aria-labelledby="entry-title">
          <div className="entry-primary">
            <div className="entry-heading">
              <span className="entry-kicker">
                <span className="entry-kicker-mark" aria-hidden="true" />
                MODEL INSPECTION WORKSPACE
              </span>
              <h1 id="entry-title">Open a model. Understand its structure.</h1>
              <p className="entry-lede">
                Submit a Hugging Face model to inspect its architecture,
                components, tensors, and supporting evidence.
              </p>
            </div>

            <section className="entry-submit" aria-labelledby="submit-title">
              <div className="entry-submit-heading">
                <div>
                  <h2 id="submit-title">Inspect a model</h2>
                </div>
                <ModeControl />
              </div>
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
                      ? "Inspection service ready"
                      : "Inspection service unavailable"}
                </span>
              </div>
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  if (model.trim() && available && !create.isPending)
                    create.mutate(model.trim());
                }}
              >
                <label className="entry-field-label" htmlFor="model-input">
                  Hugging Face model
                </label>
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
                  Public repositories and pinned checkpoint file URLs are
                  accepted.
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
                      type="button"
                      key={example}
                      onClick={() => setModel(example)}
                      aria-label={`Fill model field with ${example}`}
                    >
                      {example}
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
                    <strong>New analyses are unavailable</strong>
                    <p>
                      {service?.reason ??
                        health.error?.message ??
                        "The inspection service is not ready."}{" "}
                      Existing analyses remain accessible below.
                    </p>
                  </div>
                  <Button type="button" onClick={() => void health.refetch()}>
                    Check again
                  </Button>
                </div>
              )}
              {create.error && <ErrorNotice error={create.error} />}

              <div className="entry-mode-note">
                <span className="entry-mode-icon" aria-hidden="true">
                  {mode === "learn" ? (
                    <CircleHelp size={17} />
                  ) : (
                    <GitBranch size={17} />
                  )}
                </span>
                <p>
                  <strong>{mode === "learn" ? "Learn" : "Research"}</strong>
                  {mode === "learn"
                    ? " adds a reading guide beside the architecture so you can explore each idea in context."
                    : " opens the technical architecture workspace with components and evidence ready to inspect."}
                </p>
              </div>
            </section>
          </div>

          <aside className="entry-next" aria-labelledby="next-title">
            <div className="entry-next-topline">
              <span>THE INSPECTION PATH</span>
              <ArrowDownRight size={16} aria-hidden="true" />
            </div>
            <h2 id="next-title">What happens next</h2>
            <ol>
              <li>
                <span className="entry-next-number">01</span>
                <div>
                  <strong>Pin the repository revision</strong>
                  <p>Every result points to a specific model snapshot.</p>
                </div>
              </li>
              <li>
                <span className="entry-next-number">02</span>
                <div>
                  <strong>Read model artifacts</strong>
                  <p>
                    Configuration and checkpoint metadata inform inspection.
                  </p>
                </div>
              </li>
              <li>
                <span className="entry-next-number">03</span>
                <div>
                  <strong>Explore the reported structure</strong>
                  <p>Open components and trace facts to their evidence.</p>
                </div>
              </li>
            </ol>
            <div className="entry-next-foot">
              <ShieldCheck size={16} />
              <span>Unknown or unsupported details stay explicit.</span>
            </div>
          </aside>
        </section>

        <details className="entry-boundaries">
          <summary>
            <FileSearch size={16} />
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
              details separately; support varies by model family and artifact
              layout.
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
              <span className="entry-step-label">YOUR WORKSPACE</span>
              <h2 id="recent-title">Recent analyses</h2>
            </div>
            <span className="entry-recent-count">
              {recent.data
                ? `${jobs.length} ${jobs.length === 1 ? "analysis" : "analyses"}`
                : "Visitor-owned history"}
            </span>
          </div>
          <p className="entry-recent-copy">
            Analyses submitted from this visitor session. Select one to continue
            where you left off.
          </p>
          {recent.isPending && <Loading text="Loading your analyses" />}
          {recent.error && (
            <ErrorNotice
              error={recent.error}
              retry={() => void recent.refetch()}
            />
          )}
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
                          to={`/app/analyses/${encodeURIComponent(job.id)}?view=${job.status === "inspected" ? "artifacts" : "architecture"}`}
                        >
                          <span className="entry-model-icon" aria-hidden="true">
                            <Network size={16} />
                          </span>
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
                          to={`/app/analyses/${encodeURIComponent(job.id)}?view=${job.status === "inspected" ? "artifacts" : "architecture"}`}
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
          {recent.data && jobs.length === 0 && (
            <div className="entry-empty">
              <span className="entry-empty-icon" aria-hidden="true">
                <History size={19} />
              </span>
              <div>
                <strong>No analyses in this session yet</strong>
                <p>Submit a model above and its report will appear here.</p>
              </div>
            </div>
          )}
        </section>
      </main>
      <footer className="entry-footer">
        <FooterBrand />
        <span>Artifact-backed inspection with explicit evidence.</span>
        <a href="#main">Back to top</a>
      </footer>
    </div>
  );
}
