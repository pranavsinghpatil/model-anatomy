import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, History } from "lucide-react";
import { api } from "../api/client";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Button, ModeControl, ThemeControl } from "../components/ui";
import { usePreferences } from "../state/preferences";
import { useRecent, useService } from "../state/queries";
import { DotPattern } from "../components/vendor/dot-pattern";
import "./entry.css";

const examples = [
  { id: "HuggingFaceTB/SmolLM2-135M", label: "SmolLM2 · 135M" },
  { id: "google-bert/bert-base-uncased", label: "BERT base" },
  { id: "distilbert/distilbert-base-uncased", label: "DistilBERT" },
];

function Brand() {
  return (
    <Link className="entry-brand" to="/" aria-label="Model Anatomy home">
      <span className="entry-brand-name">Model Anatomy</span>
    </Link>
  );
}

function FooterBrand() {
  return (
    <span className="entry-footer-brand">
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
      <div className="entry-backdrop" aria-hidden="true">
        <DotPattern width={22} height={22} cr={0.8} />
      </div>
      <EntryHeader />
      <main id="main" tabIndex={-1} className="entry-main">
        <section className="entry-launch" aria-labelledby="entry-title">
          <div className="entry-primary">
            <div className="entry-heading">
              <h1 id="entry-title">See how a model is built.</h1>
              <p className="entry-lede">
                Inspect architecture, tensors and evidence from an open-weight
                Hugging Face model.
              </p>
            </div>
          </div>

          <section className="entry-submit" aria-labelledby="submit-title">
            <div className="entry-submit-heading">
              <div>
                <h2 id="submit-title">Inspect a model</h2>
              </div>
              <ModeControl />
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
                      ? "Inspection service ready"
                      : "Inspection service unavailable"}
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
                Public repositories and pinned checkpoint file URLs. Model
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
                    {example.label}
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
              <p>
                <strong>{mode === "learn" ? "Learn" : "Research"}</strong>
                {mode === "learn"
                  ? " adds reading guidance beside the architecture."
                  : " opens the architecture and evidence workspace."}
              </p>
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
              {recent.data
                ? `${jobs.length} ${jobs.length === 1 ? "analysis" : "analyses"}`
                : "Visitor-owned history"}
            </span>
          </div>
          <p className="entry-recent-copy">
            Your submitted analyses. Select one to continue.
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
