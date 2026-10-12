import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  Box,
  FileCheck2,
  FileSearch,
  GitBranch,
  History,
  Network,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import { api } from "../api/client";
import { time } from "../format";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Button, ModeControl, ThemeControl } from "../components/ui";
import { usePreferences } from "../state/preferences";
import { useRecent, useService } from "../state/queries";
export function Brand() {
  return (
    <Link className="brand" to="/" aria-label="Model Anatomy home">
      <span className="brand-icon">
        <Network size={21} />
      </span>
      <span>
        Model Anatomy
        <span className="brand-caption">OPEN WEIGHTS. OPEN QUESTIONS.</span>
      </span>
    </Link>
  );
}
function PublicHeader() {
  return (
    <header className="public-header">
      <Brand />
      <nav aria-label="Main navigation">
        <a className="desktop-link" href="/#method">
          How it works
        </a>
        <ThemeControl />
        <Link className="button button-secondary" to="/app">
          Open workspace <ArrowRight size={16} />
        </Link>
      </nav>
    </header>
  );
}
const capabilities = [
  {
    Icon: GitBranch,
    title: "See the structure",
    copy: "Navigate architecture from the model level to individual components, with repeated blocks kept readable.",
  },
  {
    Icon: ScanSearch,
    title: "Inspect the details",
    copy: "Read input and output shapes, parameters and checkpoint tensor metadata in their component context.",
  },
  {
    Icon: FileCheck2,
    title: "Verify the facts",
    copy: "Trace facts to artifacts and recorded evidence. See what is known, derived or still unresolved.",
  },
];
export function Landing() {
  return (
    <div className="public-page">
      <PublicHeader />
      <main id="main" tabIndex={-1}>
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">
              <span className="accent-dot" />
              THE MODEL, UNDER THE HOOD
            </span>
            <h1>
              From open weights
              <br />
              to clear understanding<span className="accent-text">.</span>
            </h1>
            <p>
              Explore a model’s structure. Inspect its tensors. Follow every
              fact back to the evidence.
            </p>
            <div className="hero-actions">
              <Link className="button button-primary" to="/app">
                Inspect a model <ArrowRight size={18} />
              </Link>
              <a className="text-link" href="#method">
                See how it works <span aria-hidden="true">↘</span>
              </a>
            </div>
            <div className="hero-footnote">
              <ShieldCheck size={16} />
              Static inspection · Pinned revisions · Explicit unknowns
            </div>
          </div>
          <figure
            className="method-illustration"
            aria-label="Illustration of the reconstruction process"
          >
            <div className="illustration-top">
              <span>
                <Network size={15} />
                THE RECONSTRUCTION PROCESS
              </span>
              <span className="subtle-label">ILLUSTRATION</span>
            </div>
            <div className="process-source">
              <Box size={24} />
              <div>
                <strong>Your Hugging Face model</strong>
                <span>Repository ID or model URL</span>
              </div>
              <span className="source-tag">INPUT</span>
            </div>
            <div className="process-line" />
            <div className="process-evidence">
              {[
                {
                  Icon: FileSearch,
                  title: "Artifact metadata",
                  copy: "Configuration · tensor headers",
                },
                {
                  Icon: GitBranch,
                  title: "Model structure",
                  copy: "Modules · semantic relationships",
                },
              ].map(({ Icon, title, copy }) => (
                <div key={title}>
                  <Icon size={19} />
                  <strong>{title}</strong>
                  <span>{copy}</span>
                </div>
              ))}
            </div>
            <div className="process-line" />
            <div className="process-result">
              <Network size={24} />
              <div>
                <strong>An inspectable architecture</strong>
                <span>Components, tensors and provenance</span>
              </div>
              <FileCheck2 size={20} />
            </div>
            <figcaption>
              A method illustration. Results depend on the model’s supported
              artifacts.
            </figcaption>
          </figure>
        </section>
        <section className="capabilities" aria-label="Capabilities">
          {capabilities.map(({ Icon, title, copy }, index) => (
            <article key={title}>
              <div className="capability-heading">
                <Icon size={22} />
                <span>0{index + 1}</span>
              </div>
              <h2>{title}</h2>
              <p>{copy}</p>
            </article>
          ))}
        </section>
        <section className="method-section" id="method">
          <div>
            <span className="eyebrow">BUILT FOR UNDERSTANDING</span>
            <h2>
              Inspect first.
              <br />
              Understand with evidence.
            </h2>
          </div>
          <div>
            <p>
              Model Anatomy combines configuration, instantiated module
              structure and bounded checkpoint metadata to reconstruct supported
              architectures.
            </p>
            <p>
              Inspection is static. Coverage does not prove runtime behavior.
              When architecture reconstruction is unavailable, you can inspect
              verified repository artifacts instead.
            </p>
            <Link className="text-link" to="/app">
              Bring a model into focus <ArrowRight size={16} />
            </Link>
          </div>
        </section>
        <section className="final-cta">
          <div>
            <span className="eyebrow">YOUR NEXT QUESTION STARTS HERE</span>
            <h2>What’s inside your model?</h2>
          </div>
          <Link className="button button-primary" to="/app">
            Inspect a model <ArrowRight size={18} />
          </Link>
        </section>
      </main>
      <footer className="public-footer">
        <span>Model Anatomy</span>
        <span>Artifact-backed inspection. Honest boundaries.</span>
        <Link to="/app">
          Workspace <ArrowRight size={14} />
        </Link>
      </footer>
    </div>
  );
}
export function Start() {
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const [model, setModel] = useState("");
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
      if (!active.current) return;
      void navigate(
        `/app/analyses/${encodeURIComponent(job.id)}?view=${job.status === "inspected" ? "artifacts" : mode === "learn" ? "overview" : "architecture"}`,
      );
    },
  });
  return (
    <div className="public-page start-page">
      <PublicHeader />
      <main id="main" tabIndex={-1} className="start-main">
        <div className="start-intro">
          <span className="eyebrow">INSPECTION WORKSPACE</span>
          <h1>Meet your model.</h1>
          <p>Start with a repository. Explore what the artifacts tell us.</p>
        </div>
        <section className="submission-card" aria-labelledby="submission-title">
          <div className="submission-heading">
            <h2 id="submission-title">Inspect a model</h2>
            <ModeControl />
          </div>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (model.trim() && available && !create.isPending)
                create.mutate(model.trim());
            }}
          >
            <label htmlFor="model-input">Hugging Face model</label>
            <div className="submission-input">
              <Box size={18} />
              <input
                id="model-input"
                required
                maxLength={512}
                placeholder="namespace/model or Hugging Face URL"
                autoComplete="off"
                value={model}
                onChange={(event) => setModel(event.target.value)}
              />
              <Button
                type="submit"
                variant="primary"
                disabled={!available || !model.trim() || create.isPending}
              >
                {create.isPending ? "Starting…" : "Inspect model"}
                <ArrowRight size={17} />
              </Button>
            </div>
            <p className="input-help">
              Paste a public model ID, repository URL or a pinned checkpoint
              file URL.
            </p>
          </form>
          <div className="mode-explanation">
            <strong>
              {mode === "learn"
                ? "Start with the bigger picture."
                : "Go straight to the architecture."}
            </strong>
            <p>
              {mode === "learn"
                ? "A guided introduction connects components, shapes, tensors and evidence."
                : "A compact technical workspace for exploring components and provenance."}{" "}
              You can switch modes anytime.
            </p>
          </div>
          {health.isPending && <Loading text="Checking inspection service" />}
          {!available && !health.isPending && (
            <div className="availability-note" role="status">
              <strong>New analyses unavailable</strong>
              <p>
                {service?.reason ??
                  health.error?.message ??
                  "The inspection service is not ready."}{" "}
                Existing analyses remain accessible.
              </p>
              <Button onClick={() => void health.refetch()}>Check again</Button>
            </div>
          )}
          {create.error && <ErrorNotice error={create.error} />}
        </section>
        <section className="recent-section" aria-labelledby="recent-title">
          <div className="section-title">
            <div>
              <History size={18} />
              <h2 id="recent-title">Your recent analyses</h2>
            </div>
            {jobs[0] && (
              <Link
                className="text-link"
                to={`/app/analyses/${encodeURIComponent(jobs[0].id)}`}
              >
                Continue recent analysis <ArrowRight size={16} />
              </Link>
            )}
          </div>
          <p className="muted">
            Analyses belong to this visitor session and remain available until
            they expire.
          </p>
          {recent.isPending && <Loading text="Loading recent analyses" />}
          {recent.error && (
            <ErrorNotice
              error={recent.error}
              retry={() => void recent.refetch()}
            />
          )}
          <div className="recent-list">
            {jobs.map((job) => (
              <Link
                className="recent-row"
                key={job.id}
                to={`/app/analyses/${encodeURIComponent(job.id)}`}
              >
                <span className="recent-icon">
                  <Box size={20} />
                </span>
                <span className="recent-model">
                  <strong>{job.model_id}</strong>
                  <small>{time(job.created_at)}</small>
                </span>
                <Status status={job.status} />
                <ArrowRight size={16} />
              </Link>
            ))}
          </div>
          {recent.data && !jobs.length && (
            <div className="recent-empty">
              <History size={24} />
              <h3>No analyses yet</h3>
              <p>
                Your submitted models will appear here. Start with the model you
                want to understand.
              </p>
            </div>
          )}
        </section>
        <div className="start-boundary">
          <ShieldCheck size={17} />
          <p>
            Actual artifacts, explicit limitations. Custom code is not executed.
            Architecture support varies; verified metadata stays useful.
          </p>
        </div>
      </main>
    </div>
  );
}
