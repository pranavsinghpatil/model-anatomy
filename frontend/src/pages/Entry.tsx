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
  CheckCircle2,
} from "lucide-react";
import { api } from "../api/client";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Button, ThemeControl } from "../components/ui";
import { ModelAnatomyMark } from "../components/Logo";
import { useRecent, useService } from "../state/queries";
import { DitherBackground } from "../components/DitherBackground";
import "./entry.css";

const examples = [
  { id: "HuggingFaceTB/SmolLM2-135M", label: "SmolLM2 · 135M", family: "Llama 3.2 · 30L" },
  { id: "google-bert/bert-base-uncased", label: "BERT base", family: "Encoder · 12L" },
  { id: "distilbert/distilbert-base-uncased", label: "DistilBERT", family: "Compact · 6L" },
];



type TeaserNode = {
  id: string;
  kind: string;
  label: string;
  shape: string;
  params: string;
  share: string;
  dtype: string;
  moduleType: string;
};

const teaserModels: Record<
  string,
  {
    title: string;
    totalParams: string;
    layersCount: number;
    nodes: TeaserNode[];
  }
> = {
  "HuggingFaceTB/SmolLM2-135M": {
    title: "SmolLM2 · 135M (Llama 3.2)",
    totalParams: "134,515,008",
    layersCount: 30,
    nodes: [
      {
        id: "input_ids",
        kind: "input",
        label: "Token IDs",
        shape: "[batch_size, seq_len]",
        params: "0",
        share: "0%",
        dtype: "int64",
        moduleType: "Boundary Input",
      },
      {
        id: "embed_tokens",
        kind: "embedding",
        label: "Token Embedding",
        shape: "[49152, 576]",
        params: "28,311,552",
        share: "21.0%",
        dtype: "bfloat16",
        moduleType: "nn.Embedding",
      },
      {
        id: "decoder_layers",
        kind: "structure",
        label: "Decoder Block ×30",
        shape: "[batch_size, seq_len, 576]",
        params: "85,054,464",
        share: "63.2%",
        dtype: "bfloat16",
        moduleType: "LlamaDecoderLayer ×30",
      },
      {
        id: "norm",
        kind: "normalization",
        label: "Final RMSNorm",
        shape: "[576]",
        params: "576",
        share: "<0.1%",
        dtype: "bfloat16",
        moduleType: "LlamaRMSNorm",
      },
      {
        id: "lm_head",
        kind: "projection",
        label: "LM Head",
        shape: "[576, 49152]",
        params: "28,311,552",
        share: "21.0% (tied)",
        dtype: "bfloat16",
        moduleType: "nn.Linear",
      },
      {
        id: "output",
        kind: "output",
        label: "Output Logits",
        shape: "[batch_size, seq_len, 49152]",
        params: "0",
        share: "0%",
        dtype: "bfloat16",
        moduleType: "Boundary Output",
      },
    ],
  },
  "openai-community/gpt2": {
    title: "GPT-2 · 124M (OpenAI)",
    totalParams: "124,439,808",
    layersCount: 12,
    nodes: [
      {
        id: "input_ids",
        kind: "input",
        label: "Token IDs",
        shape: "[batch_size, seq_len]",
        params: "0",
        share: "0%",
        dtype: "int64",
        moduleType: "Boundary Input",
      },
      {
        id: "wte",
        kind: "embedding",
        label: "Token Embedding",
        shape: "[50257, 768]",
        params: "38,597,376",
        share: "31.0%",
        dtype: "float32",
        moduleType: "nn.Embedding",
      },
      {
        id: "wpe",
        kind: "position",
        label: "Position Embedding",
        shape: "[1024, 768]",
        params: "786,432",
        share: "0.6%",
        dtype: "float32",
        moduleType: "nn.Embedding",
      },
      {
        id: "transformer_blocks",
        kind: "structure",
        label: "Transformer Block ×12",
        shape: "[batch_size, seq_len, 768]",
        params: "85,054,464",
        share: "68.3%",
        dtype: "float32",
        moduleType: "GPT2Block ×12",
      },
      {
        id: "ln_f",
        kind: "normalization",
        label: "Final LayerNorm",
        shape: "[768]",
        params: "1,536",
        share: "<0.1%",
        dtype: "float32",
        moduleType: "nn.LayerNorm",
      },
      {
        id: "output",
        kind: "output",
        label: "Output Logits",
        shape: "[batch_size, seq_len, 50257]",
        params: "0",
        share: "0%",
        dtype: "float32",
        moduleType: "Boundary Output",
      },
    ],
  },
  "google-bert/bert-base-uncased": {
    title: "BERT · Base (110M)",
    totalParams: "109,482,240",
    layersCount: 12,
    nodes: [
      {
        id: "input_ids",
        kind: "input",
        label: "Input IDs",
        shape: "[batch_size, seq_len]",
        params: "0",
        share: "0%",
        dtype: "int64",
        moduleType: "Boundary Input",
      },
      {
        id: "bert_embeddings",
        kind: "embedding",
        label: "Word & Pos Embeddings",
        shape: "[30522, 768]",
        params: "23,835,648",
        share: "21.8%",
        dtype: "float32",
        moduleType: "BertEmbeddings",
      },
      {
        id: "encoder_layers",
        kind: "structure",
        label: "Encoder Block ×12",
        shape: "[batch_size, seq_len, 768]",
        params: "85,054,464",
        share: "77.7%",
        dtype: "float32",
        moduleType: "BertLayer ×12",
      },
      {
        id: "pooler",
        kind: "projection",
        label: "BertPooler",
        shape: "[768, 768]",
        params: "590,592",
        share: "0.5%",
        dtype: "float32",
        moduleType: "BertPooler",
      },
      {
        id: "output",
        kind: "output",
        label: "Pooled Output",
        shape: "[batch_size, 768]",
        params: "0",
        share: "0%",
        dtype: "float32",
        moduleType: "Boundary Output",
      },
    ],
  },
};

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
            <Link to="/" className="footer-brand" aria-label="Model Anatomy home">
              <div className="brand-gem footer-gem" aria-hidden="true">
                <ModelAnatomyMark size={15} />
              </div>
              <span className="entry-brand-name">Model Anatomy</span>
              <span className="brand-tag">STUDIO</span>
            </Link>
            <p className="footer-mission">
              Reconstructing trustworthy, semantic, tensor-aware architecture graphs from open-weight Hugging Face models without downloading weights.
            </p>
            <div className="footer-tech-badges">
              <span className="tech-badge">PyTorch AST</span>
              <span className="tech-badge">SafeTensors Stream</span>
              <span className="tech-badge">Deterministic IR</span>
            </div>
          </div>

          {/* Column 2: Architecture Views */}
          <div className="footer-col">
            <h4 className="footer-col-title">Inspection Views</h4>
            <ul className="footer-nav-list">
              <li><a href="/#blueprint">Architecture Graph</a></li>
              <li><a href="/#blueprint">SafeTensors Headers</a></li>
              <li><a href="/#blueprint">Parameter &amp; Weight BOM</a></li>
              <li><a href="/#blueprint">Module AST Hierarchy</a></li>
            </ul>
          </div>

          {/* Column 3: Model Registry */}
          <div className="footer-col">
            <h4 className="footer-col-title">Model Registry</h4>
            <ul className="footer-nav-list">
              <li><Link to="/catalog">Model Catalog</Link></li>
              <li><Link to="/catalog">SmolLM2 &amp; Llama 3.2</Link></li>
              <li><Link to="/catalog">Mistral Sliding Window</Link></li>
              <li><Link to="/catalog">BERT &amp; DistilBERT</Link></li>
            </ul>
          </div>

          {/* Column 4: Evidence & Provenance */}
          <div className="footer-col">
            <h4 className="footer-col-title">Provenance</h4>
            <ul className="footer-nav-list">
              <li><Link to="/catalog">Verified Models</Link></li>
              <li><a href="/#main">Support Boundaries</a></li>
              <li><span className="footer-static-note">Zero Weight Downloads</span></li>
              <li><span className="footer-static-note">Pinned Revision Audit</span></li>
            </ul>
          </div>
        </div>

        {/* Footer Bottom Bar */}
        <div className="footer-bottom-bar">
          <div className="footer-copyright">
            <span>&copy; {new Date().getFullYear()} Model Anatomy Studio.</span>
            <span className="footer-sep">&middot;</span>
            <span className="footer-evidence-note">Artifact-backed inspection with explicit evidence.</span>
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
            <a href={isCatalog ? "/#blueprint" : "#blueprint"} className="entry-nav-link">
              <span>Blueprint</span>
            </a>
            <a href={isCatalog ? "/#metrics" : "#metrics"} className="entry-nav-link">
              <span>Metrics</span>
            </a>
            <a href={isCatalog ? "/#capabilities" : "#capabilities"} className="entry-nav-link">
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
          <Link className="entry-history-link" to="/catalog">
            <History size={14} />
            <span>Your analyses</span>
          </Link>
          <a className="entry-header-cta" href={isCatalog ? "/#submit-title" : "#submit-title"}>
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
  const [selectedTeaserModel, setSelectedTeaserModel] = useState(
    "HuggingFaceTB/SmolLM2-135M",
  );
  const [activeTeaserNodeId, setActiveTeaserNodeId] = useState("decoder_layers");

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
      if (
        !active.current ||
        window.location.pathname !== submissionPath.current
      )
        return;
      const view = job.status === "inspected" ? "artifacts" : "architecture";
      void navigate(`/app/analyses/${encodeURIComponent(job.id)}?view=${view}`);
    },
  });

  const teaserData = teaserModels[selectedTeaserModel] ?? teaserModels["HuggingFaceTB/SmolLM2-135M"];
  const activeTeaserNode =
    teaserData.nodes.find((n) => n.id === activeTeaserNodeId) ??
    teaserData.nodes[2] ??
    teaserData.nodes[0];

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
                <span className="badge-ver">v1.0</span>
              </div>
              <h1 id="entry-title">
                See how a <br />
                <span className="hero-highlight-amber">model is built.</span>
              </h1>
              <p className="entry-lede">
                Schematic layer flow, SafeTensors headers, parameter allocations, and
                deterministic evidence — reconstructed live without downloading model weights.
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
                Public model repositories and pinned checkpoint file URLs. Model weights are not downloaded.
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
                    <span className="chip-family">{example.family}</span>
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
          </section>

          {/* Optimized Architectural Telemetry Ribbon */}
          <div id="metrics" className="entry-stats-ribbon" role="region" aria-label="Key metrics">
            <div className="stat-card">
              <span className="stat-number">0 MB</span>
              <span className="stat-label">Weights Downloaded</span>
              <span className="stat-sub">HTTP range-request streaming</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">&lt; 85ms</span>
              <span className="stat-label">SafeTensors Stream</span>
              <span className="stat-sub">Metadata &amp; tensor headers</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">100%</span>
              <span className="stat-label">Deterministic Graph</span>
              <span className="stat-sub">Genuine PyTorch IR extraction</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">134M - 70B+</span>
              <span className="stat-label">Verified Scale</span>
              <span className="stat-sub">Decoder &amp; encoder models</span>
            </div>
          </div>

          {/* Interactive Live Architecture Blueprint Showcase */}
          <section
            id="blueprint"
            className="entry-interactive-teaser"
            aria-labelledby="teaser-heading"
          >
            <div className="teaser-header">
              <div>
                <span className="teaser-eyebrow">Interactive Blueprint Preview</span>
                <h2 id="teaser-heading">
                  One model, <span className="hero-highlight-amber">four inspection views</span>.
                </h2>
              </div>
              <div className="teaser-model-tabs" role="tablist" aria-label="Model switcher">
                {Object.entries(teaserModels).map(([mId, mData]) => (
                  <button
                    key={mId}
                    role="tab"
                    aria-selected={selectedTeaserModel === mId}
                    className={`teaser-tab ${selectedTeaserModel === mId ? "active" : ""}`}
                    onClick={() => {
                      setSelectedTeaserModel(mId);
                      setActiveTeaserNodeId(mData.nodes[2]?.id ?? mData.nodes[0].id);
                    }}
                  >
                    {mData.title.split(" (")[0]}
                  </button>
                ))}
              </div>
            </div>

            {/* Multiview selector pills */}
            <div className="teaser-multiview-pills" role="tablist" aria-label="Inspection views">
              <button type="button" className="multiview-pill is-active" aria-selected="true">
                <span className="pill-dot dot-cyan" />
                <span>Architecture Graph</span>
              </button>
              <button type="button" className="multiview-pill" aria-selected="false">
                <span className="pill-dot dot-indigo" />
                <span>SafeTensors Headers</span>
              </button>
              <button type="button" className="multiview-pill" aria-selected="false">
                <span className="pill-dot dot-amber" />
                <span>Parameter BOM</span>
              </button>
              <button type="button" className="multiview-pill" aria-selected="false">
                <span className="pill-dot dot-emerald" />
                <span>Module Hierarchy</span>
              </button>
            </div>

            <div className="teaser-layout">
              {/* Flow preview track */}
              <div className="teaser-flow-track-wrap">
                <div className="teaser-flow-track" aria-label="Architecture node flow">
                  {teaserData.nodes.map((n, idx) => {
                    const isSelected = activeTeaserNode.id === n.id;
                    return (
                      <div key={n.id} className="teaser-flow-step">
                        <button
                          type="button"
                          className={`teaser-node-card category-${n.kind} ${isSelected ? "is-active" : ""}`}
                          onClick={() => setActiveTeaserNodeId(n.id)}
                          aria-pressed={isSelected}
                        >
                          <div className="teaser-node-badge">
                            <span className="node-category-dot" />
                            <span>{n.kind.toUpperCase()}</span>
                          </div>
                          <strong className="teaser-node-title">{n.label}</strong>
                          <div className="teaser-node-meta">
                            <code>{n.shape}</code>
                            {n.params !== "0" && (
                              <span className="teaser-param-pill">{n.params}</span>
                            )}
                          </div>
                        </button>
                        {idx < teaserData.nodes.length - 1 && (
                          <div className="teaser-flow-arrow" aria-hidden="true">
                            <div className="arrow-pulse" />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Teaser Inspector Card */}
              <div className="teaser-inspector-card" aria-live="polite">
                <div className="teaser-inspector-head">
                  <div className="inspector-chip-row">
                    <span className="eyebrow">{activeTeaserNode.kind}</span>
                    <span className="verified-pill">
                      <CheckCircle2 size={12} /> SafeTensors Verified
                    </span>
                  </div>
                  <h3>{activeTeaserNode.label}</h3>
                  <code className="teaser-module-tag">{activeTeaserNode.moduleType}</code>
                </div>

                <div className="teaser-spec-grid">
                  <div className="spec-item">
                    <span className="spec-label">Shape</span>
                    <code className="spec-value">{activeTeaserNode.shape}</code>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Precision</span>
                    <span className="spec-value dtype-pill">{activeTeaserNode.dtype}</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Parameters</span>
                    <span className="spec-value font-mono">{activeTeaserNode.params}</span>
                  </div>
                  <div className="spec-item">
                    <span className="spec-label">Share of Model</span>
                    <span className="spec-value highlight">{activeTeaserNode.share}</span>
                  </div>
                </div>

                <div className="teaser-action-row">
                  <button
                    type="button"
                    className="teaser-load-btn"
                    onClick={() => {
                      setModel(selectedTeaserModel);
                      const input = document.getElementById("model-input");
                      input?.focus();
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                  >
                    <span>Inspect this model</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>
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
                derived directly from genuine PyTorch code without LLM guesswork.
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
                Zero weight downloads. Verified SafeTensors headers, tensor offsets,
                dimensions, and dtypes parsed via HTTP range requests in &lt;100ms.
              </p>
              <div className="bento-badge-strip">
                <span className="bento-tag">Range Requests</span>
                <span className="bento-tag">Zero Weights</span>
                <span className="bento-tag">&lt; 85ms Latency</span>
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
                100% artifact-backed. Complete parameter BOM, memory footprint,
                weight tying analysis, and pinned SHA-256 commit evidence.
              </p>
              <div className="bento-badge-strip">
                <span className="bento-tag">SHA-256 Verified</span>
                <span className="bento-tag">Parameter BOM</span>
                <span className="bento-tag">Memory Footprint</span>
              </div>
            </div>
          </div>

          {/* Architecture Catalog Banner */}
          <section id="catalog" className="entry-catalog-banner" aria-labelledby="catalog-heading">
            <div className="catalog-banner-glow" aria-hidden="true" />
            <div className="catalog-banner-content">
              <div className="catalog-banner-info">
                <div className="catalog-pill-badge">
                  <span className="entry-badge-dot" />
                  <span>Architecture Registry</span>
                  <span className="catalog-count-tag">8 Pre-verified</span>
                </div>
                <h2 id="catalog-heading" className="catalog-banner-title">
                  Explore verified model architectures
                </h2>
                <p className="catalog-banner-desc">
                  Inspect multi-scale decoders, sliding-window attention networks, and bidirectional encoders. Review tensor shapes, layer counts, and SafeTensors headers with zero weight downloads.
                </p>
                <div className="catalog-banner-tags">
                  <span className="cat-chip">SmolLM2 · 135M / 360M</span>
                  <span className="cat-chip">Llama 3.2 · 1B</span>
                  <span className="cat-chip">Mistral · 7B SWA</span>
                  <span className="cat-chip">Qwen 2.5 · 0.5B</span>
                  <span className="cat-chip">BERT Base &amp; DistilBERT</span>
                </div>
              </div>
              <div className="catalog-banner-actions">
                <Link to="/catalog" className="catalog-cta-btn">
                  <span>Browse Full Catalog</span>
                  <ArrowRight size={15} />
                </Link>
                <span className="catalog-cta-sub">
                  Instant live inspection &amp; specifications
                </span>
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

        {jobs.length > 0 && (
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
              Your submitted analyses. Select one to continue.
            </p>
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
          </section>
        )}
      </main>
      <PageFooter />
    </div>
  );
}
