import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  History,
  Layers,
  Network,
  ShieldCheck,
  Cpu,
  Activity,
  Zap,
  CheckCircle2,
  ExternalLink,
  Copy,
  Check,
  Database,
  BarChart3,
  Box,
  Eye,
  Terminal,
  HelpCircle,
} from "lucide-react";
import { api } from "../api/client";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Button, ThemeControl } from "../components/ui";
import { ModelAnatomyMark } from "../components/Logo";
import { useRecent, useService } from "../state/queries";
import { DotPattern } from "../components/vendor/dot-pattern";
import "./entry.css";

const examples = [
  { id: "HuggingFaceTB/SmolLM2-135M", label: "SmolLM2 · 135M", family: "Llama 3.2 · 30L" },
  { id: "google-bert/bert-base-uncased", label: "BERT base", family: "Encoder · 12L" },
  { id: "distilbert/distilbert-base-uncased", label: "DistilBERT", family: "Compact · 6L" },
];

const modelCatalog = [
  {
    id: "HuggingFaceTB/SmolLM2-135M",
    name: "SmolLM2 · 135M",
    family: "Llama 3.2",
    params: "134.5M",
    layers: 30,
    heads: 9,
    ctx: "2,048",
    desc: "Compact autoregressive decoder with RoPE & SwiGLU",
  },
  {
    id: "openai-community/gpt2",
    name: "GPT-2 · Small",
    family: "Causal Transformer",
    params: "124.4M",
    layers: 12,
    heads: 12,
    ctx: "1,024",
    desc: "Classic decoder-only language model with learned embeddings",
  },
  {
    id: "google-bert/bert-base-uncased",
    name: "BERT · Base",
    family: "Bidirectional Encoder",
    params: "109.5M",
    layers: 12,
    heads: 12,
    ctx: "512",
    desc: "Masked language model encoder with segment embeddings",
  },
  {
    id: "meta-llama/Llama-3.2-1B",
    name: "Llama 3.2 · 1B",
    family: "Modern Decoder",
    params: "1.23B",
    layers: 16,
    heads: 32,
    ctx: "8,192",
    desc: "High-efficiency open weights with Grouped-Query Attention",
  },
  {
    id: "mistralai/Mistral-7B-v0.1",
    name: "Mistral · 7B",
    family: "Sliding Window",
    params: "7.24B",
    layers: 32,
    heads: 32,
    ctx: "32,768",
    desc: "Extended context window model with sliding window attention",
  },
  {
    id: "Qwen/Qwen2.5-0.5B",
    name: "Qwen 2.5 · 0.5B",
    family: "Dense Transformer",
    params: "490M",
    layers: 24,
    heads: 14,
    ctx: "32,768",
    desc: "State-of-the-art compact multi-lingual architecture",
  },
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

function FooterBrand() {
  return (
    <span className="entry-footer-brand">
      <div className="brand-gem footer-gem" aria-hidden="true">
        <ModelAnatomyMark size={14} />
      </div>
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
      <div className="entry-backdrop" aria-hidden="true">
        <DotPattern width={22} height={22} cr={0.8} />
      </div>
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
              <h1 id="entry-title">See how a model is built.</h1>
              <p className="entry-lede">
                Inspect architecture, tensors and evidence from an open-weight
                Hugging Face model.
              </p>
            </div>
          </div>

          <section className="entry-submit" aria-labelledby="submit-title">
            <div className="border-beam" aria-hidden="true" />
            <div className="entry-submit-heading">
              <div>
                <h2 id="submit-title">Inspect a model</h2>
              </div>
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

          {/* Quick Stats Ribbon */}
          <div className="entry-stats-ribbon" role="region" aria-label="Key metrics">
            <div className="stat-card">
              <span className="stat-number">0 MB</span>
              <span className="stat-label">Weights Downloaded</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">&lt; 85ms</span>
              <span className="stat-label">SafeTensors Parse</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">100%</span>
              <span className="stat-label">Deterministic Evidence</span>
            </div>
            <div className="stat-card">
              <span className="stat-number">134M - 70B+</span>
              <span className="stat-label">Model Scale Range</span>
            </div>
          </div>

          {/* Interactive Live Architecture Blueprint Teaser */}
          <section
            className="entry-interactive-teaser"
            aria-labelledby="teaser-heading"
          >
            <div className="teaser-header">
              <div>
                <span className="teaser-eyebrow">Interactive Blueprint Preview</span>
                <h2 id="teaser-heading">Explore neural architecture flow</h2>
              </div>
              <div className="teaser-model-tabs" role="tablist">
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

            <div className="teaser-layout">
              {/* Flow preview track */}
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

          {/* Expanded Modern Bento Grid */}
          <div
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
                Instantiated modules, attention heads, and residual streams
                mapped from genuine PyTorch code without LLM guesswork.
              </p>
              <div className="bento-visual-preview graph-preview" aria-hidden="true">
                <div className="mock-node mock-node-attn">
                  <span>Self-Attention (QKV)</span>
                  <code>[B, S, 576]</code>
                </div>
                <div className="mock-connector" />
                <div className="mock-node mock-node-mlp">
                  <span>SwiGLU MLP Block</span>
                  <code>[B, S, 1536]</code>
                </div>
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
                Zero weight downloads. Verified SafeTensors headers, dimensions,
                tensor offsets, and byte boundaries parsed under 100ms.
              </p>
              <div className="bento-visual-preview stream-preview" aria-hidden="true">
                <div className="stream-badge">
                  <Zap size={11} /> <span>Header Stream Active</span>
                </div>
                <code>offset: 0x0041a0 · BF16 · 82.4 KB</code>
              </div>
            </div>

            <div className="entry-bento-card">
              <div className="entry-bento-header">
                <span className="entry-bento-icon" aria-hidden="true">
                  <BarChart3 size={15} />
                </span>
                <span>Executive Scorecard</span>
              </div>
              <p className="entry-bento-desc">
                Real-time parameter breakdown, attention head ratios, layer counts,
                and parameter ownership charts.
              </p>
              <div className="bento-visual-preview gauge-preview" aria-hidden="true">
                <div className="bar-split">
                  <div className="seg seg-attn" style={{ width: "38%" }} title="Attention 38%" />
                  <div className="seg seg-mlp" style={{ width: "44%" }} title="MLP 44%" />
                  <div className="seg seg-emb" style={{ width: "18%" }} title="Embed 18%" />
                </div>
                <div className="seg-legend">
                  <span>Attention 38%</span>
                  <span>MLP 44%</span>
                  <span>Embed 18%</span>
                </div>
              </div>
            </div>

            <div className="entry-bento-card">
              <div className="entry-bento-header">
                <span className="entry-bento-icon" aria-hidden="true">
                  <ShieldCheck size={15} />
                </span>
                <span>Deterministic Evidence</span>
              </div>
              <p className="entry-bento-desc">
                100% artifact-backed. Every component links directly to pinned
                Hugging Face checkpoint files and SHA-256 verified commits.
              </p>
              <div className="bento-visual-preview audit-preview" aria-hidden="true">
                <div className="audit-row">
                  <CheckCircle2 size={13} className="audit-check" />
                  <span>Commit: 93efa2f0 (Verified)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Directory Showcase */}
          <section className="entry-catalog-showcase" aria-labelledby="catalog-heading">
            <div className="catalog-header">
              <div>
                <span className="teaser-eyebrow">Supported Model Families</span>
                <h2 id="catalog-heading">Explore verified open architectures</h2>
              </div>
              <span className="catalog-badge">Ready to inspect</span>
            </div>
            <div className="catalog-grid">
              {modelCatalog.map((item) => (
                <div key={item.id} className="catalog-card">
                  <div className="catalog-card-head">
                    <span className="catalog-family-pill">{item.family}</span>
                    <span className="catalog-params-pill">{item.params}</span>
                  </div>
                  <strong className="catalog-name">{item.name}</strong>
                  <p className="catalog-desc">{item.desc}</p>
                  <div className="catalog-meta-row">
                    <span>{item.layers} layers</span>
                    <span>·</span>
                    <span>{item.heads} heads</span>
                    <span>·</span>
                    <span>{item.ctx} ctx</span>
                  </div>
                  <button
                    type="button"
                    className="catalog-btn"
                    onClick={() => {
                      setModel(item.id);
                      const input = document.getElementById("model-input");
                      input?.focus();
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                  >
                    <span>Load into inspector</span>
                    <ArrowRight size={13} />
                  </button>
                </div>
              ))}
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
