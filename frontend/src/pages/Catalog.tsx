import { useState, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Search,
  Copy,
  Check,
  Layers,
  Cpu,
  Database,
  ExternalLink,
  Sparkles,
  SlidersHorizontal,
} from "lucide-react";
import { DitherBackground } from "../components/DitherBackground";
import { EntryHeader, PageFooter } from "./Entry";
import "./entry.css";

export interface CatalogModelItem {
  id: string;
  name: string;
  family: string;
  category: "decoder" | "sliding-window" | "encoder";
  params: string;
  layers: number;
  heads: number;
  kvHeads?: number;
  ctx: string;
  hiddenDim: number;
  attention: string;
  posEncoding: string;
  description: string;
  architectureTag: string;
  badge?: string;
}

export const catalogModels: CatalogModelItem[] = [
  {
    id: "HuggingFaceTB/SmolLM2-135M",
    name: "SmolLM2 · 135M",
    family: "Llama 3.2",
    category: "decoder",
    params: "134.5M",
    layers: 30,
    heads: 9,
    kvHeads: 3,
    ctx: "2,048",
    hiddenDim: 576,
    attention: "GQA (3:1 Query-to-KV)",
    posEncoding: "RoPE (theta = 500,000)",
    description: "Ultra-compact decoder from HuggingFaceTB designed for fast local reasoning with genuine Llama 3.2 block architecture.",
    architectureTag: "LlamaForCausalLM",
    badge: "Recommended",
  },
  {
    id: "HuggingFaceTB/SmolLM2-360M",
    name: "SmolLM2 · 360M",
    family: "Llama 3.2",
    category: "decoder",
    params: "362M",
    layers: 32,
    heads: 15,
    kvHeads: 5,
    ctx: "2,048",
    hiddenDim: 960,
    attention: "GQA (3:1 Query-to-KV)",
    posEncoding: "RoPE (theta = 500,000)",
    description: "Mid-scale SmolLM2 architecture trained on 4 trillion tokens, balancing parameter density with multi-step reasoning capabilities.",
    architectureTag: "LlamaForCausalLM",
  },
  {
    id: "meta-llama/Llama-3.2-1B",
    name: "Llama 3.2 · 1B",
    family: "Modern Decoder",
    category: "decoder",
    params: "1.23B",
    layers: 16,
    heads: 32,
    kvHeads: 8,
    ctx: "8,192",
    hiddenDim: 2048,
    attention: "GQA (4:1 Query-to-KV)",
    posEncoding: "RoPE (theta = 500,000)",
    description: "Meta's flagship lightweight decoder featuring grouped-query attention and expanded 8k context window support.",
    architectureTag: "LlamaForCausalLM",
    badge: "Popular",
  },
  {
    id: "mistralai/Mistral-7B-v0.1",
    name: "Mistral · 7B",
    family: "Sliding Window",
    category: "sliding-window",
    params: "7.24B",
    layers: 32,
    heads: 32,
    kvHeads: 8,
    ctx: "32,768",
    hiddenDim: 4096,
    attention: "Sliding Window Attention (4096 window) + GQA",
    posEncoding: "RoPE (theta = 10,000)",
    description: "Pioneering sliding window attention architecture achieving high decoding throughput across 32k token contexts.",
    architectureTag: "MistralForCausalLM",
  },
  {
    id: "Qwen/Qwen2.5-0.5B",
    name: "Qwen 2.5 · 0.5B",
    family: "Dense Transformer",
    category: "decoder",
    params: "490M",
    layers: 24,
    heads: 14,
    kvHeads: 2,
    ctx: "32,768",
    hiddenDim: 896,
    attention: "GQA (7:1 Query-to-KV) + Dual-chunk",
    posEncoding: "RoPE + SwiGLU MLP",
    description: "Alibaba's dense transformer optimized with SwiGLU non-linearities and 32k context capability in under 500M weights.",
    architectureTag: "Qwen2ForCausalLM",
  },
  {
    id: "openai-community/gpt2",
    name: "GPT-2 · Small",
    family: "Causal Decoder",
    category: "decoder",
    params: "124.4M",
    layers: 12,
    heads: 12,
    kvHeads: 12,
    ctx: "1,024",
    hiddenDim: 768,
    attention: "Full MHA (1:1 Query-to-KV)",
    posEncoding: "Learned Absolute Position Embeddings",
    description: "The classic foundation causal transformer with 12 transformer decoder blocks and pre-LayerNorm residual flow.",
    architectureTag: "GPT2LMHeadModel",
  },
  {
    id: "google-bert/bert-base-uncased",
    name: "BERT · Base",
    family: "Bidirectional Encoder",
    category: "encoder",
    params: "109.5M",
    layers: 12,
    heads: 12,
    kvHeads: 12,
    ctx: "512",
    hiddenDim: 768,
    attention: "Full Bidirectional Self-Attention",
    posEncoding: "Learned Absolute Position Embeddings",
    description: "Google's landmark masked language model encoder with bidirectional token representations across 12 layers.",
    architectureTag: "BertForMaskedLM",
  },
  {
    id: "distilbert/distilbert-base-uncased",
    name: "DistilBERT",
    family: "Compact Encoder",
    category: "encoder",
    params: "66.4M",
    layers: 6,
    heads: 12,
    kvHeads: 12,
    ctx: "512",
    hiddenDim: 768,
    attention: "Distilled Bidirectional Attention",
    posEncoding: "Learned Position Embeddings",
    description: "Knowledge-distilled version of BERT retaining 97% of language understanding with 40% fewer parameters and 6 layers.",
    architectureTag: "DistilBertForMaskedLM",
  },
];

type CategoryFilter = "all" | "decoder" | "sliding-window" | "encoder";

export default function Catalog() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState<CategoryFilter>("all");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredModels = useMemo(() => {
    return catalogModels.filter((model) => {
      const matchesCategory =
        activeCategory === "all" || model.category === activeCategory;
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch =
        !q ||
        model.name.toLowerCase().includes(q) ||
        model.id.toLowerCase().includes(q) ||
        model.family.toLowerCase().includes(q) ||
        model.architectureTag.toLowerCase().includes(q) ||
        model.description.toLowerCase().includes(q);
      return matchesCategory && matchesSearch;
    });
  }, [searchQuery, activeCategory]);

  const handleCopyId = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleInspect = (modelId: string) => {
    navigate(`/app?model=${encodeURIComponent(modelId)}`);
  };

  return (
    <div className="entry-page">
      <DitherBackground />
      <EntryHeader />

      <main id="main" tabIndex={-1} className="catalog-page-main">
        {/* Catalog Hero Banner */}
        <header className="catalog-hero-section">
          <div className="catalog-hero-inner">
            <div className="catalog-eyebrow-badge">
              <span className="entry-badge-dot" />
              <span>Model Anatomy Architecture Registry</span>
              <span className="badge-ver">{catalogModels.length} Models Verified</span>
            </div>
            <h1 className="catalog-page-title">
              Verified Model <span className="hero-highlight-amber">Catalog</span>
            </h1>
            <p className="catalog-page-subtitle">
              Pre-verified open-weight architectures across transformer decoders, sliding-window attention, and bidirectional encoders. Inspect tensor dimensions, layer ASTs, and SafeTensors headers instantly without downloading weights.
            </p>
          </div>
        </header>

        {/* Filters and Search Bar */}
        <section className="catalog-toolbar-section" aria-label="Catalog filters">
          <div className="catalog-toolbar-inner">
            <div className="catalog-search-box">
              <Search size={16} className="search-icon" aria-hidden="true" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by model name, family, or Hugging Face ID..."
                className="catalog-search-input"
                aria-label="Filter model catalog"
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

            <div className="catalog-filter-tabs" role="tablist" aria-label="Model category">
              <button
                type="button"
                role="tab"
                aria-selected={activeCategory === "all"}
                className={`filter-tab ${activeCategory === "all" ? "is-active" : ""}`}
                onClick={() => setActiveCategory("all")}
              >
                All Models ({catalogModels.length})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeCategory === "decoder"}
                className={`filter-tab ${activeCategory === "decoder" ? "is-active" : ""}`}
                onClick={() => setActiveCategory("decoder")}
              >
                Decoder-Only ({catalogModels.filter((m) => m.category === "decoder").length})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeCategory === "sliding-window"}
                className={`filter-tab ${activeCategory === "sliding-window" ? "is-active" : ""}`}
                onClick={() => setActiveCategory("sliding-window")}
              >
                Sliding Window ({catalogModels.filter((m) => m.category === "sliding-window").length})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeCategory === "encoder"}
                className={`filter-tab ${activeCategory === "encoder" ? "is-active" : ""}`}
                onClick={() => setActiveCategory("encoder")}
              >
                Encoder ({catalogModels.filter((m) => m.category === "encoder").length})
              </button>
            </div>
          </div>
        </section>

        {/* Model Cards Grid */}
        <section className="catalog-models-section" aria-labelledby="catalog-list-heading">
          <div className="catalog-models-inner">
            <h2 id="catalog-list-heading" className="sr-only">
              Verified Model Architectures List
            </h2>

            {filteredModels.length === 0 ? (
              <div className="catalog-empty-state">
                <SlidersHorizontal size={32} className="empty-icon" />
                <h3>No verified models match &ldquo;{searchQuery}&rdquo;</h3>
                <p>
                  You can inspect any open-weight model directly by pasting its Hugging Face repository URL on the home inspection page.
                </p>
                <Link to="/#submit-title" className="catalog-empty-btn">
                  <span>Inspect custom repository</span>
                  <ArrowRight size={14} />
                </Link>
              </div>
            ) : (
              <div className="catalog-cards-grid">
                {filteredModels.map((item) => (
                  <article key={item.id} className="catalog-detail-card">
                    <div className="card-top-row">
                      <div className="card-tags-left">
                        <span className="card-family-tag">{item.family}</span>
                        {item.badge && <span className="card-badge-tag">{item.badge}</span>}
                      </div>
                      <span className="card-params-tag">{item.params}</span>
                    </div>

                    <div className="card-heading-group">
                      <h3 className="card-model-title">{item.name}</h3>
                      <div className="card-id-row">
                        <code className="card-repo-id">{item.id}</code>
                        <button
                          type="button"
                          className="copy-id-btn"
                          onClick={(e) => handleCopyId(item.id, e)}
                          title="Copy repository ID"
                          aria-label={`Copy repository ID for ${item.name}`}
                        >
                          {copiedId === item.id ? (
                            <Check size={13} className="copied-check" />
                          ) : (
                            <Copy size={13} />
                          )}
                        </button>
                      </div>
                    </div>

                    <p className="card-description">{item.description}</p>

                    <div className="card-specs-matrix">
                      <div className="spec-metric">
                        <span className="metric-label">Layers</span>
                        <strong className="metric-val">{item.layers}</strong>
                      </div>
                      <div className="spec-metric">
                        <span className="metric-label">Heads (Q/KV)</span>
                        <strong className="metric-val">
                          {item.heads} {item.kvHeads ? `/ ${item.kvHeads}` : ""}
                        </strong>
                      </div>
                      <div className="spec-metric">
                        <span className="metric-label">Context</span>
                        <strong className="metric-val">{item.ctx}</strong>
                      </div>
                      <div className="spec-metric">
                        <span className="metric-label">Hidden Dim</span>
                        <strong className="metric-val">{item.hiddenDim}</strong>
                      </div>
                    </div>

                    <div className="card-technical-details">
                      <div className="tech-row">
                        <span className="tech-label">Attention:</span>
                        <span className="tech-value">{item.attention}</span>
                      </div>
                      <div className="tech-row">
                        <span className="tech-label">Encoding:</span>
                        <span className="tech-value">{item.posEncoding}</span>
                      </div>
                      <div className="tech-row">
                        <span className="tech-label">AST Class:</span>
                        <code className="tech-class">{item.architectureTag}</code>
                      </div>
                    </div>

                    <div className="card-footer-actions">
                      <button
                        type="button"
                        className="card-inspect-btn"
                        onClick={() => handleInspect(item.id)}
                        aria-label={`Inspect ${item.name} architecture`}
                      >
                        <span>Inspect Architecture</span>
                        <ArrowRight size={14} />
                      </button>
                      <a
                        href={`https://huggingface.co/${item.id}`}
                        target="_blank"
                        rel="noreferrer"
                        className="card-hf-btn"
                        title="View repository on Hugging Face"
                        aria-label={`Open ${item.name} on Hugging Face`}
                      >
                        <ExternalLink size={14} />
                      </a>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      <PageFooter />
    </div>
  );
}
