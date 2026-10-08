import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  Box,
  Braces,
  ChevronDown,
  ChevronRight,
  Download,
  FileSearch,
  History,
  Layers,
  LoaderCircle,
  MessageSquare,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Square,
} from "lucide-react";
import { api, graphReady, pending } from "./api/client";
import type { Citation, ExportFormat, GraphNode } from "./api/contract";
import { count, time } from "./format";
import {
  ErrorNotice,
  IconButton,
  Loading,
  Status,
  Warnings,
} from "./components/Common";
import { Inspector, matchesCitation } from "./components/Inspector";
import { Questions } from "./components/Questions";
import { GraphCanvas } from "./graph/GraphCanvas";

type Crumb = { id: string; label: string };
type Panel = "graph" | "details" | "questions";
const exportFormats: { format: ExportFormat; label: string }[] = [
  { format: "json", label: "Architecture JSON" },
  { format: "zip", label: "Complete ZIP" },
  { format: "mermaid", label: "Mermaid" },
  { format: "markdown", label: "Markdown" },
  { format: "diagnostics", label: "Diagnostics" },
  { format: "audit", label: "Audit" },
];

export function App() {
  const client = useQueryClient();
  const [analysisId, setAnalysisId] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("analysis"),
  );
  const [model, setModel] = useState("");
  const [path, setPath] = useState<Crumb[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [panel, setPanel] = useState<Panel>("graph");
  const [sideOpen, setSideOpen] = useState(false);
  const [rightTab, setRightTab] = useState<"details" | "questions">("details");
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportError, setExportError] = useState<Error | null>(null);
  const exportMenu = useRef<HTMLDetailsElement>(null);
  const activeAnalysis = useRef(analysisId);
  activeAnalysis.current = analysisId;
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 30_000,
  });
  const serviceHealth = health.error ? undefined : health.data;
  const analysisAvailable =
    serviceHealth?.status === "ok" &&
    serviceHealth.checks.database &&
    serviceHealth.checks.storage &&
    serviceHealth.checks.worker &&
    serviceHealth.analysis_available;
  const restoredRecent = useRef(false);
  const examples = useQuery({
    queryKey: ["examples"],
    queryFn: ({ signal }) => api.examples(signal),
  });
  const recent = useQuery({
    queryKey: ["analyses"],
    queryFn: ({ signal }) => api.analyses(signal),
    refetchInterval: (query) =>
      query.state.data?.analyses.some((job) => pending(job.status))
        ? 3000
        : false,
  });
  const analysis = useQuery({
    queryKey: ["analysis", analysisId],
    queryFn: ({ signal }) => api.analysis(analysisId!, signal),
    enabled: Boolean(analysisId),
    refetchInterval: (query) =>
      pending(query.state.data?.status) ? 1500 : false,
  });
  const detail = analysis.data;
  const graphId = path.at(-1)?.id ?? detail?.root_graph_id ?? null;
  const ready = graphReady(detail?.status) && !analysis.error;
  const graph = useQuery({
    queryKey: ["graph", analysisId, graphId],
    queryFn: ({ signal }) => api.graph(analysisId!, graphId!, signal),
    enabled: Boolean(analysisId && graphId && ready),
  });
  const selectedNode =
    graph.data?.nodes.find((node) => node.id === selectedNodeId) ?? null;

  const choose = useCallback((id: string) => {
    setAnalysisId(id);
    setPath([]);
    setSelectedNodeId(null);
    setCitation(null);
    setSideOpen(false);
    setPanel("graph");
    setExportError(null);
    const url = new URL(window.location.href);
    url.searchParams.set("analysis", id);
    window.history.replaceState(null, "", url);
  }, []);
  useEffect(() => {
    if (restoredRecent.current || analysisId || !recent.data) return;
    restoredRecent.current = true;
    const ownJob = recent.data.analyses
      .filter((job) => !job.prepared)
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))[0];
    if (ownJob) choose(ownJob.id);
  }, [analysisId, recent.data, choose]);
  const create = useMutation({
    mutationFn: api.analyze,
    onSuccess: (result) => {
      client.setQueryData(["analysis", result.id], result);
      choose(result.id);
      void client.invalidateQueries({ queryKey: ["analyses"] });
    },
  });
  const cancel = useMutation({
    mutationFn: api.cancel,
    onSuccess: (result) => {
      client.setQueryData(["analysis", result.id], result);
      void client.invalidateQueries({ queryKey: ["analyses"] });
    },
  });

  const openGraph = useCallback(
    (node: GraphNode) => {
      if (!node.child_graph_id || !detail?.root_graph_id) return;
      setPath((current) => [
        ...(current.length
          ? current
          : [
              {
                id: detail.root_graph_id!,
                label: detail.model_id.split("/").at(-1)!,
              },
            ]),
        { id: node.child_graph_id!, label: node.label },
      ]);
      setSelectedNodeId(null);
      setCitation(null);
      setPanel("graph");
    },
    [detail?.root_graph_id, detail?.model_id],
  );

  async function jumpCitation(target: Citation) {
    if (!analysisId || !detail?.root_graph_id)
      throw new Error("No model graph is available.");
    const jobId = analysisId;
    const root: Crumb = {
      id: detail.root_graph_id,
      label: detail.model_id.split("/").at(-1)!,
    };
    if (target.source === "stored_ir" && target.locator === "metadata") {
      setSelectedNodeId(null);
      setCitation(target);
      setRightTab("details");
      setPanel("details");
      return;
    }
    const queue: { id: string; path: Crumb[] }[] = [
      { id: root.id, path: [root] },
    ];
    const visited = new Set<string>();
    while (queue.length && visited.size < 100) {
      const next = queue.shift()!;
      if (visited.has(next.id)) continue;
      visited.add(next.id);
      const data = await client.fetchQuery({
        queryKey: ["graph", jobId, next.id],
        queryFn: ({ signal }) => api.graph(jobId, next.id, signal),
      });
      if (activeAnalysis.current !== jobId) return;
      const found =
        data.nodes.find(
          (node) =>
            target.source === "stored_ir" &&
            target.locator === `node:${data.id}/${node.id}`,
        ) ??
        data.nodes.find((node) =>
          node.evidence.some((item) => matchesCitation(item, target)),
        ) ??
        data.nodes.find((node) => {
          const cached = client.getQueriesData<{ evidence?: Citation[] }>({
            queryKey: ["node", jobId, next.id, node.id],
          });
          return cached.some(([, item]) =>
            item?.evidence?.some((evidence) =>
              matchesCitation(evidence, target),
            ),
          );
        });
      if (found) {
        setPath(next.path);
        setSelectedNodeId(found.id);
        setCitation(target);
        setRightTab("details");
        setPanel("details");
        return;
      }
      for (const node of data.nodes)
        if (node.child_graph_id)
          queue.push({
            id: node.child_graph_id,
            path: [
              ...next.path,
              { id: node.child_graph_id, label: node.label },
            ],
          });
    }
    throw new Error(
      "No component association was found for this citation. Its source and locator remain available in the answer.",
    );
  }

  async function download(format: ExportFormat) {
    if (!analysisId) return;
    exportMenu.current?.removeAttribute("open");
    setExporting(format);
    setExportError(null);
    try {
      await api.export(analysisId, format);
    } catch (error) {
      setExportError(
        error instanceof Error ? error : new Error("Export failed."),
      );
    } finally {
      setExporting(null);
    }
  }

  const breadcrumb = path.length
    ? path
    : detail?.root_graph_id
      ? [
          {
            id: detail.root_graph_id,
            label: detail.model_id.split("/").at(-1)!,
          },
        ]
      : [];
  const emptyGraph = !analysisId
    ? "No analysis selected"
    : pending(detail?.status)
      ? "Analyzing model"
      : detail?.status === "partial"
        ? "Partial analysis has no graph"
        : detail?.status === "completed"
          ? "No graph returned"
          : detail?.status === "unsupported"
            ? "Unsupported architecture"
            : detail?.status === "cancelled"
              ? "Analysis cancelled"
              : detail?.status === "timed_out"
                ? "Analysis timed out"
                : "Architecture unavailable";
  const inspect = (node: GraphNode) => {
    setSelectedNodeId(node.id);
    setCitation(null);
    setRightTab("details");
  };

  return (
    <div className={`app panel-${panel} ${sideOpen ? "sidebar-open" : ""}`}>
      <header className="app-header">
        <div className="brand">
          <Network size={23} strokeWidth={1.6} />
          <span>Model Anatomy</span>
          <span className="brand-version">IR 1.1</span>
        </div>
        <div
          className="service-state"
          title={
            health.error?.message ??
            serviceHealth?.reason ??
            (serviceHealth
              ? `Storage: ${serviceHealth.storage_backend}; database: ${serviceHealth.checks.database ? "ok" : "unavailable"}; storage: ${serviceHealth.checks.storage ? "ok" : "unavailable"}; worker: ${serviceHealth.checks.worker ? "ok" : "unavailable"}`
              : "Health unknown")
          }
        >
          <span
            className={`service-dot ${serviceHealth?.status === "ok" ? "online" : serviceHealth?.status === "degraded" ? "degraded" : ""}`}
          />
          {serviceHealth
            ? serviceHealth.status === "degraded"
              ? "API degraded"
              : `API ${serviceHealth.api_version}`
            : health.isPending
              ? "Connecting"
              : "API unavailable"}
        </div>
        <IconButton
          label={sideOpen ? "Close recent analyses" : "Open recent analyses"}
          onClick={() => setSideOpen((value) => !value)}
        >
          <History size={17} />
        </IconButton>
      </header>
      <div className="input-band">
        <form
          className="model-form"
          onSubmit={(event) => {
            event.preventDefault();
            if (model.trim() && !create.isPending && analysisAvailable)
              create.mutate(model.trim());
          }}
        >
          <label htmlFor="model-input">Hugging Face model</label>
          <div className="model-field">
            <Box size={16} />
            <input
              id="model-input"
              placeholder="namespace/model or Hugging Face URL"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              required
              maxLength={512}
              autoComplete="off"
            />
          </div>
          <button
            className="analyze-button"
            type="submit"
            disabled={create.isPending || !model.trim() || !analysisAvailable}
            title={
              !analysisAvailable
                ? "Analysis unavailable until service health is confirmed"
                : "Analyze Model"
            }
          >
            {create.isPending ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Activity size={15} />
            )}
            <span>Analyze Model</span>
          </button>
        </form>
        {(health.error ||
          serviceHealth?.analysis_available === false ||
          serviceHealth?.status === "degraded") && (
          <div className="availability-note">
            {serviceHealth?.reason ??
              health.error?.message ??
              "Analysis unavailable."}{" "}
            New analyses are unavailable.
          </div>
        )}
        {create.error && <ErrorNotice error={create.error} />}
      </div>
      <div className="workspace">
        {sideOpen && (
          <button
            className="sidebar-backdrop"
            aria-label="Close recent analyses"
            onClick={() => setSideOpen(false)}
          />
        )}
        <aside className="sidebar">
          <div className="sidebar-heading">
            <h2>Workspace</h2>
            <IconButton
              label="Close recent analyses"
              onClick={() => setSideOpen(false)}
            >
              <PanelLeftClose size={15} />
            </IconButton>
          </div>
          <section className="sidebar-section">
            <h3>Prepared analyses</h3>
            {examples.isPending && <Loading text="Loading examples" />}
            {examples.error && (
              <ErrorNotice
                error={examples.error}
                retry={() => void examples.refetch()}
              />
            )}
            {examples.data?.examples.map((example) => (
              <button
                className={`analysis-link ${example.id === analysisId ? "active" : ""}`}
                key={example.id}
                onClick={() => choose(example.id)}
              >
                <span className="analysis-icon">
                  <Layers size={17} />
                </span>
                <span className="analysis-link-copy">
                  <strong>{example.title}</strong>
                  <span className="prepared-label">Prepared</span>
                  <span>{example.architecture ?? "Unknown architecture"}</span>
                  <small>
                    {count(example.parameter_count, true)} parameters
                  </small>
                </span>
                <ChevronRight size={14} />
              </button>
            ))}
          </section>
          <section className="sidebar-section recent-section">
            <div className="section-heading">
              <History size={14} />
              <h3>Recent jobs</h3>
            </div>
            {recent.isPending && <Loading text="Loading jobs" />}
            {recent.error && (
              <ErrorNotice
                error={recent.error}
                retry={() => void recent.refetch()}
              />
            )}
            {recent.data?.analyses.length === 0 && (
              <p className="empty-copy">No recent jobs</p>
            )}
            {recent.data?.analyses.map((job) => (
              <button
                className={`recent-link ${job.id === analysisId ? "active" : ""}`}
                key={job.id}
                onClick={() => choose(job.id)}
              >
                <strong title={job.model_id}>{job.model_id}</strong>
                <span>
                  <Status status={job.status} />
                  <small>{time(job.created_at)}</small>
                </span>
              </button>
            ))}
          </section>
          <div className="sidebar-footer">
            <Braces size={14} />
            <span>Artifact-backed architecture</span>
          </div>
        </aside>
        <main className="main-workspace">
          <div className="model-overview">
            <div className="model-title-row">
              <div className="model-title">
                <IconButton
                  label="Open recent analyses"
                  onClick={() => setSideOpen(true)}
                >
                  <PanelLeftOpen size={16} />
                </IconButton>
                <div>
                  <span className="eyebrow">
                    {detail
                      ? (detail.architecture ?? "Architecture unknown")
                      : "Workspace"}
                  </span>
                  <h1>{detail?.model_id ?? "Model Anatomy"}</h1>
                </div>
              </div>
              <div className="model-actions">
                {detail && <Status status={detail.status} />}
                {detail?.prepared && (
                  <span className="prepared-label">Prepared</span>
                )}
                {detail?.cached && (
                  <span className="prepared-label">Cached</span>
                )}
                {detail && pending(detail.status) && (
                  <IconButton
                    label="Cancel pending job"
                    disabled={cancel.isPending}
                    onClick={() => cancel.mutate(detail.id)}
                  >
                    {cancel.isPending ? (
                      <LoaderCircle size={15} className="spin" />
                    ) : (
                      <Square size={14} />
                    )}
                  </IconButton>
                )}
                <details className="export-menu" ref={exportMenu}>
                  <summary
                    aria-label="Export analysis"
                    title="Export analysis"
                    aria-disabled={!ready || exporting !== null}
                    onClick={(event) => {
                      if (!ready || exporting) event.preventDefault();
                    }}
                  >
                    {exporting ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : (
                      <Download size={16} />
                    )}
                    <ChevronDown size={12} />
                  </summary>
                  <div className="export-options">
                    {exportFormats.map((item) => (
                      <button
                        key={item.format}
                        disabled={!ready || exporting !== null}
                        onClick={() => void download(item.format)}
                      >
                        <Download size={13} />
                        {item.label}
                      </button>
                    ))}
                  </div>
                </details>
              </div>
            </div>
            {detail && (
              <div className="model-metrics">
                <div>
                  <span>Parameters</span>
                  <strong title={count(detail?.parameter_count)}>
                    {count(detail?.parameter_count, true)}
                  </strong>
                </div>
                {(["hidden", "heads", "layers", "vocabulary"] as const).map(
                  (key) => (
                    <div key={key}>
                      <span>
                        {key === "hidden"
                          ? "Hidden size"
                          : key === "vocabulary"
                            ? "Vocabulary"
                            : key[0].toUpperCase() + key.slice(1)}
                      </span>
                      <strong>
                        {typeof detail?.dimensions[key] === "number"
                          ? count(detail.dimensions[key] as number)
                          : (detail?.dimensions[key] ?? "Unknown")}
                      </strong>
                    </div>
                  ),
                )}
              </div>
            )}
            {detail?.revision && (
              <div className="revision" title={detail.revision}>
                <span>Revision</span>
                <code>{detail.revision}</code>
              </div>
            )}
            {detail && pending(detail.status) && (
              <div className="job-progress" role="status">
                <LoaderCircle size={14} className="spin" />
                <span>{detail.stage || detail.status}</span>
                <span className="indeterminate-track" />
              </div>
            )}
            {detail?.error && (
              <div className="analysis-error" role="alert">
                <strong>{detail.error.code}</strong>
                <span>{detail.error.message}</span>
              </div>
            )}
            {detail?.status === "partial" && (
              <div className="partial-notice">
                Partial analysis
                {detail.stage && detail.stage !== detail.status
                  ? ` / ${detail.stage}`
                  : ""}
              </div>
            )}
            {detail && <Warnings warnings={detail.warnings} />}
            {cancel.error && <ErrorNotice error={cancel.error} />}
            {exportError && <ErrorNotice error={exportError} />}
          </div>
          <div className="graph-toolbar">
            <nav className="breadcrumbs" aria-label="Graph breadcrumb">
              {breadcrumb.map((crumb, index) => (
                <span key={`${crumb.id}:${index}`}>
                  {index > 0 && <ChevronRight size={13} />}
                  <button
                    aria-current={
                      index === breadcrumb.length - 1 ? "page" : undefined
                    }
                    onClick={() => {
                      setPath(breadcrumb.slice(0, index + 1));
                      setSelectedNodeId(null);
                      setCitation(null);
                    }}
                  >
                    {crumb.label}
                  </button>
                </span>
              ))}
            </nav>
            {path.length > 1 && (
              <button
                className="collapse-button"
                onClick={() => {
                  setPath((value) => value.slice(0, -1));
                  setSelectedNodeId(null);
                  setCitation(null);
                }}
              >
                <Layers size={14} />
                Collapse
                <ArrowRight size={12} />
              </button>
            )}
          </div>
          <div className="canvas-area">
            {analysis.error ? (
              <div className="canvas-state">
                <ErrorNotice
                  error={analysis.error}
                  retry={() => void analysis.refetch()}
                />
              </div>
            ) : analysisId && analysis.isPending ? (
              <div className="canvas-state">
                <Loading text="Loading analysis" />
              </div>
            ) : ready && graphId ? (
              graph.error ? (
                <div className="canvas-state">
                  <ErrorNotice
                    error={graph.error}
                    retry={() => void graph.refetch()}
                  />
                </div>
              ) : graph.data ? (
                <GraphCanvas
                  key={`${analysisId}:${graphId}`}
                  graph={graph.data}
                  selectedNode={selectedNodeId}
                  select={inspect}
                  open={openGraph}
                />
              ) : (
                <div className="canvas-state">
                  <Loading text="Loading architecture graph" />
                </div>
              )
            ) : (
              <div className="canvas-state">
                <Network size={32} strokeWidth={1.4} />
                <strong>{emptyGraph}</strong>
                {detail?.stage && <p>{detail.stage}</p>}
              </div>
            )}
          </div>
          <footer className="graph-footer">
            <span>
              <span className="legend-line" />
              Dataflow
            </span>
            <span>
              <span className="legend-line relation" />
              Structural relation
            </span>
            <span className="footer-id" title={graphId ?? ""}>
              {graphId ?? "No graph"}
            </span>
          </footer>
        </main>
        <aside className={`right-panel right-${rightTab}`}>
          <div
            className="right-tabs"
            role="tablist"
            aria-label="Analysis panels"
          >
            <button
              role="tab"
              aria-selected={rightTab === "details"}
              onClick={() => setRightTab("details")}
            >
              <FileSearch size={15} />
              Details
            </button>
            <button
              role="tab"
              aria-selected={rightTab === "questions"}
              onClick={() => setRightTab("questions")}
            >
              <MessageSquare size={15} />
              Questions
            </button>
          </div>
          <div className="detail-panel">
            {detail && analysisId && graphId && ready ? (
              <Inspector
                key={`${analysisId}:${graphId}:${selectedNodeId}`}
                analysis={detail}
                analysisId={analysisId}
                graphId={graphId}
                node={selectedNode}
                openGraph={openGraph}
                citation={citation}
              />
            ) : (
              <div className="panel-empty">
                <FileSearch size={28} />
                <strong>Component details</strong>
                <p>
                  {analysisId
                    ? "Architecture unavailable"
                    : "No analysis selected"}
                </p>
              </div>
            )}
          </div>
          <div className="questions-panel">
            {analysisId ? (
              <Questions
                key={analysisId}
                analysisId={analysisId}
                graphId={graphId ?? undefined}
                nodeId={selectedNodeId ?? undefined}
                nodeLabel={selectedNode?.label}
                ready={ready}
                available={serviceHealth?.ai_available === true}
                jump={jumpCitation}
              />
            ) : (
              <div className="panel-empty">
                <MessageSquare size={28} />
                <strong>Model questions</strong>
                <p>No analysis selected</p>
              </div>
            )}
          </div>
        </aside>
      </div>
      <nav className="mobile-tabs" aria-label="Workspace views">
        {(
          [
            { id: "graph", icon: Network, label: "Graph" },
            { id: "details", icon: FileSearch, label: "Details" },
            { id: "questions", icon: MessageSquare, label: "Questions" },
          ] as const
        ).map((item) => (
          <button
            key={item.id}
            aria-current={panel === item.id ? "page" : undefined}
            onClick={() => {
              setPanel(item.id);
              if (item.id !== "graph") setRightTab(item.id);
            }}
          >
            <item.icon size={17} />
            {item.label}
          </button>
        ))}
      </nav>
    </div>
  );
}
