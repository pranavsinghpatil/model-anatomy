import { useEffect, useRef, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronDown,
  Download,
  Plus,
  Network,
  ChevronRight,
  LayoutDashboard,
  Files,
  ShieldCheck,
  ExternalLink,
  Compass,
  Cpu,
  Check,
  Copy,
} from "lucide-react";
import { analysisHref, api, graphReady, pending } from "../api/client";
import type { Citation, ExportFormat } from "../api/contract";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { ModelAnatomyMark } from "../components/Logo";
import { Menu, MenuItem, ThemeControl } from "../components/ui";
import { Overview, Report } from "../components/AnalysisSummary";
import { ArtifactInspection } from "../components/ArtifactInspection";
import { ArchitectureWorkspace } from "../components/ArchitectureWorkspace";
import { Progress, Recovery } from "../components/Progress";
import { matchesCitation } from "../components/Inspector";
import { useRecent, useService } from "../state/queries";
import { useWorkspaceSession, type Crumb } from "../state/workspace";
import { count, dimensionCount } from "../format";
import { DotPattern } from "../components/vendor/dot-pattern";
import {
  FluidHoverHighlight,
  type HighlightRect,
} from "../components/vendor/fluid-hover-highlight";
import "./workspace.css";
const views = ["architecture", "overview", "artifacts", "report"] as const;
const formats: { format: ExportFormat; label: string }[] = [
  { format: "json", label: "Report JSON" },
  { format: "zip", label: "Complete ZIP" },
  { format: "mermaid", label: "Mermaid" },
  { format: "markdown", label: "Markdown" },
  { format: "diagnostics", label: "Diagnostics" },
  { format: "audit", label: "Audit" },
];
export default function WorkspaceRoute() {
  const { id } = useParams();
  return <Workspace key={id} id={id!} />;
}
function Workspace({ id }: { id: string }) {
  const [search, setSearch] = useSearchParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const { service, available } = useService();
  const recent = useRecent();
  const [state, setState] = useWorkspaceSession(id);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [copiedModelId, setCopiedModelId] = useState(false);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportError, setExportError] = useState<Error | null>(null);
  const viewNav = useRef<HTMLElement>(null);
  const [highlight, setHighlight] = useState<HighlightRect | null>(null);
  function highlightView(target: HTMLElement) {
    const item = target.getBoundingClientRect();
    const nav = viewNav.current!.getBoundingClientRect();
    const scale =
      Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const left = Math.max(0, (item.left - nav.left) / scale);
    setHighlight({
      left,
      top: 4,
      width: Math.min(item.width / scale, viewNav.current!.clientWidth - left),
      height: viewNav.current!.clientHeight - 8,
    });
  }
  const active = useRef(true);
  const submissionPath = useRef(window.location.pathname);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const analysis = useQuery({
    queryKey: ["analysis", id],
    queryFn: ({ signal }) => api.analysis(id, signal),
    refetchInterval: (q) => (pending(q.state.data?.status) ? 1500 : false),
  });
  const detail = analysis.data;
  const artifact = detail?.status === "inspected";
  const sourceTopology =
    detail?.quality?.mode === "publisher_reference_topology";
  const previousStatus = useRef(detail?.status);
  const ready = graphReady(detail?.status) && !analysis.error;
  const viewParam = search.get("view");
  const awaiting = search.get("awaiting") === "1";
  const handoff =
    awaiting && detail && !pending(detail.status)
      ? detail.status === "inspected"
        ? "artifacts"
        : graphReady(detail.status)
          ? "architecture"
          : null
      : null;
  const view =
    handoff ??
    views.find((v) => v === viewParam) ??
    (artifact ? "artifacts" : "architecture");
  useEffect(() => {
    if (!detail) return;
    const wasPending = pending(previousStatus.current);
    previousStatus.current = detail.status;
    if (wasPending && !pending(detail.status))
      void client.invalidateQueries({ queryKey: ["catalog"] });
    if (pending(detail.status)) return;
    if (awaiting || wasPending) {
      const next =
        detail.status === "inspected"
          ? "artifacts"
          : graphReady(detail.status)
            ? "architecture"
            : (views.find((item) => item === viewParam) ?? "architecture");
      const params = new URLSearchParams(search);
      params.delete("awaiting");
      params.set("view", next);
      if (params.toString() !== search.toString())
        setSearch(params, { replace: true });
      return;
    }
    if (!views.some((v) => v === viewParam))
      setSearch({ view }, { replace: true });
  }, [detail, awaiting, viewParam, view, search, setSearch, client]);
  const graphId = state.path.at(-1)?.id ?? detail?.root_graph_id;
  const graph = useQuery({
    queryKey: ["graph", id, graphId],
    queryFn: ({ signal }) => api.graph(id, graphId!, signal),
    enabled: ready && Boolean(graphId) && view === "architecture",
  });
  const cancel = useMutation({
    mutationFn: api.cancel,
    onSuccess: (job) => {
      client.setQueryData(["analysis", id], job);
      void client.invalidateQueries({ queryKey: ["analyses"] });
    },
  });
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
      void navigate(analysisHref(job));
    },
  });
  async function download(format: ExportFormat) {
    setExporting(format);
    setExportError(null);
    try {
      await api.export(id, format);
    } catch (error) {
      if (active.current)
        setExportError(
          error instanceof Error ? error : new Error("Export failed."),
        );
    } finally {
      if (active.current) setExporting(null);
    }
  }
  async function jump(target: Citation) {
    if (!detail?.root_graph_id) throw new Error("No model graph is available.");
    if (target.source === "stored_ir" && target.locator === "metadata") {
      setCitation(target);
      setState((s) => ({
        ...s,
        selected: { ...s.selected, [graphId!]: null },
        inspectorOpen: true,
        questions: false,
      }));
      return;
    }
    const root: Crumb = {
      id: detail.root_graph_id,
      label: detail.model_id.split("/").at(-1)!,
    };
    const queue = [{ id: root.id, path: [root] }];
    const visited = new Set<string>();
    while (queue.length && visited.size < 100) {
      const next = queue.shift()!;
      if (visited.has(next.id)) continue;
      visited.add(next.id);
      const data = await client.fetchQuery({
        queryKey: ["graph", id, next.id],
        queryFn: ({ signal }) => api.graph(id, next.id, signal),
      });
      if (!active.current) return;
      const found =
        data.nodes.find(
          (node) =>
            target.source === "stored_ir" &&
            target.locator === `node:${data.id}/${node.id}`,
        ) ||
        data.nodes.find((node) =>
          node.evidence.some((item) => matchesCitation(item, target)),
        ) ||
        data.nodes.find((node) =>
          client
            .getQueriesData<{ evidence?: Citation[] }>({
              queryKey: ["node", id, next.id, node.id],
            })
            .some(([, item]) =>
              item?.evidence?.some((e) => matchesCitation(e, target)),
            ),
        );
      if (found) {
        setCitation(target);
        setState((s) => ({
          ...s,
          path: next.path,
          selected: { ...s.selected, [data.id]: found.id },
          inspectorOpen: true,
          questions: false,
          inspectorTab: "evidence",
        }));
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
      "No component association was found. The citation’s source and locator remain available in the answer.",
    );
  }
  return (
    <main id="main" tabIndex={-1} className="workspace-shell">
      <header className="workspace-header">
        <Link
          className="workspace-home"
          to="/"
          aria-label="Back to start"
          title="Model Anatomy · All analyses"
        >
          <span className="brand-gem workspace-brand-gem" aria-hidden="true">
            <ModelAnatomyMark size={18} />
          </span>
          <span className="entry-brand-name">Model Anatomy</span>
        </Link>
        <div className="model-switcher">
          <Menu
            label="Switch model"
            trigger={
              <>
                <span className="model-name" title={detail?.model_id}>
                  {detail?.model_id ?? "Loading analysis"}
                </span>
                <ChevronDown size={14} />
              </>
            }
          >
            {recent.data?.analyses
              .map((stored) => (stored.id === detail?.id ? detail : stored))
              .map((job) => (
                <MenuItem
                  key={job.id}
                  onSelect={() => void navigate(analysisHref(job))}
                >
                  {job.model_id}
                  <Status status={job.status} />
                </MenuItem>
              ))}
            <MenuItem onSelect={() => void navigate("/app")}>
              New analysis
            </MenuItem>
          </Menu>
        </div>
        {detail?.model_id && (
          <div className="workspace-model-tools" aria-label="Model quick tools">
            <button
              type="button"
              className="button button-ghost button-icon-only workspace-copy-btn"
              onClick={() => {
                void navigator.clipboard.writeText(detail.model_id);
                setCopiedModelId(true);
                setTimeout(() => setCopiedModelId(false), 2000);
              }}
              title={copiedModelId ? "Copied model ID" : "Copy model ID"}
              aria-label="Copy model ID"
            >
              {copiedModelId ? (
                <Check size={13} aria-hidden="true" />
              ) : (
                <Copy size={13} aria-hidden="true" />
              )}
            </button>
            {detail.model_id.includes("/") && (
              <a
                href={`https://huggingface.co/${detail.model_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="button button-ghost button-icon-only workspace-hf-link"
                title="Open repository on Hugging Face"
                aria-label="Open repository on Hugging Face"
              >
                <ExternalLink size={13} aria-hidden="true" />
              </a>
            )}
            {detail.architecture && (
              <span
                className="workspace-arch-pill"
                title={`Detected architecture: ${detail.architecture}`}
              >
                <Cpu size={12} aria-hidden="true" />
                <span>{detail.architecture}</span>
              </span>
            )}
          </div>
        )}
        {detail && <Status status={detail.status} />}
        <div className="workspace-header-actions">
          <Link
            className="button button-ghost workspace-catalog-btn"
            to="/catalog"
            aria-label="Browse catalog"
            title="Browse Model Catalog"
          >
            <Compass size={15} aria-hidden="true" />
            <span className="wide-label">Catalog</span>
          </Link>
          <ThemeControl />
          <Link
            className="button button-secondary"
            to="/app"
            aria-label="New analysis"
          >
            <Plus size={16} />
            <span className="wide-label">New analysis</span>
          </Link>
          <Menu
            label="Export analysis"
            disabled={(!ready && !artifact) || exporting !== null}
            trigger={
              <>
                <Download size={16} />
                <span className="wide-label">
                  {exporting ? "Exporting…" : "Export"}
                </span>
              </>
            }
          >
            {formats
              .filter((f) => !artifact || f.format !== "mermaid")
              .map((f) => (
                <MenuItem
                  key={f.format}
                  onSelect={() => void download(f.format)}
                >
                  {f.label}
                </MenuItem>
              ))}
          </Menu>
        </div>
      </header>
      <div className="workspace-viewbar">
        <nav
          className="primary-views"
          aria-label="Workspace views"
          ref={viewNav}
          onPointerLeave={() => setHighlight(null)}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget))
              setHighlight(null);
          }}
        >
          <FluidHoverHighlight rect={highlight} />
          {views.map((item) => {
            const Icon =
              item === "architecture"
                ? Network
                : item === "overview"
                  ? LayoutDashboard
                  : item === "artifacts"
                    ? Files
                    : ShieldCheck;
            return (
              <Link
                key={item}
                onPointerEnter={(event) => highlightView(event.currentTarget)}
                onFocus={(event) => highlightView(event.currentTarget)}
                to={`?view=${item}`}
                aria-current={view === item ? "page" : undefined}
                aria-disabled={!ready && !artifact ? true : undefined}
                tabIndex={!ready && !artifact ? -1 : undefined}
                onClick={(event) => {
                  if (!ready && !artifact) event.preventDefault();
                }}
                title={
                  item === "architecture" && artifact
                    ? "No architecture was reconstructed; see available artifacts"
                    : undefined
                }
              >
                <Icon size={14} className="view-tab-icon" aria-hidden="true" />
                <span>{item[0].toUpperCase() + item.slice(1)}</span>
              </Link>
            );
          })}
        </nav>
        <div className="workspace-view-options">
          {detail && ready && !artifact && (
            <>
              {detail.status === "partial" && (
                <Link to="?view=report" className="viewbar-findings">
                  View findings
                  <ChevronRight size={13} aria-hidden="true" />
                </Link>
              )}
              <div
                className="workspace-telemetry-ribbon"
                aria-label="Model telemetry summary"
              >
                <div className="telemetry-chip" title="Total Parameters">
                  <span className="telemetry-label">Params</span>
                  <span className="telemetry-val">
                    {count(detail.parameter_count, true)}
                  </span>
                </div>
                {detail.dimensions.layers != null && (
                  <div className="telemetry-chip" title="Layer count">
                    <span className="telemetry-label">Layers</span>
                    <span className="telemetry-val">
                      {dimensionCount(detail.dimensions.layers)}
                    </span>
                  </div>
                )}
                {detail.dimensions.hidden != null && (
                  <div className="telemetry-chip" title="Hidden dimension">
                    <span className="telemetry-label">Hidden</span>
                    <span className="telemetry-val">
                      {dimensionCount(detail.dimensions.hidden)}
                    </span>
                  </div>
                )}
                {detail.dimensions.heads != null && (
                  <div className="telemetry-chip" title="Attention heads">
                    <span className="telemetry-label">Heads</span>
                    <span className="telemetry-val">
                      {dimensionCount(detail.dimensions.heads)}
                    </span>
                  </div>
                )}
                {detail.revision && (
                  <div
                    className="telemetry-chip telemetry-revision"
                    title={`Verified revision: ${detail.revision}`}
                  >
                    <span className="telemetry-label">Rev</span>
                    <span className="telemetry-val font-mono">
                      {detail.revision.slice(0, 7)}
                    </span>
                  </div>
                )}
              </div>
              <details
                key={view}
                className="model-facts"
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    event.currentTarget.open = false;
                    event.currentTarget.querySelector("summary")?.focus();
                  }
                }}
              >
                <summary>
                  Model details <ChevronDown size={13} aria-hidden="true" />
                </summary>
                <div
                  className="model-context"
                  role="region"
                  aria-label="Model context"
                >
                  <strong className="context-architecture">
                    {detail.architecture ?? "Architecture unknown"}
                  </strong>
                  <dl>
                    <div>
                      <dt>Parameters</dt>
                      <dd>{count(detail.parameter_count, true)}</dd>
                    </div>
                    {detail.dimensions.layers != null && (
                      <div>
                        <dt>Layers</dt>
                        <dd>{dimensionCount(detail.dimensions.layers)}</dd>
                      </div>
                    )}
                    {detail.dimensions.hidden != null && (
                      <div>
                        <dt>Hidden size</dt>
                        <dd>{dimensionCount(detail.dimensions.hidden)}</dd>
                      </div>
                    )}
                  </dl>
                  <Link to="?view=report" className="context-report">
                    Inspection findings
                    <ChevronRight size={13} aria-hidden="true" />
                  </Link>
                </div>
              </details>
            </>
          )}
        </div>
      </div>
      <div className="workbench-body">
        <div className="workbench-pattern" aria-hidden="true">
          <DotPattern width={22} height={22} cr={0.8} />
        </div>
        <div className="workbench-main">
          {exportError && <ErrorNotice error={exportError} />}{" "}
          {cancel.error && <ErrorNotice error={cancel.error} />}{" "}
          {create.error && <ErrorNotice error={create.error} />}
          {analysis.error ? (
            <div className="page-state">
              <ErrorNotice
                error={analysis.error}
                retry={() => void analysis.refetch()}
              />
              <Link className="button button-primary" to="/app">
                Start a new inspection
              </Link>
            </div>
          ) : !detail ? (
            <div className="page-state">
              <Loading text="Loading analysis" />
            </div>
          ) : pending(detail.status) ? (
            <Progress
              detail={detail}
              cancel={() => cancel.mutate(id)}
              cancelling={cancel.isPending}
            />
          ) : !ready && !artifact ? (
            <Recovery detail={detail} />
          ) : (
            <div className={`workspace-content view-${view}`}>
              {view === "architecture" && (
                <h1 className="sr-only">{detail.model_id} architecture</h1>
              )}
              {view === "overview" && <Overview detail={detail} />}
              {view === "report" && <Report detail={detail} />}
              {view === "artifacts" && (
                <div className="artifacts-page">
                  <ArtifactInspection
                    key={id}
                    architecture={!artifact}
                    analysisId={id}
                    modelId={detail.model_id}
                    revision={detail.revision}
                    summary={detail.inspection}
                    onInspectFile={(url) => create.mutate(url)}
                    inspectionDisabled={!available || create.isPending}
                  />
                </div>
              )}
              {view === "architecture" &&
                (artifact ? (
                  <div className="page-state">
                    <h1>No architecture graph</h1>
                    <p>
                      {detail.inspection?.message ??
                        "This result verified the repository artifacts at the pinned revision. No architecture graph was reconstructed."}
                    </p>
                    <p>
                      {detail.inspection
                        ? `Verified ${count(detail.inspection.file_count)} repository files and ${count(detail.inspection.tensor_count)} tensor headers${detail.revision ? ` at revision ${detail.revision.slice(0, 12)}` : ""}.`
                        : "Open the artifact inventory for the files and tensor headers that were verified."}
                    </p>
                    {detail.inspection?.reason_code && (
                      <code>{detail.inspection.reason_code}</code>
                    )}
                    <Link
                      className="button button-primary"
                      to="?view=artifacts"
                    >
                      Explore artifacts
                    </Link>
                  </div>
                ) : graph.error ? (
                  <div className="page-state">
                    <ErrorNotice
                      error={graph.error}
                      retry={() => void graph.refetch()}
                    />
                  </div>
                ) : graph.data ? (
                  <>
                    {sourceTopology && (
                      <div className="source-topology-notice" role="note">
                        <strong>Publisher reference topology · partial</strong>
                        <span>
                          Components come from reviewed source at this revision.
                          Checkpoint tensors, parameter totals, and runtime
                          dataflow are not verified. Dashed links show reference
                          order only.
                        </span>
                      </div>
                    )}
                    <ArchitectureWorkspace
                      analysis={detail}
                      graph={graph.data}
                      state={state}
                      setState={setState}
                      citation={citation}
                      onClearCitation={() => {
                        if (citation)
                          setState((s) => ({ ...s, inspectorTab: "module" }));
                        setCitation(null);
                      }}
                      onCitation={jump}
                      ai={service?.ai_available === true}
                    />
                  </>
                ) : graphId ? (
                  <div className="page-state">
                    <Loading text="Loading architecture graph" />
                  </div>
                ) : (
                  <Recovery detail={detail} />
                ))}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
