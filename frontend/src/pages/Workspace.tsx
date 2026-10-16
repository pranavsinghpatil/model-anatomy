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
  LayoutDashboard,
  Files,
  FileCheck2,
  ChevronRight,
} from "lucide-react";
import { api, graphReady, pending } from "../api/client";
import type { Citation, ExportFormat } from "../api/contract";
import { ErrorNotice, Loading, Status } from "../components/Common";
import { Menu, MenuItem, ModeControl, ThemeControl } from "../components/ui";
import { Overview, Report } from "../components/AnalysisSummary";
import { ArtifactInspection } from "../components/ArtifactInspection";
import { ArchitectureWorkspace } from "../components/ArchitectureWorkspace";
import { Progress, Recovery } from "../components/Progress";
import { LearnGuide } from "../components/LearnGuide";
import { matchesCitation } from "../components/Inspector";
import { useRecent, useService } from "../state/queries";
import { usePreferences } from "../state/preferences";
import { useWorkspaceSession, type Crumb } from "../state/workspace";
import { count, dimensionCount } from "../format";
import "./workspace.css";
const views = ["architecture", "overview", "artifacts", "report"] as const;
const viewIcons = {
  architecture: Network,
  overview: LayoutDashboard,
  artifacts: Files,
  report: FileCheck2,
};
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
  const { mode } = usePreferences();
  const { service, available } = useService();
  const recent = useRecent();
  const [state, setState] = useWorkspaceSession(id);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportError, setExportError] = useState<Error | null>(null);
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
  const previousStatus = useRef(detail?.status);
  const ready = graphReady(detail?.status) && !analysis.error;
  const viewParam = search.get("view");
  const view =
    views.find((v) => v === viewParam) ??
    (artifact ? "artifacts" : "architecture");
  useEffect(() => {
    if (artifact && pending(previousStatus.current))
      setSearch({ view: "artifacts" }, { replace: true });
    else if (detail && !views.some((v) => v === viewParam))
      setSearch({ view }, { replace: true });
    previousStatus.current = detail?.status;
  }, [detail, artifact, viewParam, view, setSearch]);
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
      void navigate(
        `/app/analyses/${encodeURIComponent(job.id)}?view=${job.status === "inspected" ? "artifacts" : "architecture"}`,
      );
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
          <Network size={23} />
        </Link>
        <div className="model-switcher">
          <span className="workspace-product">
            Model Anatomy <ChevronRight size={12} /> <span>Inspection</span>
          </span>
          <Menu
            label="Switch model"
            trigger={
              <>
                <span className="model-name">
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
                  onSelect={() =>
                    void navigate(`/app/analyses/${encodeURIComponent(job.id)}`)
                  }
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
        {detail && <Status status={detail.status} />}
        <nav className="primary-views" aria-label="Workspace views">
          {views.map((item) => {
            const Icon = viewIcons[item];
            return (
              <Link
                key={item}
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
                <Icon size={16} strokeWidth={1.8} />
                <span>{item[0].toUpperCase() + item.slice(1)}</span>
              </Link>
            );
          })}
        </nav>
        <div className="workspace-header-actions">
          <ModeControl />
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
      <div className="workbench-body">
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
              {view === "architecture" && !artifact && (
                <div
                  className="model-context"
                  role="region"
                  aria-label="Model context"
                  tabIndex={0}
                >
                  <span className="context-architecture">
                    <Network size={14} />{" "}
                    {detail.architecture ?? "Architecture"}
                  </span>
                  <span>
                    <strong>{count(detail.parameter_count, true)}</strong>{" "}
                    parameters
                  </span>
                  {detail.dimensions.layers != null && (
                    <span>
                      <strong>
                        {dimensionCount(detail.dimensions.layers)}
                      </strong>{" "}
                      layers
                    </span>
                  )}
                  {detail.dimensions.hidden != null && (
                    <span>
                      <strong>
                        {dimensionCount(detail.dimensions.hidden)}
                      </strong>{" "}
                      hidden size
                    </span>
                  )}
                  <Link to="?view=report" className="context-report">
                    {detail.status === "partial"
                      ? "Partial reconstruction · view findings"
                      : "Inspection findings"}
                    <ChevronRight size={13} />
                  </Link>
                </div>
              )}
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
                  {mode === "learn" && (
                    <LearnGuide analysis={detail} node={null} />
                  )}
                </div>
              )}
              {view === "architecture" &&
                (artifact ? (
                  <div className="page-state">
                    <h1>Architecture unavailable</h1>
                    <p>
                      This result contains verified repository artifacts and
                      metadata. No architecture graph was reconstructed.
                    </p>
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
                    {mode === "learn" && (
                      <LearnGuide
                        analysis={detail}
                        node={
                          graph.data.nodes.find(
                            (n) => n.id === state.selected[graph.data.id],
                          ) ?? null
                        }
                        onStep={(step) => {
                          if (step !== "orient")
                            setState((s) => ({
                              ...s,
                              selected: {
                                ...s.selected,
                                [graph.data.id]:
                                  s.selected[graph.data.id] ??
                                  graph.data.nodes.find(
                                    (node) =>
                                      node.kind !== "input" &&
                                      node.kind !== "output",
                                  )?.id ??
                                  graph.data.nodes[0]?.id ??
                                  null,
                              },
                              inspectorOpen: true,
                              questions: false,
                              inspectorTab:
                                step === "tensors"
                                  ? "tensors"
                                  : step === "evidence"
                                    ? "evidence"
                                    : "module",
                            }));
                        }}
                      />
                    )}
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
