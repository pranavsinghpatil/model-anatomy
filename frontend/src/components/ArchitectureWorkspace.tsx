import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "react-router-dom";
import * as Tabs from "@radix-ui/react-tabs";
import { Group, Panel, Separator } from "react-resizable-panels";
import {
  ArrowLeft,
  ChevronRight,
  Layers,
  ListTree,
  MessageSquare,
  PanelLeftClose,
  PanelRightClose,
  PanelRightOpen,
  Network,
  Box,
} from "lucide-react";
import type {
  ArchitectureGraph,
  AnalysisDetail,
  Citation,
  GraphEdge,
  GraphNode,
} from "../api/contract";
import { shape } from "../format";
import { GraphCanvas } from "../graph/GraphCanvas";
import { nodeTitle } from "../graph/presentation";
import { Inspector, EvidenceList } from "./Inspector";
import { Questions } from "./Questions";
import { Button, Drawer } from "./ui";
import {
  useMedia,
  type WorkspaceSession,
  type Crumb,
  type GraphLayoutPreference,
} from "../state/workspace";
import type { Dispatch, ReactNode, SetStateAction } from "react";
import "./architecture.css";
import "../anatomy/anatomy.css";
import { defaultAnatomyView } from "../anatomy/model";
import { SceneBoundary } from "../anatomy/SceneBoundary";
import { Loading } from "./Common";
const AnatomyScene = lazy(() =>
  import("../anatomy/AnatomyScene").then((module) => ({
    default: module.AnatomyScene,
  })),
);
type Props = {
  analysis: AnalysisDetail;
  graph: ArchitectureGraph;
  state: WorkspaceSession;
  setState: Dispatch<SetStateAction<WorkspaceSession>>;
  citation: Citation | null;
  onClearCitation: () => void;
  onCitation: (citation: Citation) => Promise<void>;
  ai: boolean;
};
export function ArchitectureWorkspace({
  analysis,
  graph,
  state,
  setState,
  citation,
  onClearCitation,
  onCitation,
  ai,
}: Props) {
  const desktop = useMedia("(min-width: 1280px)");
  const docked = useMedia("(min-width: 1024px)");
  const [drawer, setDrawer] = useState(false);
  const [edge, setEdge] = useState<GraphEdge | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedScene = searchParams.get("scene");
  const scene =
    requestedScene === "anatomy" || requestedScene === "graph"
      ? requestedScene
      : state.scene;
  useEffect(() => {
    if (requestedScene === "anatomy" || requestedScene === "graph")
      setState((s) =>
        s.scene === requestedScene ? s : { ...s, scene: requestedScene },
      );
  }, [requestedScene, setState]);
  const changeScene = (value: "graph" | "anatomy") => {
    setState((s) => ({ ...s, scene: value }));
    const next = new URLSearchParams(searchParams);
    if (value === "graph") next.delete("scene");
    else next.set("scene", value);
    setSearchParams(next, { replace: true });
  };
  useEffect(() => setEdge(null), [graph.id]);
  useEffect(() => {
    if (citation) setEdge(null);
  }, [citation]);
  const panelRef = useRef<HTMLDivElement>(null);
  const selected =
    graph.nodes.find((n) => n.id === state.selected[graph.id]) ?? null;
  const focusPanel = () => {
    if (!docked)
      requestAnimationFrame(() =>
        panelRef.current?.querySelector<HTMLElement>("button")?.focus(),
      );
  };
  const select = useCallback(
    (node: GraphNode) => {
      setEdge(null);
      onClearCitation();
      setState((s) => ({
        ...s,
        selected: { ...s.selected, [graph.id]: node.id },
        inspectorOpen: true,
        questions: false,
      }));
      setDrawer(false);
      if (!matchMedia("(min-width: 1024px)").matches)
        requestAnimationFrame(() =>
          panelRef.current?.querySelector<HTMLElement>("button")?.focus(),
        );
    },
    [graph.id, setState, onClearCitation],
  );
  const focusPart = useCallback(
    (node: GraphNode) => {
      setEdge(null);
      onClearCitation();
      setState((s) => ({
        ...s,
        selected: { ...s.selected, [graph.id]: node.id },
        questions: false,
      }));
    },
    [graph.id, setState, onClearCitation],
  );
  const root: Crumb = {
    id: analysis.root_graph_id!,
    label: analysis.model_id.split("/").at(-1)!,
  };
  const crumbs = state.path.length ? state.path : [root];
  const open = useCallback(
    (node: GraphNode) => {
      if (!node.child_graph_id) return;
      setEdge(null);
      onClearCitation();
      setDrawer(false);
      setState((s) => ({
        ...s,
        path: [
          ...(s.path.length
            ? s.path
            : [
                {
                  id: analysis.root_graph_id!,
                  label: analysis.model_id.split("/").at(-1)!,
                },
              ]),
          { id: node.child_graph_id!, label: node.label },
        ],
        inspectorOpen: false,
        questions: false,
      }));
    },
    [analysis.root_graph_id, analysis.model_id, setState, onClearCitation],
  );
  const viewport = useCallback(
    (value: { x: number; y: number; zoom: number }) =>
      setState((s) => ({
        ...s,
        viewports: { ...s.viewports, [graph.id]: value },
      })),
    [graph.id, setState],
  );
  function close() {
    setState((s) => ({ ...s, inspectorOpen: false }));
    requestAnimationFrame(() => {
      const target =
        document.querySelector<HTMLElement>(
          `[data-component-id="${CSS.escape(selected?.id ?? "")}"]`,
        ) ?? document.querySelector<HTMLElement>(".components-trigger");
      target?.focus();
    });
  }
  const outline = (
    <div className="component-outline">
      <p className="outline-scope">
        Current graph · {graph.nodes.length} components
      </p>
      <ul aria-label="Components in current graph">
        {graph.nodes.map((node) => (
          <li key={node.id}>
            <button
              data-component-id={node.id}
              className={`component-select ${node.id === selected?.id ? "selected" : ""}`}
              aria-pressed={node.id === selected?.id}
              onClick={() => select(node)}
            >
              <span className="outline-mark" />
              <span>
                <strong>{nodeTitle(node)}</strong>
                <small>
                  {node.kind.replaceAll("_", " ")}
                  {node.repeat > 1 ? ` · ${node.repeat} instances` : ""}
                  {nodeTitle(node) !== node.label && (
                    <span className="outline-module-type">{node.label}</span>
                  )}
                </small>
              </span>
            </button>
            {node.child_graph_id && (
              <Button
                className="outline-open"
                variant="ghost"
                aria-label={`Open ${node.label} nested graph`}
                onClick={() => open(node)}
              >
                <ChevronRight size={15} />
              </Button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
  const navigator = (
    <aside className="component-navigator" aria-label="Component navigator">
      <div className="panel-heading">
        <h2>
          <ListTree size={16} />
          Components
        </h2>
        <Button
          variant="ghost"
          aria-label="Hide components"
          onClick={() => setState((s) => ({ ...s, navigationOpen: false }))}
        >
          <PanelLeftClose size={16} />
        </Button>
      </div>
      {outline}
    </aside>
  );
  const componentsControl = !desktop ? (
    <Drawer
      open={drawer}
      onOpenChange={setDrawer}
      title="Components"
      trigger={
        <Button className="components-trigger" aria-label="Open components">
          <ListTree size={16} />
          <span>Components</span>
        </Button>
      }
    >
      {outline}
    </Drawer>
  ) : (
    !state.navigationOpen && (
      <Button
        className="components-trigger"
        aria-label="Open components"
        onClick={() => setState((s) => ({ ...s, navigationOpen: true }))}
      >
        <ListTree size={16} />
        <span>Components</span>
      </Button>
    )
  );
  const graphPath = (
    <nav className="breadcrumbs" aria-label="Graph path">
      {crumbs.map((crumb, index) => (
        <span key={crumb.id}>
          {index > 0 && <ChevronRight size={13} />}
          <button
            aria-current={index === crumbs.length - 1 ? "page" : undefined}
            onClick={() => {
              setEdge(null);
              onClearCitation();
              setState((s) => ({
                ...s,
                path: crumbs.slice(0, index + 1),
                inspectorOpen: false,
              }));
            }}
          >
            {crumb.label}
          </button>
        </span>
      ))}
    </nav>
  );
  const presentation = (
    <div
      className="scene-switcher"
      role="group"
      aria-label="Architecture presentation"
    >
      <button
        aria-pressed={scene === "graph"}
        onClick={() => changeScene("graph")}
      >
        <Network size={13} /> Graph
      </button>
      <button
        aria-pressed={scene === "anatomy"}
        onClick={() => changeScene("anatomy")}
      >
        <Box size={13} /> Anatomy
      </button>
    </div>
  );
  const panelActions = (
    <div className="graph-panel-actions">
      <Button
        variant="ghost"
        aria-label="Model questions"
        onClick={() => {
          setState((s) => ({ ...s, questions: true, inspectorOpen: true }));
          focusPanel();
        }}
      >
        <MessageSquare size={17} />
        <span>Questions</span>
      </Button>
      <Button
        variant="ghost"
        aria-label={
          state.inspectorOpen ? "Close analysis panel" : "Open analysis panel"
        }
        onClick={() =>
          state.inspectorOpen
            ? close()
            : setState((s) => ({ ...s, inspectorOpen: true }))
        }
      >
        {state.inspectorOpen ? (
          <PanelRightClose size={16} />
        ) : (
          <PanelRightOpen size={16} />
        )}
        <span>{state.inspectorOpen ? "Hide details" : "Show details"}</span>
      </Button>
    </div>
  );
  const graphHeader = (search: ReactNode) => (
    <div className="graph-header">
      <div className="graph-header-context">
        {componentsControl}
        {graphPath}
        {presentation}
        {panelActions}
      </div>
      {search}
    </div>
  );
  const canvas = (
    <div className="architecture-canvas-region">
      {scene === "anatomy" && (
        <div className="graph-topbar">
          {componentsControl}
          {graphPath}
          {presentation}
          {panelActions}
        </div>
      )}
      {scene === "anatomy" ? (
        <SceneBoundary key={graph.id} technical={() => changeScene("graph")}>
          <Suspense
            fallback={
              <div className="canvas-state">
                <Loading text="Opening model assembly" />
              </div>
            }
          >
            <AnatomyScene
              analysis={analysis}
              graph={graph}
              selected={selected}
              view={state.anatomyViews?.[graph.id] ?? defaultAnatomyView}
              onView={(view) =>
                setState((s) => ({
                  ...s,
                  anatomyViews: { ...s.anatomyViews, [graph.id]: view },
                }))
              }
              select={select}
              focus={focusPart}
              open={open}
              evidence={(node) => {
                select(node);
                setState((s) => ({ ...s, inspectorTab: "evidence" }));
              }}
              technical={() => changeScene("graph")}
            />
          </Suspense>
        </SceneBoundary>
      ) : (
        <GraphCanvas
          key={`${analysis.id}:${graph.id}`}
          graph={graph}
          selectedNode={selected?.id ?? null}
          selectedEdge={edge}
          select={select}
          open={open}
          header={graphHeader}
          legend={
            <>
              <span>
                <i />
                Dataflow
              </span>
              <span>
                <i className="structural" />
                Structural relation
              </span>
            </>
          }
          viewport={state.viewports[graph.id]}
          onViewportChange={viewport}
          layoutPreference={
            state.graphLayouts[graph.id] ?? {
              direction: "DOWN",
              density: "compact",
            }
          }
          onLayoutPreferenceChange={(preference: GraphLayoutPreference) =>
            setState((s) => ({
              ...s,
              graphLayouts: { ...s.graphLayouts, [graph.id]: preference },
            }))
          }
          selectEdge={(value) => {
            onClearCitation();
            setEdge(value);
            if (!value) return;
            setState((s) => ({ ...s, inspectorOpen: true, questions: false }));
            focusPanel();
          }}
        />
      )}
    </div>
  );
  const inspector = (
    <aside
      className="workspace-inspector"
      ref={panelRef}
      aria-label="Inspector"
    >
      <Tabs.Root
        className="inspector-panel-tabs"
        value={state.questions ? "questions" : "details"}
        onValueChange={(value) =>
          setState((s) => ({ ...s, questions: value === "questions" }))
        }
      >
        <div className="inspector-header">
          <h2 className="sr-only">Inspector</h2>
          {!docked && (
            <Button className="inspector-back" variant="ghost" onClick={close}>
              <ArrowLeft size={16} />
              Back to graph
            </Button>
          )}
          <Tabs.List className="panel-tabs" aria-label="Analysis panels">
            <Tabs.Trigger value="details">Details</Tabs.Trigger>
            <Tabs.Trigger value="questions">Questions</Tabs.Trigger>
          </Tabs.List>
          <Button
            variant="ghost"
            aria-label="Close analysis panel"
            onClick={close}
          >
            <PanelRightClose size={16} />
          </Button>
        </div>
        <Tabs.Content
          value="details"
          className="inspector-scroll"
          forceMount
          hidden={state.questions}
        >
          {edge ? (
            <div className="edge-inspector">
              <span className="eyebrow">CONNECTION</span>
              <h2>{edge.label ?? edge.relation.replaceAll("_", " ")}</h2>
              <dl className="properties">
                <dt>Source component</dt>
                <dd>
                  <code>{edge.source}</code>
                </dd>
                <dt>Target component</dt>
                <dd>
                  <code>{edge.target}</code>
                </dd>
                <dt>Source port</dt>
                <dd>{edge.source_port ?? "Unknown"}</dd>
                <dt>Target port</dt>
                <dd>{edge.target_port ?? "Unknown"}</dd>
                <dt>Relationship</dt>
                <dd>{edge.relation}</dd>
                <dt>Shape</dt>
                <dd>
                  <code>{shape(edge.tensor?.shape)}</code>
                </dd>
              </dl>
              <h3>Evidence</h3>
              <EvidenceList items={edge.evidence} />
            </div>
          ) : (
            <Inspector
              analysis={analysis}
              analysisId={analysis.id}
              graphId={graph.id}
              node={selected}
              openGraph={open}
              citation={citation}
              tab={state.inspectorTab}
              onTabChange={(tab) =>
                setState((s) => ({ ...s, inspectorTab: tab }))
              }
            />
          )}
        </Tabs.Content>
        <Tabs.Content
          value="questions"
          forceMount
          hidden={!state.questions}
          className="question-container"
        >
          <Questions
            key={analysis.id}
            analysisId={analysis.id}
            graphId={graph.id}
            nodeId={selected?.id}
            nodeLabel={selected?.label}
            ready
            available={ai}
            jump={onCitation}
          />
        </Tabs.Content>
      </Tabs.Root>
    </aside>
  );
  if (!docked)
    return (
      <div
        className={`architecture-layout mobile-layout ${state.inspectorOpen ? "show-inspector" : ""}`}
      >
        <div className="mobile-canvas">{canvas}</div>
        {state.inspectorOpen && inspector}
      </div>
    );
  return (
    <Group
      className="architecture-layout"
      orientation="horizontal"
      defaultLayout={state.layout}
      onLayoutChanged={(layout) => setState((s) => ({ ...s, layout }))}
    >
      {desktop && state.navigationOpen && (
        <>
          <Panel id="navigator" defaultSize={240} minSize={200} maxSize={360}>
            {navigator}
          </Panel>
          <Separator
            className="resize-handle"
            aria-label="Resize component navigator"
          />
        </>
      )}
      <Panel id="canvas" minSize={360}>
        {canvas}
      </Panel>
      {state.inspectorOpen && (
        <>
          <Separator className="resize-handle" aria-label="Resize inspector" />
          <Panel
            id="inspector-panel"
            defaultSize={360}
            minSize={300}
            maxSize="45%"
          >
            {inspector}
          </Panel>
        </>
      )}
    </Group>
  );
}
