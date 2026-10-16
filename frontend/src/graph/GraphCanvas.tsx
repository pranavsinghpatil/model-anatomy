import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Background,
  BackgroundVariant,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  ViewportPortal,
  useReactFlow,
  useStore,
  type Viewport,
} from "@xyflow/react";
import {
  ArrowDownToLine,
  ArrowLeftRight,
  Focus,
  ScanEye,
  Layers,
  Maximize,
  Search,
  ZoomIn,
  ZoomOut,
  Tags,
  ChevronDown,
} from "lucide-react";
import type { ArchitectureGraph, GraphEdge, GraphNode } from "../api/contract";
import { count, scopedNodeId } from "../format";
import { ModelNode } from "./AnatomyNode";
import { nodeTitle } from "./presentation";
import type { GraphLayoutPreference } from "../state/workspace";
import { ErrorNotice, IconButton, Loading } from "../components/Common";
import { layoutGraph, type AnatomyEdge, type AnatomyNode } from "./layout";
import "./graph.css";

const nodeTypes = { anatomy: ModelNode };

function readableAnchor(nodes: AnatomyNode[], selectedNode?: string | null) {
  return (
    nodes.find((node) => node.data.original.id === selectedNode) ??
    nodes.find((node) => node.data.original.repeat > 1) ??
    nodes.find((node) => node.data.original.child_graph_id) ??
    nodes.find(
      (node) => !["input", "output"].includes(node.data.original.kind),
    ) ??
    nodes[0]
  );
}

function Canvas({
  graph,
  selectedNode,
  selectedEdge,
  select,
  open,
  viewport,
  onViewportChange,
  layoutPreference,
  onLayoutPreferenceChange,
  selectEdge,
  header,
  legend,
}: {
  graph: ArchitectureGraph;
  selectedNode: string | null;
  selectedEdge?: GraphEdge | null;
  select: (node: GraphNode) => void;
  open: (node: GraphNode) => void;
  viewport?: Viewport;
  onViewportChange?: (viewport: Viewport) => void;
  layoutPreference: GraphLayoutPreference;
  onLayoutPreferenceChange: (preference: GraphLayoutPreference) => void;
  selectEdge?: (edge: GraphEdge | null) => void;
  header?: (search: ReactNode) => ReactNode;
  legend?: ReactNode;
}) {
  const [layout, setLayout] = useState<{
    nodes: AnatomyNode[];
    edges: AnatomyEdge[];
  } | null>(null);
  const [layoutSignature, setLayoutSignature] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [search, setSearch] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [direction, setDirection] = useState(layoutPreference.direction);
  const [density, setDensity] = useState(layoutPreference.density);
  const [connectionLabels, setConnectionLabels] = useState(
    layoutPreference.connectionLabels ?? false,
  );
  const [neighborhood, setNeighborhood] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [zoom, setZoom] = useState(viewport?.zoom ?? 1);
  const [reduceMotion, setReduceMotion] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const flow = useReactFlow<AnatomyNode, AnatomyEdge>();
  const canvasWidth = useStore((s) => s.width);
  const canvasHeight = useStore((s) => s.height);
  const lastSelection = useRef({ node: selectedNode, width: 0 });
  const fitAfterLayout = useRef(false);
  useEffect(() => {
    const previous = lastSelection.current;
    lastSelection.current = { node: selectedNode, width: canvasWidth };
    if (
      previous.node === selectedNode &&
      (!previous.width || previous.width === canvasWidth)
    )
      return;
    if (!selectedNode || !layout || !canvasWidth || !canvasHeight) return;
    const node = flow.getNode(scopedNodeId(graph.id, selectedNode));
    if (!node) return;
    const current = flow.getViewport();
    const left = node.position.x * current.zoom + current.x;
    const top = node.position.y * current.zoom + current.y;
    if (
      left < 20 ||
      left + (node.width ?? 216) * current.zoom > canvasWidth - 20 ||
      top < 20 ||
      top + (node.height ?? 64) * current.zoom > canvasHeight - 20
    )
      void flow.setCenter(
        node.position.x + (node.width ?? 216) / 2,
        node.position.y + (node.height ?? 64) / 2,
        {
          zoom: current.zoom,
          duration: reduceMotion ? 0 : 150,
        },
      );
  }, [
    selectedNode,
    layout,
    canvasWidth,
    canvasHeight,
    flow,
    graph.id,
    reduceMotion,
  ]);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduceMotion(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    let active = true;
    setLayout(null);
    setLayoutSignature(null);
    setError(null);
    setSearch("");
    layoutGraph(graph, direction, density)
      .then((result) => {
        if (active) {
          setLayout(result);
          setLayoutSignature(`${direction}:${density}`);
        }
      })
      .catch((error) => {
        if (active)
          setError(
            error instanceof Error ? error : new Error("Graph layout failed."),
          );
      });
    return () => {
      active = false;
    };
  }, [graph, attempt, direction, density]);
  useEffect(() => {
    if (
      !layout ||
      !canvasWidth ||
      !canvasHeight ||
      layoutSignature !== `${direction}:${density}` ||
      (viewport && !fitAfterLayout.current)
    )
      return;
    const preferReadableInitialView =
      !viewport && !fitAfterLayout.current && canvasHeight < 500;
    fitAfterLayout.current = false;
    const frame = requestAnimationFrame(() => {
      if (!layout.nodes.length) return;
      if (preferReadableInitialView) {
        void flow.fitView({
          nodes: [readableAnchor(layout.nodes, selectedNode)],
          padding: 0.18,
          minZoom: 0.75,
          maxZoom: 0.85,
          duration: 0,
        });
      } else {
        void flow.fitView({
          padding: 0.12,
          minZoom: 0.08,
          maxZoom: 1,
          duration: 0,
        });
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [
    layout,
    layoutSignature,
    direction,
    density,
    flow,
    viewport,
    canvasHeight,
    selectedNode,
  ]);
  const connected = useMemo(() => {
    if (!selectedNode) return new Set<string>();
    const ids = new Set([selectedNode]);
    for (const edge of graph.edges) {
      if (edge.source === selectedNode) ids.add(edge.target);
      if (edge.target === selectedNode) ids.add(edge.source);
    }
    return ids;
  }, [graph.edges, selectedNode]);
  const nodes = useMemo(
    () =>
      layout?.nodes.map((node) => ({
        ...node,
        selected: node.data.original.id === selectedNode,
        data: {
          ...node.data,
          openGraph: open,
          dimmed: Boolean(
            (search &&
              !`${node.data.original.label} ${node.data.original.module_path ?? ""}`
                .toLowerCase()
                .includes(search.toLowerCase())) ||
            (neighborhood && !connected.has(node.data.original.id)),
          ),
        },
      })) ?? [],
    [layout, selectedNode, open, search, neighborhood, connected],
  );
  const matches = graph.nodes.filter((node) =>
    `${node.label} ${node.module_path ?? ""}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  const edges = useMemo(
    () =>
      layout?.edges.map((edge) => ({
        ...edge,
        selected: edge.data?.original === selectedEdge,
        style: {
          ...edge.style,
          opacity:
            (neighborhood || selectedNode) &&
            edge.source !== scopedNodeId(graph.id, selectedNode ?? "") &&
            edge.target !== scopedNodeId(graph.id, selectedNode ?? "")
              ? 0.25
              : 1,
        },
      })) ?? [],
    [layout, selectedEdge, neighborhood, graph.id, selectedNode],
  );
  if (error)
    return (
      <div className="canvas-state">
        <ErrorNotice
          error={error}
          retry={() => setAttempt((value) => value + 1)}
        />
      </div>
    );
  if (!layout)
    return (
      <div className="canvas-state">
        <Loading text="Arranging architecture" />
      </div>
    );
  if (!graph.nodes.length)
    return (
      <div className="canvas-state">
        <Layers size={26} />
        <strong>No components returned</strong>
        <p>This graph has no nodes.</p>
      </div>
    );
  return (
    <div
      className={`graph-canvas ${connectionLabels ? "show-edge-labels" : ""} ${zoom < 0.6 ? "is-distant" : ""} ${zoom < 0.85 ? "is-overview" : ""} ${zoom >= 0.85 ? "is-detailed" : ""}`}
      data-testid="graph-canvas"
    >
      {header?.(
        <div className="graph-search">
          <Search size={14} aria-hidden="true" />
          <input
            aria-label="Find component"
            placeholder="Find in current graph"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          {search && (
            <div className="search-results">
              {matches.length ? (
                matches.map((node) => (
                  <button
                    key={node.id}
                    onClick={() => {
                      select(node);
                      setSearch("");
                      const found = flow.getNode(
                        scopedNodeId(graph.id, node.id),
                      );
                      if (found)
                        void flow.fitView({
                          nodes: [found],
                          maxZoom: 1.2,
                          duration: reduceMotion ? 0 : 250,
                        });
                    }}
                  >
                    {nodeTitle(node)}
                  </button>
                ))
              ) : (
                <span>No matching components</span>
              )}
            </div>
          )}
        </div>,
      )}
      <ReactFlow<AnatomyNode, AnatomyEdge>
        nodes={nodes}
        edges={edges}
        // Restore a saved view on mount; let React Flow finish gestures and
        // transitions before persisting. Controlling every animation frame
        // resynchronizes the pan/zoom engine and interrupts fit operations.
        defaultViewport={viewport}
        onViewportChange={(nextViewport) => {
          setZoom(nextViewport.zoom);
        }}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => select(node.data.original)}
        onEdgeClick={(_, edge) => {
          if (edge.data) selectEdge?.(edge.data.original);
        }}
        onEdgesChange={(changes) => {
          const selected = changes.find(
            (change) => change.type === "select" && change.selected,
          );
          if (selected?.type === "select") {
            const edge = layout.edges.find((edge) => edge.id === selected.id);
            if (edge?.data) selectEdge?.(edge.data.original);
          } else if (
            changes.some(
              (change) => change.type === "select" && !change.selected,
            )
          ) {
            selectEdge?.(null);
          }
        }}
        onNodesChange={(changes) => {
          const selected = changes.find(
            (change) => change.type === "select" && change.selected,
          );
          if (selected?.type === "select") {
            const node = layout.nodes.find((node) => node.id === selected.id);
            if (node) select(node.data.original);
          }
        }}
        onMoveEnd={(_, nextViewport) => {
          setZoom(nextViewport.zoom);
          onViewportChange?.(nextViewport);
        }}
        nodesDraggable={false}
        nodesConnectable={false}
        deleteKeyCode={null}
        edgesFocusable={true}
        minZoom={0.08}
        maxZoom={1.8}
        proOptions={{ hideAttribution: true }}
        ariaLabelConfig={{
          "node.a11yDescription.default":
            "Select a component to inspect its evidence.",
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={22}
          size={1}
          color="var(--graph-grid)"
        />
        {nodes.some((node) => node.data.isolated) && (
          <ViewportPortal>
            <div
              className="inventory-shelf"
              style={{
                left:
                  Math.min(
                    ...nodes
                      .filter((node) => node.data.isolated)
                      .map((node) => node.position.x),
                  ) - 16,
                top: 12,
                width: 296,
                height:
                  Math.max(
                    ...nodes
                      .filter((node) => node.data.isolated)
                      .map((node) => node.position.y + (node.height ?? 0)),
                  ) + 4,
              }}
            >
              <strong>Unconnected components</strong>
              <span>No connections were returned for these modules.</span>
            </div>
          </ViewportPortal>
        )}
        {canvasWidth >= 700 && canvasHeight >= 360 && (
          <MiniMap
            className="graph-minimap"
            style={{ width: 144, height: 96 }}
            position="bottom-right"
            nodeColor={(node) =>
              node.selected ? "var(--accent)" : "var(--border)"
            }
            nodeStrokeColor="var(--muted)"
            nodeStrokeWidth={1}
            maskColor="var(--graph-mask)"
            pannable
            zoomable
          />
        )}
      </ReactFlow>
      <div className="graph-dock">
        <div
          className="graph-dock-main"
          role="group"
          aria-label="Graph view controls"
        >
          <button
            className="layout-direction"
            type="button"
            aria-label="Fit graph"
            title="Fit the entire current graph"
            onClick={() =>
              void flow.fitView({
                padding: 0.12,
                maxZoom: 1,
                duration: reduceMotion ? 0 : 200,
              })
            }
          >
            <Maximize size={16} />
            <span>Fit</span>
          </button>
          <button
            className="layout-direction"
            type="button"
            aria-label="Readable view"
            title="Focus the selected component at a readable scale"
            onClick={() =>
              void flow.fitView({
                nodes: [readableAnchor(layout.nodes, selectedNode)],
                padding: 0.2,
                minZoom: 0.85,
                maxZoom: 1,
                duration: reduceMotion ? 0 : 200,
              })
            }
          >
            <ScanEye size={16} />
            <span>Readable</span>
          </button>
          <span
            className="zoom-readout"
            aria-label={`Zoom level ${Math.round(zoom * 100)} percent`}
          >
            {Math.round(zoom * 100)}%
          </span>
          <div className="graph-zoom" role="group" aria-label="Zoom controls">
            <IconButton
              label="Zoom out"
              onClick={() =>
                void flow.zoomOut({ duration: reduceMotion ? 0 : 150 })
              }
            >
              <ZoomOut size={15} />
            </IconButton>
            <IconButton
              label="Zoom in"
              onClick={() =>
                void flow.zoomIn({ duration: reduceMotion ? 0 : 150 })
              }
            >
              <ZoomIn size={15} />
            </IconButton>
          </div>
          <details
            className="graph-options"
            open={optionsOpen}
            onToggle={(event) => setOptionsOpen(event.currentTarget.open)}
          >
            <summary aria-label="Options">
              <span>Options</span>
              <ChevronDown size={14} aria-hidden="true" />
            </summary>
            <div
              className="graph-options-panel"
              role="group"
              aria-label="Graph options"
            >
              <button
                className="layout-direction"
                type="button"
                aria-label={`Switch to ${direction === "DOWN" ? "horizontal" : "vertical"} layout`}
                title={`Current layout: ${direction === "DOWN" ? "vertical" : "horizontal"}`}
                onClick={() => {
                  fitAfterLayout.current = true;
                  const next = direction === "DOWN" ? "RIGHT" : "DOWN";
                  setDirection(next);
                  onLayoutPreferenceChange({
                    direction: next,
                    density,
                    connectionLabels,
                  });
                }}
              >
                <ArrowLeftRight size={15} />
                <span>{direction === "DOWN" ? "Vertical" : "Horizontal"}</span>
              </button>
              <button
                className={`layout-direction density-toggle ${density === "detailed" ? "active" : ""}`}
                type="button"
                aria-label={
                  density === "compact"
                    ? "Show detailed node cards"
                    : "Show compact node cards"
                }
                aria-pressed={density === "detailed"}
                onClick={() => {
                  fitAfterLayout.current = true;
                  const next = density === "compact" ? "detailed" : "compact";
                  setDensity(next);
                  onLayoutPreferenceChange({
                    direction,
                    density: next,
                    connectionLabels,
                  });
                }}
              >
                <Layers size={15} />
                <span>{density === "compact" ? "Details" : "Compact"}</span>
              </button>
              <button
                className={`layout-direction neighborhood-toggle ${neighborhood ? "active" : ""}`}
                type="button"
                aria-pressed={neighborhood}
                aria-label="Focus selected component and connected neighbors"
                disabled={!selectedNode}
                onClick={() => setNeighborhood((value) => !value)}
              >
                <Focus size={15} />
                <span>Neighbors</span>
              </button>
              <IconButton
                label={
                  connectionLabels
                    ? "Hide connection labels"
                    : "Show connection labels"
                }
                aria-pressed={connectionLabels}
                onClick={() => {
                  setConnectionLabels(!connectionLabels);
                  onLayoutPreferenceChange({
                    direction,
                    density,
                    connectionLabels: !connectionLabels,
                  });
                }}
              >
                <Tags size={16} />
              </IconButton>
            </div>
          </details>
        </div>
        <div className="graph-dock-meta">
          <div className="graph-legend">{legend}</div>
          <div className="canvas-caption">
            <ArrowDownToLine size={12} aria-hidden="true" />
            <span>{count(graph.nodes.length)} components</span>
            <span className="caption-divider" />
            <span>{count(graph.edges.length)} relationships</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function GraphCanvas(props: Parameters<typeof Canvas>[0]) {
  return (
    <ReactFlowProvider>
      <Canvas {...props} />
    </ReactFlowProvider>
  );
}
