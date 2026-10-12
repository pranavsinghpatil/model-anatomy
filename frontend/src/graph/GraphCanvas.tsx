import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
  type Viewport,
  type NodeProps,
} from "@xyflow/react";
import {
  ArrowDownToLine,
  ChevronRight,
  ScanEye,
  Layers,
  Maximize,
  Search,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { ArchitectureGraph, GraphEdge, GraphNode } from "../api/contract";
import { count, scopedNodeId } from "../format";
import { ErrorNotice, IconButton, Loading } from "../components/Common";
import { layoutGraph, type AnatomyEdge, type AnatomyNode } from "./layout";
import "./graph.css";

function ModelNode({ data, selected }: NodeProps<AnatomyNode>) {
  const node = data.original;
  return (
    <div
      className={`model-node ${selected ? "is-selected" : ""} ${data.dimmed ? "is-dimmed" : ""}`}
      data-node-id={node.id}
    >
      <Handle type="target" position={Position.Top} />
      <div className="node-kind">
        <span className={`kind-mark ${node.child_graph_id ? "violet" : ""}`} />
        {node.kind.replaceAll("_", " ")}
        {node.repeat > 1 && (
          <span
            className="repeat"
            title={node.repetition?.mode ?? "Unknown repetition mode"}
          >
            <Layers size={11} />
            {count(node.repeat)}x
          </span>
        )}
      </div>
      <div className="node-title" title={node.label}>
        {node.label}
      </div>
      <div className="node-footer">
        <span title={count(node.parameter_count)}>
          {count(node.parameter_count, true)} params
        </span>
        {node.child_graph_id && (
          <button
            className="node-open nodrag nopan"
            aria-label={`Open ${node.label}`}
            title={`Open ${node.label}`}
            onClick={(event) => {
              event.stopPropagation();
              data.openGraph(node);
            }}
          >
            <ChevronRight size={16} />
          </button>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
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
  selectEdge,
}: {
  graph: ArchitectureGraph;
  selectedNode: string | null;
  selectedEdge?: GraphEdge | null;
  select: (node: GraphNode) => void;
  open: (node: GraphNode) => void;
  viewport?: Viewport;
  onViewportChange?: (viewport: Viewport) => void;
  selectEdge?: (edge: GraphEdge | null) => void;
}) {
  const [layout, setLayout] = useState<{
    nodes: AnatomyNode[];
    edges: AnatomyEdge[];
  } | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [search, setSearch] = useState("");
  const [attempt, setAttempt] = useState(0);
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
      left + 242 * current.zoom > canvasWidth - 20 ||
      top < 20 ||
      top + 108 * current.zoom > canvasHeight - 20
    )
      void flow.setCenter(node.position.x + 121, node.position.y + 54, {
        zoom: current.zoom,
        duration: reduceMotion ? 0 : 150,
      });
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
    setError(null);
    setSearch("");
    layoutGraph(graph)
      .then((result) => {
        if (active) setLayout(result);
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
  }, [graph, attempt]);
  useEffect(() => {
    if (!layout || viewport) return;
    const frame = requestAnimationFrame(() => {
      if (!layout.nodes.length) return;
      void flow.fitView({
        // Focus a component: a zoom floor on a wide graph can crop both branches.
        nodes: [readableAnchor(layout.nodes)],
        padding: 0.25,
        minZoom: 0.85,
        maxZoom: 1,
        duration: 0,
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [layout, flow, viewport]);
  const nodes = useMemo(
    () =>
      layout?.nodes.map((node) => ({
        ...node,
        selected: node.data.original.id === selectedNode,
        data: {
          ...node.data,
          openGraph: open,
          dimmed: Boolean(
            search &&
            !`${node.data.original.label} ${node.data.original.module_path ?? ""}`
              .toLowerCase()
              .includes(search.toLowerCase()),
          ),
        },
      })) ?? [],
    [layout, selectedNode, open, search],
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
      })) ?? [],
    [layout, selectedEdge],
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
      className={`graph-canvas ${zoom < 0.55 ? "is-distant" : ""}`}
      data-testid="graph-canvas"
    >
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
        {...(!viewport
          ? {
              fitView: true,
              fitViewOptions: {
                nodes: [readableAnchor(layout.nodes)],
                padding: 0.25,
                minZoom: 0.85,
                maxZoom: 1,
              },
            }
          : {})}
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
        {canvasWidth >= 700 && canvasHeight >= 360 && (
          <MiniMap
            className="graph-minimap"
            position="bottom-right"
            nodeColor={(node) =>
              (node as AnatomyNode).data.original.child_graph_id
                ? "var(--graph-child)"
                : "var(--graph-node)"
            }
            maskColor="var(--graph-mask)"
            pannable
            zoomable
          />
        )}
      </ReactFlow>
      <div className="graph-zoom">
        <IconButton
          label="Zoom in"
          onClick={() => void flow.zoomIn({ duration: reduceMotion ? 0 : 150 })}
        >
          <ZoomIn size={16} />
        </IconButton>
        <IconButton
          label="Zoom out"
          onClick={() =>
            void flow.zoomOut({ duration: reduceMotion ? 0 : 150 })
          }
        >
          <ZoomOut size={16} />
        </IconButton>
      </div>
      <div className="canvas-tools">
        <div className="graph-search">
          <Search size={14} />
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
                    {node.label}
                  </button>
                ))
              ) : (
                <span>No matching components</span>
              )}
            </div>
          )}
        </div>
        <IconButton
          label="Fit graph"
          onClick={() =>
            void flow.fitView({
              padding: 0.2,
              maxZoom: 1,
              duration: reduceMotion ? 0 : 200,
            })
          }
        >
          <Maximize size={16} />
        </IconButton>
        <IconButton
          label="Readable view"
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
        </IconButton>
      </div>
      <div className="canvas-caption">
        <ArrowDownToLine size={12} />
        <span>{count(graph.nodes.length)} components</span>
        <span className="caption-divider" />
        <span>{count(graph.edges.length)} relationships</span>
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
