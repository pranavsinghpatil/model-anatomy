import { useEffect, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
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
import type { ArchitectureGraph, GraphNode } from "../api/contract";
import { count, scopedNodeId } from "../format";
import { ErrorNotice, IconButton, Loading } from "../components/Common";
import { layoutGraph, type AnatomyEdge, type AnatomyNode } from "./layout";

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
  select,
  open,
}: {
  graph: ArchitectureGraph;
  selectedNode: string | null;
  select: (node: GraphNode) => void;
  open: (node: GraphNode) => void;
}) {
  const [layout, setLayout] = useState<{
    nodes: AnatomyNode[];
    edges: AnatomyEdge[];
  } | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [search, setSearch] = useState("");
  const [attempt, setAttempt] = useState(0);
  const flow = useReactFlow<AnatomyNode, AnatomyEdge>();
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
    if (!layout) return;
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
  }, [layout, flow]);
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
    <div className="graph-canvas" data-testid="graph-canvas">
      <ReactFlow<AnatomyNode, AnatomyEdge>
        nodes={nodes}
        edges={layout.edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => select(node.data.original)}
        onNodeDoubleClick={(_, node) => {
          if (node.data.original.child_graph_id) open(node.data.original);
        }}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesFocusable={true}
        fitView
        fitViewOptions={{
          nodes: [readableAnchor(layout.nodes)],
          padding: 0.25,
          minZoom: 0.85,
          maxZoom: 1,
        }}
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
          color="#d9dee1"
        />
        <MiniMap
          position="bottom-right"
          nodeColor={(node) =>
            (node as AnatomyNode).data.original.child_graph_id
              ? "#9b8ab4"
              : "#70aaa2"
          }
          maskColor="rgba(245,247,248,0.7)"
          pannable
          zoomable
        />
      </ReactFlow>
      <div className="graph-zoom">
        <IconButton
          label="Zoom in"
          onClick={() => void flow.zoomIn({ duration: 150 })}
        >
          <ZoomIn size={16} />
        </IconButton>
        <IconButton
          label="Zoom out"
          onClick={() => void flow.zoomOut({ duration: 150 })}
        >
          <ZoomOut size={16} />
        </IconButton>
      </div>
      <div className="canvas-tools">
        <div className="graph-search">
          <Search size={14} />
          <input
            aria-label="Find component"
            placeholder="Find component"
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
                          duration: 250,
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
            void flow.fitView({ padding: 0.2, maxZoom: 1, duration: 200 })
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
              duration: 200,
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
