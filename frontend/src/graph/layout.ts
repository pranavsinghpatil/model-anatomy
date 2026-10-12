import ELK from "elkjs/lib/elk-api";
import ElkWorker from "elkjs/lib/elk-worker.min.js?worker";
import { MarkerType, type Edge, type Node } from "@xyflow/react";
import type { ArchitectureGraph, GraphNode } from "../api/contract";
import { scopedNodeId, shape } from "../format";

export type AnatomyNode = Node<
  {
    original: GraphNode;
    graphId: string;
    openGraph: (node: GraphNode) => void;
    dimmed: boolean;
  },
  "anatomy"
>;
export type AnatomyEdge = Edge<{
  original: ArchitectureGraph["edges"][number];
}>;
const elk = new ELK({ workerFactory: () => new ElkWorker() });
if (import.meta.hot) import.meta.hot.dispose(() => elk.terminateWorker());

export async function layoutGraph(
  graph: ArchitectureGraph,
): Promise<{ nodes: AnatomyNode[]; edges: AnatomyEdge[] }> {
  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  if (
    nodeIds.size !== graph.nodes.length ||
    graph.edges.some(
      (edge) => !nodeIds.has(edge.source) || !nodeIds.has(edge.target),
    )
  ) {
    throw new Error(
      "The service graph contains duplicate node IDs or unresolved edge endpoints.",
    );
  }
  const edges: AnatomyEdge[] = graph.edges.map((edge, index) => ({
    id: JSON.stringify([graph.id, "edge", index]),
    source: scopedNodeId(graph.id, edge.source),
    target: scopedNodeId(graph.id, edge.target),
    label:
      edge.label ??
      (edge.tensor
        ? shape(edge.tensor.shape)
        : edge.relation === "dataflow"
          ? undefined
          : edge.relation),
    type: "smoothstep",
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 12,
      height: 12,
      color:
        edge.relation === "dataflow"
          ? "var(--graph-edge)"
          : "var(--graph-semantic-edge)",
    },
    data: { original: edge },
    style: {
      stroke:
        edge.relation === "dataflow"
          ? "var(--graph-edge)"
          : "var(--graph-semantic-edge)",
      strokeWidth: 1.3,
      ...(edge.relation !== "dataflow" ? { strokeDasharray: "5 4" } : {}),
    },
    labelStyle: { fill: "var(--muted)", fontSize: 10 },
    labelBgStyle: { fill: "var(--surface)", fillOpacity: 0.96 },
    labelBgPadding: [5, 3],
  }));
  const result = await elk.layout({
    id: graph.id,
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.spacing.nodeNode": "40",
      "elk.layered.spacing.nodeNodeBetweenLayers": "56",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    },
    children: graph.nodes.map((node) => ({
      id: scopedNodeId(graph.id, node.id),
      width: 242,
      height: 108,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  });
  const positions = new Map(
    result.children?.map((node) => [
      node.id,
      { x: node.x ?? 0, y: node.y ?? 0 },
    ]),
  );
  return {
    nodes: graph.nodes.map((original) => ({
      id: scopedNodeId(graph.id, original.id),
      type: "anatomy",
      position: positions.get(scopedNodeId(graph.id, original.id)) ?? {
        x: 0,
        y: 0,
      },
      data: { original, graphId: graph.id, openGraph: () => {}, dimmed: false },
    })),
    edges,
  };
}
