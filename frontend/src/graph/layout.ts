import ELK, { type ElkNode } from "elkjs/lib/elk-api";
import ElkWorker from "elkjs/lib/elk-worker.min.js?worker";
import { MarkerType, type Edge, type Node } from "@xyflow/react";
import type { ArchitectureGraph, GraphNode } from "../api/contract";
import { scopedNodeId } from "../format";
import { edgeLabel, nodeSize, nodeTitle } from "./presentation";

export type AnatomyNode = Node<
  {
    original: GraphNode;
    graphId: string;
    direction: "DOWN" | "RIGHT";
    density: "compact" | "detailed";
    openGraph: (node: GraphNode) => void;
    dimmed: boolean;
    isolated: boolean;
    sourcePorts: GraphPort[];
    targetPorts: GraphPort[];
  },
  "anatomy"
>;
export type GraphPort = { id: string; label: string; position: number };
export type AnatomyEdge = Edge<{
  original: ArchitectureGraph["edges"][number];
  route?: { x: number; y: number }[];
  labelPosition?: { x: number; y: number };
}>;
const elk = new ELK({ workerFactory: () => new ElkWorker() });
const layouts = new WeakMap<ArchitectureGraph, Map<string, Promise<ElkNode>>>();
if (import.meta.hot) import.meta.hot.dispose(() => elk.terminateWorker());

export async function layoutGraph(
  graph: ArchitectureGraph,
  direction: "DOWN" | "RIGHT" = "DOWN",
  density: "compact" | "detailed" = "compact",
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
  const portsByNode = new Map<string, { source: string[]; target: string[] }>();
  const nodeTitles = new Map(
    graph.nodes.map((node) => [node.id, nodeTitle(node)]),
  );
  for (const node of graph.nodes)
    portsByNode.set(node.id, { source: [], target: [] });
  const addPort = (
    nodeId: string,
    side: "source" | "target",
    rawPort: string,
  ) => {
    const ports = portsByNode.get(nodeId)![side];
    const port = rawPort;
    if (!ports.includes(port)) ports.push(port);
  };
  for (const edge of graph.edges) {
    addPort(edge.source, "source", edge.source_port);
    addPort(edge.target, "target", edge.target_port);
  }
  const portId = (nodeId: string, side: "source" | "target", port: string) =>
    `${scopedNodeId(graph.id, nodeId)}::${side}::${encodeURIComponent(port)}`;
  const portsFor = (
    nodeId: string,
    side: "source" | "target",
    positions: Map<string, number>,
  ): GraphPort[] => {
    const labels = portsByNode.get(nodeId)![side];
    return labels.map((label) => ({
      id: portId(nodeId, side, label),
      label: label || "Unspecified port",
      position: positions.get(portId(nodeId, side, label)) ?? 50,
    }));
  };
  const edges: AnatomyEdge[] = graph.edges.map((edge, index) => ({
    id: JSON.stringify([graph.id, "edge", index]),
    source: scopedNodeId(graph.id, edge.source),
    target: scopedNodeId(graph.id, edge.target),
    sourceHandle: portId(edge.source, "source", edge.source_port),
    targetHandle: portId(edge.target, "target", edge.target_port),
    ariaLabel: `${nodeTitles.get(edge.source) ?? edge.source} → ${nodeTitles.get(edge.target) ?? edge.target} · ${edgeLabel(edge, true)}`,
    label: edgeLabel(edge),
    type: "routed",
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
      ...(edge.relation === "conditional_dependency"
        ? { strokeDasharray: "2 5" }
        : edge.relation !== "dataflow"
          ? { strokeDasharray: "8 4" }
          : {}),
    },
    labelStyle: { fill: "var(--muted)", fontSize: 10 },
    labelBgStyle: { fill: "var(--surface)", fillOpacity: 0.96 },
    labelBgPadding: [5, 3],
  }));
  const connectedIds = new Set(
    graph.edges.flatMap((edge) => [edge.source, edge.target]),
  );
  const isolated = graph.edges.length
    ? graph.nodes.filter((node) => !connectedIds.has(node.id))
    : [];
  const connectedNodes = graph.edges.length
    ? graph.nodes.filter((node) => connectedIds.has(node.id))
    : graph.nodes;
  let graphLayouts = layouts.get(graph);
  if (!graphLayouts) {
    graphLayouts = new Map();
    layouts.set(graph, graphLayouts);
  }
  const signature = `${direction}:${density}`;
  let pendingLayout = graphLayouts.get(signature);
  if (!pendingLayout) {
    pendingLayout = elk.layout({
      id: graph.id,
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": direction,
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.spacing.nodeNode": "32",
        "elk.spacing.edgeNode": "24",
        "elk.layered.spacing.nodeNodeBetweenLayers": "64",
        "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      },
      children: connectedNodes.map((node) => {
        const nodePorts = portsByNode.get(node.id)!;
        const side = direction === "RIGHT" ? "EAST" : "SOUTH";
        const opposite = direction === "RIGHT" ? "WEST" : "NORTH";
        return {
          id: scopedNodeId(graph.id, node.id),
          ...nodeSize(node, density),
          ports: [
            ...nodePorts.target.map((port) => ({
              id: portId(node.id, "target", port),
              width: 0,
              height: 0,
              layoutOptions: { "elk.port.side": opposite },
            })),
            ...nodePorts.source.map((port) => ({
              id: portId(node.id, "source", port),
              width: 0,
              height: 0,
              layoutOptions: { "elk.port.side": side },
            })),
          ],
          layoutOptions: { "elk.portConstraints": "FIXED_ORDER" },
        };
      }),
      edges: edges.map((edge) => {
        const text = edgeLabel(edge.data!.original, true);
        return {
          id: edge.id,
          sources: [edge.sourceHandle!],
          targets: [edge.targetHandle!],
          labels: [
            {
              id: `${edge.id}::label`,
              text,
              width: Math.min(240, Math.max(64, text.length * 5.6 + 16)),
              height: 20,
            },
          ],
        };
      }),
    });
    graphLayouts.set(signature, pendingLayout);
    pendingLayout.catch(() => {
      if (graphLayouts.get(signature) === pendingLayout)
        graphLayouts.delete(signature);
    });
  }
  const result = await pendingLayout;
  const positions = new Map(
    result.children?.map((node) => [
      node.id,
      { x: node.x ?? 0, y: node.y ?? 0 },
    ]),
  );
  // Keep disconnected inventory visible in a separate shelf. A missing edge
  // remains missing; spatial organization must not manufacture a dataflow.
  const inventoryX =
    Math.max(
      0,
      ...(result.children ?? []).map(
        (node) => (node.x ?? 0) + (node.width ?? 0),
      ),
    ) + 100;
  let inventoryY = 64;
  for (const node of isolated) {
    positions.set(scopedNodeId(graph.id, node.id), {
      x: inventoryX,
      y: inventoryY,
    });
    inventoryY += nodeSize(node, density).height + 28;
  }
  const routes = new Map<string, { x: number; y: number }[]>();
  const labelPositions = new Map<string, { x: number; y: number }>();
  const portPositions = new Map<string, number>();
  for (const node of result.children ?? []) {
    for (const port of node.ports ?? []) {
      const coordinate =
        direction === "RIGHT"
          ? (port.y ?? 0) + (port.height ?? 0) / 2
          : (port.x ?? 0) + (port.width ?? 0) / 2;
      const extent = direction === "RIGHT" ? node.height : node.width;
      if (extent) portPositions.set(port.id, (coordinate / extent) * 100);
    }
  }
  for (const edge of result.edges ?? []) {
    const label = edge.labels?.[0];
    if (label && Number.isFinite(label.x) && Number.isFinite(label.y)) {
      labelPositions.set(edge.id, {
        x: (label.x ?? 0) + (label.width ?? 0) / 2,
        y: (label.y ?? 0) + (label.height ?? 0) / 2,
      });
    }
    if (edge.sections?.length !== 1) continue;
    const section = edge.sections[0];
    const points = [
      section.startPoint,
      ...(section.bendPoints ?? []),
      section.endPoint,
    ].filter((point): point is { x: number; y: number } => Boolean(point));
    if (points.length > 1) routes.set(edge.id, points);
  }
  return {
    nodes: graph.nodes.map((original) => ({
      id: scopedNodeId(graph.id, original.id),
      type: "anatomy",
      // These match the fixed card and ELK dimensions. Controlled React Flow
      // nodes need dimensions on the user node for the minimap to render them.
      ...nodeSize(original, density),
      position: positions.get(scopedNodeId(graph.id, original.id)) ?? {
        x: 0,
        y: 0,
      },
      data: {
        original,
        graphId: graph.id,
        direction,
        density,
        openGraph: () => {},
        dimmed: false,
        isolated: isolated.some((node) => node.id === original.id),
        sourcePorts: portsFor(original.id, "source", portPositions),
        targetPorts: portsFor(original.id, "target", portPositions),
      },
    })),
    edges: edges.map((edge) => ({
      ...edge,
      data: {
        ...edge.data!,
        route: routes.get(edge.id),
        labelPosition: labelPositions.get(edge.id),
      },
    })),
  };
}
