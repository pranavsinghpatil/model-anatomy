import type { ArchitectureGraph, GraphNode } from "../api/contract";
import { nodeTitle } from "../graph/presentation";
import type { Face, Vec3 } from "./solids";
import { box } from "./solids";
import { mechanism } from "./mechanisms";

export type Station = { node: GraphNode; origin: Vec3; disconnected: boolean };
export type Assembly = {
  stations: Station[];
  width: number;
  depth: number;
  omitted: number;
  relationCount: number;
  hasDataflow: boolean;
  connections: { id: string; source: string; target: string }[];
};
export type AnatomyView = { separation: number; angle: number; zoom: number };
export const defaultAnatomyView: AnatomyView = {
  separation: 0,
  angle: 42,
  zoom: 1,
};

// Presentation ranks are derived only from returned dataflow edges. They do
// not supply missing edges or turn hierarchy into an execution trace.
export function buildAssembly(
  graph: ArchitectureGraph,
  selectedId?: string | null,
): Assembly {
  const ids = new Set(graph.nodes.map((node) => node.id));
  if (
    ids.size !== graph.nodes.length ||
    graph.edges.some((edge) => !ids.has(edge.source) || !ids.has(edge.target))
  )
    throw new Error("The report has unresolved component relationships.");
  const selected = graph.nodes.find((node) => node.id === selectedId);
  const nodes = graph.nodes.slice(0, 24);
  if (selected && !nodes.includes(selected)) nodes[nodes.length - 1] = selected;
  const included = new Set(nodes.map((node) => node.id));
  const edges = graph.edges.filter(
    (edge) =>
      edge.relation === "dataflow" &&
      included.has(edge.source) &&
      included.has(edge.target),
  );
  const connected = new Set(
    edges.flatMap((edge) => [edge.source, edge.target]),
  );
  const degrees = new Map(nodes.map((node) => [node.id, 0]));
  const ranks = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges)
    degrees.set(edge.target, degrees.get(edge.target)! + 1);
  const queue = nodes
    .filter((node) => degrees.get(node.id) === 0)
    .map((node) => node.id);
  const visited = new Set<string>();
  while (queue.length) {
    const id = queue.shift()!;
    if (visited.has(id)) continue;
    visited.add(id);
    for (const edge of edges.filter((edge) => edge.source === id)) {
      ranks.set(
        edge.target,
        Math.max(ranks.get(edge.target)!, ranks.get(id)! + 1),
      );
      degrees.set(edge.target, degrees.get(edge.target)! - 1);
      if (degrees.get(edge.target) === 0) queue.push(edge.target);
    }
  }
  const rows = new Map<number, number>();
  const branchCounts = new Map<number, number>();
  for (const node of nodes) {
    if (!connected.has(node.id) || !visited.has(node.id)) continue;
    const rank = ranks.get(node.id)!;
    branchCounts.set(rank, (branchCounts.get(rank) ?? 0) + 1);
  }
  const inventoryRow = Math.max(1, ...branchCounts.values()) + 1;
  let inventoryIndex = 0;
  const hasDataflow = edges.length > 0;
  const stations = nodes.map((node, index): Station => {
    const disconnected =
      hasDataflow && (!connected.has(node.id) || !visited.has(node.id));
    const inventory = disconnected ? inventoryIndex++ : index;
    const rank =
      hasDataflow && !disconnected ? ranks.get(node.id)! : inventory % 6;
    const row = disconnected
      ? inventoryRow + Math.floor(inventory / 6)
      : hasDataflow
        ? (rows.get(rank) ?? 0)
        : Math.floor(index / 6);
    if (!disconnected && hasDataflow) rows.set(rank, row + 1);
    return { node, origin: [rank * 104, row * 108, 0], disconnected };
  });
  const width =
    Math.max(0, ...stations.map((station) => station.origin[0])) + 96;
  const depth =
    Math.max(0, ...stations.map((station) => station.origin[1])) + 104;
  return {
    stations,
    width,
    depth,
    omitted: graph.nodes.length - nodes.length,
    relationCount: graph.edges.length,
    hasDataflow,
    connections: edges.map((edge, index) => ({
      id: `${edge.source}:${edge.source_port}:${edge.target}:${edge.target_port}:${index}`,
      source: edge.source,
      target: edge.target,
    })),
  };
}

export function assemblyFaces(assembly: Assembly, separation: number): Face[] {
  if (!assembly.stations.length) return [];
  const floor = box(
    "bench-base",
    -48,
    -52,
    -12,
    assembly.width,
    assembly.depth,
    10,
  );
  const feet = [
    [-38, -42],
    [assembly.width - 62, -42],
    [-38, assembly.depth - 66],
    [assembly.width - 62, assembly.depth - 66],
  ].flatMap(([x, y], i) => box(`bench-foot-${i}`, x, y, -20, 14, 14, 8));
  const rails = [-42, assembly.depth - 60].flatMap((y, index) =>
    box(`bench-rail-${index}`, -42, y, -2, assembly.width - 12, 5, 3),
  );
  return [
    ...floor,
    ...feet,
    ...rails,
    ...assembly.stations.flatMap((station) =>
      mechanism(station.node, station.origin, separation),
    ),
  ];
}

export function project(point: Vec3, angle: number): [number, number] {
  const a = (angle * Math.PI) / 180;
  return [
    point[0] * Math.cos(a) - point[1] * Math.sin(a),
    (point[0] * Math.sin(a) + point[1] * Math.cos(a)) * 0.5 - point[2] * 0.866,
  ];
}
export function faceDepth(face: Face, angle: number) {
  const a = (angle * Math.PI) / 180;
  return (
    face.points.reduce(
      (depth, point) =>
        depth +
        point[0] * Math.sin(a) * 0.866 +
        point[1] * Math.cos(a) * 0.866 +
        point[2] * 0.5,
      0,
    ) / face.points.length
  );
}
export function faceVisible(face: Face, angle: number) {
  if (face.points.length < 3) return false;
  const [a, b, c] = face.points;
  const u = b.map((value, i) => value - a[i]);
  const v = c.map((value, i) => value - a[i]);
  const normal = [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  const radians = (angle * Math.PI) / 180;
  return (
    normal[0] * Math.sin(radians) * 0.866 +
      normal[1] * Math.cos(radians) * 0.866 +
      normal[2] * 0.5 >
    0.0001
  );
}
export function facePath(face: Face, angle: number) {
  return (
    face.points
      .map(
        (point, i) =>
          `${i ? "L" : "M"}${project(point, angle)
            .map((value) => value.toFixed(2))
            .join(",")}`,
      )
      .join(" ") + "Z"
  );
}
export function assemblyFrame(assembly: Assembly) {
  const poses = [0, 1].flatMap((separation) =>
    assemblyFaces(assembly, separation),
  );
  const points = [30, 42, 60].flatMap((angle) =>
    poses.flatMap((face) => face.points.map((point) => project(point, angle))),
  );
  // Avoid spreading the full projected mesh into Math.min/max: a bounded
  // collection of detailed mechanisms can still exceed engine argument limits.
  let minX = 0;
  let minY = 0;
  let maxX = 100;
  let maxY = 100;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  return {
    x: minX - 36,
    y: minY - 32,
    width: maxX - minX + 72,
    height: maxY - minY + 64,
  };
}
export function partLabel(node: GraphNode) {
  return `${nodeTitle(node)} · ${node.kind.replaceAll("_", " ")}${node.repeat > 1 ? ` · ${node.repeat} reported instances` : ""}`;
}
