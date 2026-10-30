import { expect, test } from "@playwright/test";
import {
  assemblyFaces,
  assemblyFrame,
  buildAssembly,
} from "../src/anatomy/model";
import { nodeTitle } from "../src/graph/presentation";
import { embedding, rootGraph } from "./service";
import type { ArchitectureGraph, GraphNode } from "../src/api/contract";

test("wide parallel branches cannot overlap disconnected inventory", () => {
  const nodes: GraphNode[] = Array.from({ length: 7 }, (_, index) => ({
    ...embedding,
    id: `parallel-${index}`,
  }));
  const graph: ArchitectureGraph = {
    ...rootGraph,
    nodes,
    edges: nodes.slice(1, 6).map((node) => ({
      ...rootGraph.edges[0],
      source: nodes[0].id,
      target: node.id,
      relation: "dataflow",
    })),
  };
  const assembly = buildAssembly(graph);
  const inventory = assembly.stations.find((station) => station.disconnected)!;
  const lastBranchY = Math.max(
    ...assembly.stations
      .filter((station) => !station.disconnected)
      .map((station) => station.origin[1]),
  );
  expect(inventory.origin[1]).toBeGreaterThan(lastBranchY + 78);
  expect(assembly.connections).toHaveLength(graph.edges.length);
  expect(
    assembly.connections.some((edge) => edge.target === inventory.node.id),
  ).toBe(false);
});

test("largest illustrated attention inventory has finite bounds and bounded geometry", () => {
  const graph: ArchitectureGraph = {
    ...rootGraph,
    nodes: Array.from({ length: 30 }, (_, index) => ({
      ...embedding,
      id: `attention-${index}`,
      kind: "self_attention",
      repeat: 1000000,
    })),
    edges: [],
  };
  const assembly = buildAssembly(graph, "attention-29");
  expect(assembly.stations).toHaveLength(24);
  expect(
    assembly.stations.some((station) => station.node.id === "attention-29"),
  ).toBe(true);
  expect(assembly.omitted).toBe(6);
  expect(assembly.connections).toEqual([]);
  const frame = assemblyFrame(assembly);
  expect(Object.values(frame).every(Number.isFinite)).toBe(true);
  expect(frame.width).toBeGreaterThan(0);
  expect(frame.height).toBeGreaterThan(0);
  expect(assemblyFaces(assembly, 1).length).toBeLessThan(12000);
});

test("known semantic labels remain readable while generic classes use ownership paths", () => {
  const native = {
    ...embedding,
    label: "CustomModel",
    module_type: "CustomModel",
    module_path: "model.vision_tower",
    kind: "module",
  };
  expect(nodeTitle(native)).toBe("vision_tower");
  expect(
    nodeTitle({
      ...native,
      label: "Dropout",
      module_type: "Dropout",
      module_path: "model.drop",
      kind: "dropout",
    }),
  ).toBe("Dropout");
});
