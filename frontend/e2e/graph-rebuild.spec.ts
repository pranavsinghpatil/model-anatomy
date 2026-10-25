import { expect, test } from "@playwright/test";
import type { ArchitectureGraph, GraphNode } from "../src/api/contract";
import { count, shape } from "../src/format";
import { block, embedding, mockService, rootGraph } from "./service";
import { graphNode, openAnalysis } from "./helpers";

async function nodeCenters(page: import("@playwright/test").Page) {
  return page.locator(".model-node").evaluateAll((nodes) =>
    Object.fromEntries(
      nodes.map((node) => {
        const rect = node.getBoundingClientRect();
        return [
          node.getAttribute("data-node-id")!,
          { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 },
        ];
      }),
    ),
  ) as Promise<Record<string, { x: number; y: number }>>;
}

function averageEdgeTravel(
  centers: Record<string, { x: number; y: number }>,
  axis: "x" | "y",
) {
  const edges = rootGraph.edges.filter((edge) => edge.relation === "dataflow");
  return (
    edges.reduce((sum, edge) => {
      const from = centers[edge.source];
      const to = centers[edge.target];
      return sum + (from && to ? Math.abs(to[axis] - from[axis]) : 0);
    }, 0) / edges.length
  );
}

test("fits the full graph, relayouts direction, and focuses only real neighbors", async ({
  page,
  isMobile,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await expect(page.locator(".model-node")).toHaveCount(rootGraph.nodes.length);
  // Short workspaces start at a readable component; Fit explicitly requests
  // the overview whose complete bounds this test verifies.
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();

  await expect
    .poll(() =>
      page.getByTestId("graph-canvas").evaluate((canvas) => {
        const bounds = canvas.getBoundingClientRect();
        return Array.from(canvas.querySelectorAll(".model-node")).every(
          (node) => {
            const rect = node.getBoundingClientRect();
            return (
              rect.left >= bounds.left - 1 &&
              rect.right <= bounds.right + 1 &&
              rect.top >= bounds.top - 1 &&
              rect.bottom <= bounds.bottom + 1
            );
          },
        );
      }),
    )
    .toBe(true);

  const verticalCenters = await nodeCenters(page);
  expect(averageEdgeTravel(verticalCenters, "y")).toBeGreaterThan(
    averageEdgeTravel(verticalCenters, "x"),
  );
  await page.getByText("Options", { exact: true }).click();
  await page
    .getByRole("button", { name: "Switch to horizontal layout" })
    .click();
  await page
    .getByRole("button", { name: "Switch to vertical layout" })
    .waitFor();
  const horizontalCenters = await nodeCenters(page);
  expect(averageEdgeTravel(horizontalCenters, "x")).toBeGreaterThan(
    averageEdgeTravel(horizontalCenters, "y"),
  );
  await page.getByText("Options", { exact: true }).click();

  await page.getByText("Options", { exact: true }).click();
  await page.getByRole("button", { name: "Show detailed node cards" }).click();
  await page.getByText("Options", { exact: true }).click();
  await expect(
    page.locator(".model-node.density-detailed").first(),
  ).toBeVisible();
  await expect
    .poll(() =>
      page.locator(".model-node.density-detailed").evaluateAll((nodes) =>
        nodes
          .filter((node) => {
            const card = node.getBoundingClientRect();
            const last =
              node.querySelector(".node-open") ??
              node.querySelector(".node-footer");
            return (
              last && last.getBoundingClientRect().bottom > card.bottom + 1
            );
          })
          .map((node) => node.getAttribute("data-node-id")),
      ),
    )
    .toEqual([]);
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await page.getByText("Options", { exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Switch to vertical layout" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Show compact node cards" }),
  ).toBeVisible();
  await page.getByText("Options", { exact: true }).click();

  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await graphNode(page, embedding.label);
  if (isMobile)
    await page.getByRole("button", { name: "Back to graph" }).click();
  await page.getByText("Options", { exact: true }).click();
  await page
    .getByRole("button", {
      name: "Focus selected component and connected neighbors",
    })
    .click();
  await page.getByText("Options", { exact: true }).click();

  const expected = new Set([embedding.id]);
  for (const edge of rootGraph.edges) {
    if (edge.source === embedding.id) expected.add(edge.target);
    if (edge.target === embedding.id) expected.add(edge.source);
  }
  const visible = await page
    .locator(".model-node:not(.is-dimmed)")
    .evaluateAll((nodes) =>
      nodes.map((node) => node.getAttribute("data-node-id")),
    );
  expect(new Set(visible)).toEqual(expected);

  await page
    .getByRole("button", { name: "Readable view", exact: true })
    .click();
  const selected = page.locator(`.model-node[data-node-id="${embedding.id}"]`);
  await expect(selected.locator(".node-kind")).toContainText(embedding.kind);
  await expect(selected.locator(".node-ports")).toBeVisible();
  await expect(
    selected.locator(".node-ports > div").first().locator("code"),
  ).toHaveText(
    embedding.inputs[0] ? shape(embedding.inputs[0].shape) : "Unknown",
  );
  await expect(
    selected.locator(".node-ports > div").last().locator("code"),
  ).toHaveText(
    embedding.outputs[0] ? shape(embedding.outputs[0].shape) : "Unknown",
  );
  await expect(selected.locator(".node-footer strong")).toHaveText(
    count(embedding.parameter_count, true),
  );
  await expect(selected.locator(".node-footer strong")).toHaveAttribute(
    "title",
    count(embedding.parameter_count),
  );
});

test("distinguishes graph boundaries, repeats, module paths, and isolated inventory", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);

  const input = rootGraph.nodes.find((node) => node.kind === "input")!;
  const output = rootGraph.nodes.find((node) => node.kind === "output")!;
  const sharedClass = (id: string, modulePath: string): GraphNode => ({
    ...block,
    id,
    label: "SharedLayer",
    kind: "module",
    module_type: "SharedLayer",
    module_path: modulePath,
    repeat: 1,
    repetition: null,
    child_graph_id: null,
  });
  const firstLayer = sharedClass("layer-first", "model.layers.0");
  const secondLayer = sharedClass("layer-second", "model.layers.1");
  const repeatedBlock: GraphNode = { ...block, id: "repeated-block" };
  const isolated: GraphNode = {
    ...embedding,
    id: "isolated-embedding",
    child_graph_id: null,
  };
  const edgeTemplate = rootGraph.edges[0];
  const fixture: ArchitectureGraph = {
    ...rootGraph,
    id: "graph-ui-distinction-fixture",
    nodes: [input, firstLayer, secondLayer, repeatedBlock, output, isolated],
    edges: [
      { ...edgeTemplate, source: input.id, target: firstLayer.id },
      { ...edgeTemplate, source: firstLayer.id, target: secondLayer.id },
      { ...edgeTemplate, source: secondLayer.id, target: repeatedBlock.id },
      { ...edgeTemplate, source: repeatedBlock.id, target: output.id },
    ],
  };
  expect(
    fixture.edges.every(
      (edge) => edge.source !== isolated.id && edge.target !== isolated.id,
    ),
  ).toBe(true);
  let fixtureServed = false;
  await page.route("**/api/v1/analyses/job-test/graphs/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === `/api/v1/analyses/job-test/graphs/${rootGraph.id}`) {
      fixtureServed = true;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(fixture),
      });
      return;
    }
    await route.fallback();
  });

  await openAnalysis(page);
  await expect.poll(() => fixtureServed).toBe(true);
  await expect(page.locator(".model-node")).toHaveCount(fixture.nodes.length);

  const first = page.locator(`.model-node[data-node-id="${firstLayer.id}"]`);
  const second = page.locator(`.model-node[data-node-id="${secondLayer.id}"]`);
  await expect(first.locator(".node-title")).toHaveText("SharedLayer [0]");
  await expect(second.locator(".node-title")).toHaveText("SharedLayer [1]");
  await expect(first.locator(".node-kind")).toHaveText("module");
  await expect(second.locator(".node-kind")).toHaveText("module");
  await expect(first.locator(".node-title")).toHaveAttribute(
    "title",
    "SharedLayer · model.layers.0",
  );
  await expect(second.locator(".node-title")).toHaveAttribute(
    "title",
    "SharedLayer · model.layers.1",
  );

  const repeated = page.locator(
    `.model-node[data-node-id="${repeatedBlock.id}"]`,
  );
  await expect(repeated).toHaveClass(/is-repeated/);
  await expect(repeated.locator(".repeat")).toHaveText(
    `×${count(block.repeat)}`,
  );
  await expect(
    repeated.getByRole("button", { name: "Open Transformer Block" }),
  ).toContainText("Explore representative block");

  const inputNode = page.locator(`.model-node[data-node-id="${input.id}"]`);
  const outputNode = page.locator(`.model-node[data-node-id="${output.id}"]`);
  await expect(inputNode).toHaveClass(/category-boundary/);
  await expect(outputNode).toHaveClass(/category-boundary/);
  const inputIcon = await inputNode
    .locator(".anatomy-node-icon svg")
    .evaluate((icon) => icon.innerHTML);
  const outputIcon = await outputNode
    .locator(".anatomy-node-icon svg")
    .evaluate((icon) => icon.innerHTML);
  expect(inputIcon).not.toBe(outputIcon);

  await expect(
    page.getByText("Unconnected components", { exact: true }),
  ).toBeVisible();
  const connectedRight = Math.max(
    ...(await Promise.all(
      fixture.nodes
        .filter((node) => node.id !== isolated.id)
        .map(async (node) => {
          const bounds = await page
            .locator(`.model-node[data-node-id="${node.id}"]`)
            .boundingBox();
          return bounds!.x + bounds!.width;
        }),
    )),
  );
  const isolatedBounds = await page
    .locator(`.model-node[data-node-id="${isolated.id}"]`)
    .boundingBox();
  expect(isolatedBounds!.x).toBeGreaterThan(connectedRight);
  await expect(page.locator(".react-flow__edge-path")).toHaveCount(
    fixture.edges.length,
  );
  await expect(
    page.locator(`.react-flow__edge[data-id*="${isolated.id}"]`),
  ).toHaveCount(0);

  await page.getByText("Options", { exact: true }).click();
  await page.getByRole("button", { name: "Show detailed node cards" }).click();
  await page.getByText("Options", { exact: true }).click();
  const detailedInput = page.locator(
    `.model-node[data-node-id="${input.id}"] .boundary-shape`,
  );
  const detailedOutput = page.locator(
    `.model-node[data-node-id="${output.id}"] .boundary-shape`,
  );
  await expect(detailedInput).toHaveText(
    input.outputs[0] ? shape(input.outputs[0].shape) : "Unknown shape",
  );
  await expect(detailedOutput).toHaveText(
    output.inputs[0] ? shape(output.inputs[0].shape) : "Unknown shape",
  );
});
