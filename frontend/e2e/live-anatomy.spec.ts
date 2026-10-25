import { expect, test } from "@playwright/test";
import { writeFile } from "node:fs/promises";
import type { AnalysisDetail, ArchitectureGraph } from "../src/api/contract";

async function clickVisibleFacet(
  page: import("@playwright/test").Page,
  nodeId: string,
  isMobile: boolean,
) {
  const faces = page.locator(`.assembly-face[data-part-id="${nodeId}"]`);
  let point: { x: number; y: number } | null = null;
  for (let index = 0; index < (await faces.count()) && !point; index += 1) {
    const face = faces.nth(index);
    if (!(await face.isVisible())) continue;
    point = await face.evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      for (let row = 1; row < 10; row += 1) {
        for (let column = 1; column < 10; column += 1) {
          const x = bounds.left + (bounds.width * column) / 10;
          const y = bounds.top + (bounds.height * row) / 10;
          if (document.elementFromPoint(x, y) === element) return { x, y };
        }
      }
      return null;
    });
  }
  expect(
    point,
    `Facet ${nodeId} should expose a hit-test point`,
  ).not.toBeNull();
  if (isMobile) await page.touchscreen.tap(point!.x, point!.y);
  else await page.mouse.click(point!.x, point!.y);
}

test("live SmolLM2-135M supports graph, anatomy, lazy children, evidence, and export", async ({
  page,
  isMobile,
}, testInfo) => {
  test.setTimeout(360_000);
  test.skip(
    process.env.E2E_LIVE_API !== "1",
    "Set E2E_LIVE_API=1 when the isolated live API is available on port 8001.",
  );
  const errors: string[] = [];
  const requests: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/api/v1/analyses/"))
      requests.push(`${request.method()} ${url.pathname}`);
  });

  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("HuggingFaceTB/SmolLM2-135M");
  const inspect = page.getByRole("button", {
    name: "Inspect model",
    exact: true,
  });
  await expect(inspect).toBeEnabled({ timeout: 60_000 });
  const createdResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await page.screenshot({ path: testInfo.outputPath("live-entry.png") });
  await inspect.click();
  const created = (await (await createdResponse).json()) as AnalysisDetail;
  await expect(page).toHaveURL(new RegExp(`/app/analyses/${created.id}`));
  await expect(
    page.getByRole("link", { name: "Architecture", exact: true }),
  ).toHaveAttribute("aria-current", "page", { timeout: 150_000 });

  const analysisPath = `/api/v1/analyses/${created.id}`;
  await expect
    .poll(
      async () =>
        (
          (await page.request
            .get(analysisPath)
            .then((response) => response.json())) as AnalysisDetail
        ).status,
      { timeout: 240_000, intervals: [1000, 2000, 4000] },
    )
    .toBe("completed");
  const analysis = (await (
    await page.request.get(analysisPath)
  ).json()) as AnalysisDetail;
  expect(analysis.architecture).toMatch(/llama/i);
  expect(analysis.revision).toBeTruthy();
  expect(analysis.parameter_count).toBeGreaterThan(0);
  expect(analysis.input_names).toContain("input_ids");
  expect(analysis.quality?.architecture_checks).toEqual({
    module_inventory: true,
    parameter_mapping: true,
    native_architecture: true,
    module_topology: true,
  });

  const graphResponse = await page.request.get(
    `${analysisPath}/graphs/${analysis.root_graph_id}`,
  );
  expect(graphResponse.ok()).toBe(true);
  const root = (await graphResponse.json()) as ArchitectureGraph;
  expect(root.nodes.length).toBeGreaterThan(0);
  const nodeIds = new Set(root.nodes.map((node) => node.id));
  expect(
    root.edges.every(
      (edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target),
    ),
  ).toBe(true);
  const representative = root.nodes.find((node) =>
    Boolean(node.child_graph_id),
  );
  expect(representative).toBeTruthy();
  const evidenceNode = root.nodes.find(
    (node) =>
      node.evidence.length > 0 &&
      node.module_path &&
      node.parameter_count != null &&
      node.parameter_count > 0,
  );
  expect(evidenceNode).toBeTruthy();

  await expect(page.locator(".model-node").first()).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("live-technical-workspace.png"),
  });
  const presentation = page.getByRole("group", {
    name: "Architecture presentation",
  });
  await presentation
    .getByRole("button", { name: "Anatomy", exact: true })
    .click();
  const assembly = page.getByTestId("anatomy-drawing");
  await expect(assembly).toBeVisible({ timeout: 60_000 });
  await expect(assembly.locator("svg .assembly-face").first()).toBeVisible();
  await expect(assembly).toHaveAttribute("data-motion", /visible|reduced/);

  const rootChildId = representative!.child_graph_id!;
  expect(
    requests.some((request) => request.includes(`/graphs/${rootChildId}`)),
  ).toBe(false);
  await clickVisibleFacet(page, representative!.id, isMobile);
  const backToGraph = page.getByRole("button", {
    name: "Back to graph",
    exact: true,
  });
  if (isMobile && (await backToGraph.isVisible())) await backToGraph.click();
  await page.getByRole("button", { name: "Explore inside" }).click();
  await expect(
    page.getByRole("navigation", { name: "Graph path" }).getByRole("button", {
      name: representative!.label,
      exact: true,
    }),
  ).toBeVisible();
  await expect
    .poll(() =>
      requests.some((request) => request.includes(`/graphs/${rootChildId}`)),
    )
    .toBe(true);

  const childResponse = await page.request.get(
    `${analysisPath}/graphs/${rootChildId}`,
  );
  expect(childResponse.ok()).toBe(true);
  const child = (await childResponse.json()) as ArchitectureGraph;
  const attention = child.nodes.find((node) => node.kind === "self_attention");
  const mlp = child.nodes.find((node) => node.kind === "gated_mlp");
  expect(attention).toBeTruthy();
  expect(mlp).toBeTruthy();
  await expect(
    assembly.locator(`.assembly-face[data-part-id="${attention!.id}"]`).first(),
  ).toBeVisible();
  await expect(
    assembly.locator(`.assembly-face[data-part-id="${mlp!.id}"]`).first(),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "Graph path" })
    .getByRole("button")
    .first()
    .click();
  await expect(page.getByTestId("anatomy-drawing")).toBeVisible();

  const inspectNode = root.nodes.find((node) => node.id === evidenceNode!.id)!;
  await clickVisibleFacet(page, inspectNode.id, isMobile);
  const inspector = page.getByTestId("inspector");
  const evidenceAction = page.getByRole("button", { name: /^Evidence/ });
  const evidenceTab = inspector.getByRole("tab", { name: /^Evidence/ });
  if (await evidenceAction.count()) await evidenceAction.click();
  else await evidenceTab.click();
  await expect(evidenceTab).toHaveAttribute("aria-selected", "true");
  await expect(
    inspector.getByRole("heading", { name: "Provenance" }),
  ).toBeVisible({
    timeout: 60_000,
  });

  const tensorTab = inspector.getByRole("tab", { name: /^Tensors/ });
  await tensorTab.click();
  await expect(
    inspector.getByRole("heading", { name: "Owned tensors" }),
  ).toBeVisible({
    timeout: 60_000,
  });
  await expect(inspector.locator(".tensor-row").first()).toBeVisible({
    timeout: 60_000,
  });

  const exportResponse = await page.request.get(`${analysisPath}/exports/json`);
  expect(exportResponse.ok()).toBe(true);
  const report = await exportResponse.json();
  expect(report.metadata.revision).toBe(analysis.revision);
  const evidencePath = testInfo.outputPath("live-analysis-evidence.json");
  await writeFile(
    evidencePath,
    JSON.stringify(
      {
        analysis_id: created.id,
        revision: analysis.revision,
        architecture: analysis.architecture,
        parameter_count: analysis.parameter_count,
        root_graph_id: analysis.root_graph_id,
        root_node_count: root.nodes.length,
        child_graph_id: rootChildId,
        child_node_kinds: child.nodes.map((node) => node.kind),
        exported_revision: report.metadata.revision,
      },
      null,
      2,
    ),
  );
  await testInfo.attach("live-analysis-evidence.json", {
    path: evidencePath,
    contentType: "application/json",
  });
  await page.getByLabel("Export analysis", { exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("menuitem", { name: "Report JSON", exact: true })
    .click();
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.json$/);
  await page.screenshot({ path: testInfo.outputPath("live-anatomy.png") });
  expect(errors).toEqual([]);
});
