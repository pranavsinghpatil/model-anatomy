import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  expectGraphFramed,
  expectReadableGraph,
  graphNode,
  hasNoHorizontalOverflow,
  openAnalysis,
  openNestedGraph,
  showDetails,
  showQuestions,
} from "./helpers";
import {
  analysisFor,
  block,
  embedding,
  graphs,
  mockService,
  rootGraph,
} from "./service";

test("component detail tabs keep separate readable targets", async ({
  page,
}) => {
  await mockService(page);
  await openAnalysis(page);
  await graphNode(page, embedding.label);
  const tabs = page.getByRole("tablist", { name: "Component detail views" });
  await expect(tabs).toBeVisible();
  const bounds = await tabs.getByRole("tab").evaluateAll((elements) =>
    elements.map((element) => {
      const box = element.getBoundingClientRect();
      return {
        x: box.x,
        right: box.right,
        width: box.width,
        height: box.height,
      };
    }),
  );
  expect(bounds).toHaveLength(3);
  expect(bounds.every((box) => box.width >= 60 && box.height >= 38)).toBe(true);
  expect(bounds[0].right).toBeLessThanOrEqual(bounds[1].x + 1);
  expect(bounds[1].right).toBeLessThanOrEqual(bounds[2].x + 1);
  await tabs.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(tabs.getByRole("tab", { name: /^Tensors/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

async function clickVisibleEdge(page: Page, edge: Locator) {
  const point = await edge.evaluate((edgeElement) => {
    const edgeId = edgeElement.getAttribute("data-id");
    const path = edgeElement.querySelector(
      ".react-flow__edge-interaction",
    ) as SVGPathElement | null;
    if (!edgeId || !path) throw new Error("Edge has no interaction path");

    const length = path.getTotalLength();
    for (let step = 1; step < 20; step += 1) {
      const point = path.getPointAtLength((length * step) / 20);
      const screen = new DOMPoint(point.x, point.y).matrixTransform(
        path.getScreenCTM()!,
      );
      const hit = document.elementFromPoint(screen.x, screen.y);
      if (
        hit?.closest(".react-flow__edge")?.getAttribute("data-id") === edgeId
      ) {
        return { x: screen.x, y: screen.y };
      }
    }
    throw new Error(`Edge ${edgeId} has no exposed hit-test point`);
  });
  await page.mouse.click(point.x, point.y);
}

test("whole architecture opens framed with a readable inspection view and reversible panels", async ({
  page,
  isMobile,
}, testInfo) => {
  await mockService(page);
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expectGraphFramed(page);
  await page.screenshot({
    path: testInfo.outputPath("readable-workspace.png"),
  });
  await page
    .getByRole("button", { name: "Readable view", exact: true })
    .click();
  await expectReadableGraph(page);
  if (isMobile) {
    await page
      .getByRole("button", { name: "Open components", exact: true })
      .click();
    await expect(
      page.getByRole("dialog", { name: "Components", exact: true }),
    ).toBeVisible();
    await page.keyboard.press("Escape");
  } else {
    await page
      .getByRole("button", { name: "Open components", exact: true })
      .click();
    await expect(
      page.getByRole("complementary", { name: /component/i }),
    ).toBeVisible();
  }
  await graphNode(page, embedding.label);
  await showDetails(page, isMobile);
  await expect(page.getByTestId("inspector")).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("selected-component.png"),
  });
});

test("real artifact graph, scoped nesting, tensor and evidence inspection", async ({
  page,
  isMobile,
}, testInfo) => {
  const requests = await mockService(page);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test\?view=architecture/);
  await expect(page.locator(".react-flow__node")).toHaveCount(
    rootGraph.nodes.length,
  );
  await expectGraphFramed(page);
  await expect(
    page.locator(`[data-node-id="${block.id}"] .repeat`),
  ).toContainText(`×${block.repeat}`);
  await page.screenshot({ path: testInfo.outputPath("workspace.png") });
  await graphNode(page, embedding.label);
  await showDetails(page, isMobile);
  await expect(
    page
      .getByTestId("inspector")
      .getByRole("heading", { name: embedding.label, exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(page.locator(".tensor-row").first()).toBeVisible();
  await page.locator(".tensor-summary").first().click();
  await expect(page.locator(".tensor-extra").first()).toContainText("artifact");
  await page.getByRole("tab", { name: /^Evidence/ }).click();
  await expect(page.locator(".evidence-item").first()).toBeVisible();
  if (isMobile) await page.getByRole("link", { name: "Architecture" }).click();
  await graphNode(page, block.label);
  await openNestedGraph(page, block.label);
  await expect(
    page
      .getByRole("navigation", { name: "Graph path" })
      .getByRole("button", { name: block.label, exact: true }),
  ).toBeVisible();
  await expect
    .poll(() =>
      requests.some((request) =>
        request.path.includes(`/graphs/${block.child_graph_id}`),
      ),
    )
    .toBe(true);
  await page
    .getByRole("navigation", { name: "Graph path" })
    .getByRole("button")
    .first()
    .click();
  await expect(page.locator(".react-flow__node")).toHaveCount(
    rootGraph.nodes.length,
  );
  await expect(page.locator("html")).toHaveJSProperty(
    "scrollWidth",
    await page.locator("html").evaluate((element) => element.clientWidth),
  );
  expect(errors).toEqual([]);
});

test("click and keyboard edge selection expose the original relationship", async ({
  page,
}) => {
  // Hit-test a settled SVG path. Fit transitions otherwise move it between
  // reading its screen coordinates and dispatching the pointer event.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  const edges = page.locator(".react-flow__edge");
  await expect(edges).toHaveCount(rootGraph.edges.length);
  await expectGraphFramed(page);
  const minimap = page.getByTestId("rf__minimap");
  if (await minimap.isVisible()) {
    await expect(minimap.locator(".react-flow__minimap-node")).toHaveCount(
      rootGraph.nodes.length,
    );
  }
  await clickVisibleEdge(page, edges.first());
  const inspector = page.getByRole("complementary", { name: "Inspector" });
  await expect(inspector).toContainText("Source component");
  await expect(inspector).toContainText(rootGraph.edges[0].source);
  await expect(inspector).toContainText(rootGraph.edges[0].target);
  const back = page.getByRole("button", { name: "Back to graph", exact: true });
  if (await back.isVisible()) await back.click();

  const keyboardEdge = edges.nth(1);
  await keyboardEdge.focus();
  await keyboardEdge.press("Enter");
  await expect(inspector).toContainText(rootGraph.edges[1].source);
  await expect(inspector).toContainText(rootGraph.edges[1].target);
  if (await back.isVisible()) await back.click();
  await expect(keyboardEdge).toHaveClass(/selected/);
});

test("opening a nested graph clears a previously selected connection", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expectGraphFramed(page);
  await clickVisibleEdge(page, page.locator(".react-flow__edge").first());
  const inspector = page.getByRole("complementary", { name: "Inspector" });
  await expect(
    inspector.getByRole("heading", {
      name:
        rootGraph.edges[0].label ??
        rootGraph.edges[0].relation.replaceAll("_", " "),
      exact: true,
    }),
  ).toBeVisible();
  await expect(inspector).toContainText("Source component");

  const back = page.getByRole("button", { name: "Back to graph", exact: true });
  if (await back.isVisible()) await back.click();
  await page
    .getByRole("button", { name: "Open components", exact: true })
    .isVisible()
    .then(async (visible) => {
      if (visible)
        await page
          .getByRole("button", { name: "Open components", exact: true })
          .click();
    });
  await page
    .getByRole("button", {
      name: `Open ${block.label} nested graph`,
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("navigation", { name: "Graph path" }).getByRole("button", {
      name: block.label,
      exact: true,
    }),
  ).toBeVisible();
  const nestedNode = graphs.get(block.child_graph_id!)!.nodes[0];
  await graphNode(page, nestedNode.label);
  await expect(
    inspector.getByRole("tablist", { name: "Analysis panels" }),
  ).toBeVisible();
  await expect(inspector).not.toContainText("Source component");
});

test("nested navigation preserves each graph viewport through resize and reduced motion", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockService(page);
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();

  await graphNode(page, embedding.label);
  await expect(
    page.getByRole("complementary", { name: "Inspector" }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("button", { name: "Back to graph" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Back to graph" }).click();

  await graphNode(page, block.label);
  await page
    .getByRole("button", { name: "Back to graph", exact: true })
    .click();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByRole("button", { name: "Zoom in" }).click();
  const viewport = page.locator(".react-flow__viewport");
  const rootTransform = await viewport.getAttribute("style");
  await page
    .getByRole("button", { name: "Open components", exact: true })
    .click();
  await page
    .getByRole("button", {
      name: `Open ${block.label} nested graph`,
      exact: true,
    })
    .click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await page
    .getByRole("navigation", { name: "Graph path" })
    .getByRole("button")
    .first()
    .click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect.poll(() => viewport.getAttribute("style")).toBe(rootTransform);

  await page.emulateMedia({ reducedMotion: "reduce" });
  expect(
    await page.evaluate(
      () => matchMedia("(prefers-reduced-motion: reduce)").matches,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Fit graph" }).click();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addStyleTag({ content: "html { zoom: 2; }" });
  await expect.poll(() => page.evaluate(hasNoHorizontalOverflow)).toBe(true);
});

test("worker-offline health still opens a stored inspection", async ({
  page,
  isMobile,
}) => {
  const cachedJob = { ...analysisFor(), cached: true };
  const requests = await mockService(page, {
    job: cachedJob,
    recent: [cachedJob],
    health: {
      status: "degraded",
      api_version: "v1",
      checks: { database: true, storage: true, worker: false },
      analysis_available: false,
      ai_available: false,
      storage_backend: "local",
      reason: "Worker unavailable",
    },
  });
  await page.goto("/app");
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "Stored inspection service available" }),
  ).toContainText("Stored inspection service available");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test\?view=architecture$/);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    analysisFor().model_id,
  );
  await showQuestions(page, isMobile);
  await expect(page.getByLabel("Ask about model evidence")).toBeDisabled();
  await expect(
    page.getByText("AI service unavailable", { exact: true }),
  ).toBeVisible();
  expect(requests.filter((request) => request.method === "POST")).toHaveLength(
    1,
  );
  expect(requests.find((request) => request.method === "POST")?.body).toEqual({
    model: "org/model",
  });
  expect(requests.some((request) => request.path === "/health")).toBe(true);
  expect(
    requests.some((request) => request.path.includes("capabilities")),
  ).toBe(false);
});

test("worker-offline cache miss returns the service error without a fallback", async ({
  page,
}) => {
  const requests = await mockService(page, {
    postStatus: 503,
    postError: "worker_unavailable",
    health: {
      status: "degraded",
      api_version: "v1",
      checks: { database: true, storage: true, worker: false },
      analysis_available: false,
      ai_available: false,
      storage_backend: "local",
      reason: "Worker unavailable",
    },
  });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/cache-miss");
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "Analysis could not be started",
  );
  await expect(page).toHaveURL(/\/app$/);
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(requests.find((request) => request.method === "POST")?.body).toEqual({
    model: "org/cache-miss",
  });
});

test("new visitor starts with an empty workspace", async ({
  page,
}, testInfo) => {
  const requests = await mockService(page, { noRecent: true });
  await page.goto("/app");
  await expect(
    page.getByText(/select a model|inspect a model/i).first(),
  ).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await expect(page).not.toHaveURL(/analysis=/);
  await expect(page.locator(".model-metrics")).toHaveCount(0);
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath("empty-workspace.png") });
  expect(
    requests.some((request) => /\/analyses\/[^/?]+/.test(request.path)),
  ).toBe(false);
  expect(requests.some((request) => request.path.includes("/examples"))).toBe(
    false,
  );
});

test("legacy example query cannot synthesize an analysis or graph", async ({
  page,
}) => {
  const requests = await mockService(page);
  await page.goto("/?analysis=example-gpt2");
  await expect(page).toHaveURL(
    /\/app\/analyses\/example-gpt2(?:\?view=architecture)?$/,
  );
  await expect(page.getByRole("alert")).toContainText("Analysis not found");
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(requests.some((request) => request.path.includes("/examples"))).toBe(
    false,
  );
  expect(requests.some((request) => request.path.includes("/graphs/"))).toBe(
    false,
  );
});

test("optional AI configuration does not disable healthy model analysis", async ({
  page,
}) => {
  await mockService(page, { ai: false, noRecent: true });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled();
});

for (const healthError of ["network", "invalid"] as const)
  test(`unknown health (${healthError}) does not claim online or block stored reports`, async ({
    page,
  }) => {
    await mockService(page, { healthError });
    await page.goto("/app");
    await expect(
      page
        .getByRole("status")
        .filter({ hasText: "Stored inspections are unavailable" }),
    ).toContainText("Stored inspections are unavailable");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeDisabled();
  });

test("submits exact contract and cancels a pending job", async ({ page }) => {
  const job = {
    ...analysisFor("job-test"),
    status: "running" as const,
    stage: "Resolving artifacts",
    root_graph_id: null,
    parameter_count: null,
    dimensions: {},
  };
  const requests = await mockService(page, { job });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/new-model");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Cancel inspection", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("progressbar", { name: "Inspection in progress" }),
  ).toBeVisible();
  await expect(page.locator(".progress-stage")).toContainText(
    "Resolving artifacts",
  );
  expect(requests.find((request) => request.method === "POST")?.body).toEqual({
    model: "org/new-model",
  });
  await page
    .getByRole("button", { name: "Cancel inspection", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Inspection cancelled", exact: true }),
  ).toBeVisible();
  expect(
    requests.some(
      (request) =>
        request.method === "DELETE" &&
        request.path === "/api/v1/analyses/job-test",
    ),
  ).toBe(true);
});

test("polls until completed and renders actual returned graph", async ({
  page,
}) => {
  await mockService(page, {
    job: {
      ...analysisFor("job-test"),
      status: "queued",
      stage: "queued",
      root_graph_id: null,
    },
    pollComplete: true,
  });
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Cancel pending job" }),
  ).toHaveCount(0);
});

test("AI response keeps provenance and citation jumps to associated component", async ({
  page,
  isMobile,
}) => {
  const requests = await mockService(page);
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await graphNode(page, embedding.label);
  await showQuestions(page, isMobile);
  await page
    .getByLabel("Ask about model evidence")
    .fill("Which module owns this component?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByText("AI generated", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Evidence-assisted answers", { exact: true }),
  ).toBeVisible();
  const warning = page.getByText(
    "AI interpretation is not independently verified. Citations and numeric values are checked; confirm other claims against the cited evidence.",
  );
  await expect(warning).toHaveCount(1);
  await expect(warning).not.toBeVisible();
  await page.getByText("1 warning", { exact: true }).click();
  await expect(warning).toBeVisible();
  expect(
    requests.find((request) => request.path.endsWith("/questions"))?.body,
  ).toEqual({
    question: "Which module owns this component?",
    graph_id: rootGraph.id,
    node_id: embedding.id,
  });
  await page.locator(".citation-button").click();
  await expect(
    page.getByTestId("inspector").getByRole("tab", { name: /^Evidence/ }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".citation-record")).toContainText(
    `node:${rootGraph.id}/${embedding.id}`,
  );
});

for (const unknown of [false, true])
  test(`metadata citation exposes exact stored facts${unknown ? " with unknown dimensions" : ""}`, async ({
    page,
    isMobile,
  }, testInfo) => {
    const analysis = unknown
      ? {
          ...analysisFor("job-test"),
          architecture: null,
          revision: null,
          parameter_count: null,
          dimensions: { hidden: null, heads: null, layers: 0 },
          warnings: ["Checkpoint metadata is incomplete."],
        }
      : analysisFor("job-test");
    await mockService(page, { metadataCitation: true, job: analysis });
    await openAnalysis(page, analysis.id);
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    await showQuestions(page, isMobile);
    await page
      .getByLabel("Ask about model evidence")
      .fill("What model metadata is stored?");
    await page.getByRole("button", { name: "Send question" }).click();
    await page.locator(".citation-button").click();
    const evidence = page.getByTestId("model-evidence");
    await expect(
      evidence.getByRole("heading", { name: "Model evidence" }),
    ).toBeVisible();
    await expect(evidence.locator(".citation-record")).toContainText(
      "metadata",
    );
    for (const key of ["model_id", "revision", "architecture"] as const)
      await expect(evidence.getByTestId(`metadata-${key}`)).toHaveText(
        analysis[key] ?? "Unknown",
      );
    const formatted = (value: number | number[] | null | undefined): string =>
      Array.isArray(value)
        ? value.map((item) => item.toLocaleString("en-US")).join(" / ")
        : value == null
          ? "Unknown"
          : value.toLocaleString("en-US");
    await expect(evidence.getByTestId("metadata-parameter_count")).toHaveText(
      formatted(analysis.parameter_count),
    );
    for (const key of ["hidden", "heads", "layers", "vocabulary"] as const)
      await expect(evidence.getByTestId(`metadata-${key}`)).toHaveText(
        formatted(analysis.dimensions[key]),
      );
    if (analysis.warnings.length)
      await expect(
        evidence.getByTestId("metadata-warnings").locator("li"),
      ).toHaveText(analysis.warnings);
    else
      await expect(evidence.getByTestId("metadata-warnings")).toHaveText(
        "No warnings returned.",
      );
    await page.screenshot({
      path: testInfo.outputPath("metadata-citation.png"),
    });
    if (isMobile)
      await page.getByRole("link", { name: "Architecture" }).click();
    await graphNode(page, block.label);
    await expect(
      page.getByTestId("inspector").getByRole("tab", { name: /^Component/ }),
    ).toHaveAttribute("aria-selected", "true");
    await expect(
      page.getByTestId("inspector").locator(".citation-record"),
    ).toHaveCount(0);
    await expect(
      page.getByLabel("Export analysis", { exact: true }),
    ).toBeVisible();
  });

test("exports JSON and ZIP through the authenticated API", async ({ page }) => {
  await mockService(page);
  await openAnalysis(page, "job-test", "report");
  for (const label of ["Report JSON", "Complete ZIP"]) {
    await page.getByLabel("Export analysis", { exact: true }).click();
    const downloaded = page.waitForEvent("download");
    await page.getByRole("menuitem", { name: label, exact: true }).click();
    expect((await downloaded).suggestedFilename()).toMatch(
      label.endsWith("JSON") ? /\.json$/ : /\.zip$/,
    );
  }
});

test("partial and symbolic dimensions remain honest", async ({
  page,
  isMobile,
}) => {
  await mockService(page, {
    job: {
      ...analysisFor("job-test"),
      status: "partial",
      parameter_count: null,
      dimensions: { hidden: null, heads: null, layers: 0 },
    },
    symbolic: true,
  });
  await openAnalysis(page, "job-test", "overview");
  await expect(page.locator(".model-metrics")).toContainText("0");
  await expect(page.locator(".model-metrics")).toContainText("Unknown");
  await expect(
    page.getByRole("link", { name: "View findings", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(
    page.locator(".workspace-header").getByLabel("Analysis status: partial"),
  ).toBeVisible();
  await graphNode(page, embedding.label);
  await showDetails(page, isMobile);
  await expect(page.locator(".port").first()).toContainText("[batch, sequence");
  await expect(page.locator(".inspector-stat")).toContainText(
    embedding.parameter_count.toLocaleString("en-US"),
  );
});

test("tensor pagination requests bounded pages with stable selection", async ({
  page,
  isMobile,
}) => {
  const requests = await mockService(page, { paginated: true });
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await graphNode(page, embedding.label);
  await showDetails(page, isMobile);
  await page.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(page.locator(".tensor-row")).toHaveCount(50);
  await page.getByRole("button", { name: "Next tensor page" }).click();
  await expect(page.locator(".tensor-row")).toHaveCount(1);
  await expect(page.locator(".tensor-row")).toContainText("test.pagination.50");
  expect(
    requests.some((request) => request.path.endsWith("?offset=50&limit=50")),
  ).toBe(true);
});

for (const status of [422, 429, 503])
  test(`structured ${status} submission failure stays actionable`, async ({
    page,
  }) => {
    await mockService(page, { postStatus: status });
    await page.goto("/app");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await page
      .getByRole("button", { name: "Inspect model", exact: true })
      .click();
    await expect(page.getByRole("alert")).toContainText(
      "Analysis could not be started",
    );
    await expect(page.getByRole("alert")).toContainText("test-request");
  });

for (const status of [404, 410])
  test(`structured ${status} analysis response prevents stale graph`, async ({
    page,
  }) => {
    await mockService(page, { getStatus: status });
    await openAnalysis(page);
    await expect(page.getByRole("alert")).toContainText(
      status === 410 ? "Analysis has expired" : "Analysis not found",
    );
    await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  });

test("unavailable AI returns a structured error without fabricated answer", async ({
  page,
  isMobile,
}) => {
  await mockService(page, { questionError: true });
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await showQuestions(page, isMobile);
  await page.getByLabel("Ask about model evidence").fill("What is known?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Explanation provider unavailable",
  );
  await expect(page.locator(".answer")).toHaveCount(0);
});

test("empty service does not synthesize a graph", async ({ page }) => {
  await mockService(page, { noRecent: true });
  await page.goto("/app");
  await expect(page.getByText("No analyses in this session yet")).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
});
