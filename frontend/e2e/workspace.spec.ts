import { expect, test } from "@playwright/test";
import {
  expectGraphFramed,
  expectReadableGraph,
  graphNode,
  showDetails,
  showQuestions,
} from "./helpers";
import {
  analysisFor,
  block,
  embedding,
  mockService,
  rootGraph,
} from "./service";

test("readable graph with reversible workspace panels", async ({
  page,
  isMobile,
}, testInfo) => {
  await mockService(page);
  await page.goto("/?analysis=example-gpt2");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expectReadableGraph(page);
  await expect(page.locator(".right-panel")).not.toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("readable-workspace.png"),
  });
  await expectGraphFramed(page);
  await page
    .getByRole("button", { name: "Readable view", exact: true })
    .click();
  await expectReadableGraph(page);
  if (!isMobile) {
    const before = (await page.getByTestId("graph-canvas").boundingBox())!
      .width;
    await page
      .getByRole("button", { name: "Close recent analyses", exact: true })
      .first()
      .click();
    await expect(page.locator(".sidebar")).not.toBeVisible();
    await expect
      .poll(
        async () =>
          (await page.getByTestId("graph-canvas").boundingBox())!.width,
      )
      .toBeGreaterThan(before);
    await page
      .getByRole("button", { name: "Open recent analyses", exact: true })
      .first()
      .click();
    await expect(page.locator(".sidebar")).toBeVisible();
    await page
      .getByRole("button", { name: "Open analysis panel", exact: true })
      .click();
    await expect(page.locator(".right-panel")).toBeVisible();
    await expect(
      page
        .getByRole("tablist", { name: "Analysis panels" })
        .getByRole("button"),
    ).toHaveCount(0);
    await page
      .locator(".right-tabs")
      .getByRole("button", { name: "Close analysis panel" })
      .click();
    await expect(page.locator(".right-panel")).not.toBeVisible();
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
  await page.goto("/?analysis=example-gpt2");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page).toHaveURL(/analysis=example-gpt2/);
  await expect(page.locator(".react-flow__node")).toHaveCount(
    rootGraph.nodes.length,
  );
  await expectGraphFramed(page);
  await expect(
    page.locator(`[data-node-id="${block.id}"] .repeat`),
  ).toContainText(`${block.repeat}x`);
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
  if (isMobile)
    await page
      .getByRole("navigation", { name: "Workspace views" })
      .getByRole("button", { name: "Graph" })
      .click();
  await graphNode(page, block.label);
  await page
    .getByRole("button", { name: `Open ${block.label}`, exact: true })
    .click();
  await expect(
    page
      .getByRole("navigation", { name: "Graph breadcrumb" })
      .getByRole("button", { name: block.label }),
  ).toBeVisible();
  await expect
    .poll(() =>
      requests.some((request) =>
        request.path.includes(`/graphs/${block.child_graph_id}`),
      ),
    )
    .toBe(true);
  await page.getByRole("button", { name: "Collapse", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(
    rootGraph.nodes.length,
  );
  await expect(page.locator("html")).toHaveJSProperty(
    "scrollWidth",
    await page.locator("html").evaluate((element) => element.clientWidth),
  );
  expect(errors).toEqual([]);
});

test("503 health disables new analysis without hiding prepared or recent reports", async ({
  page,
  isMobile,
}) => {
  const requests = await mockService(page, {
    recent: [
      analysisFor("example-gpt2"),
      { ...analysisFor("job-old"), created_at: "2026-10-07T09:00:00Z" },
      analysisFor("job-test"),
    ],
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
  await page.goto("/");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page).toHaveURL(/analysis=job-test/);
  await expect(page.locator(".service-state")).toHaveText("API degraded");
  await expect(page.locator(".service-dot")).not.toHaveClass(/online/);
  await expect(page.locator(".availability-note")).toContainText(
    "Worker unavailable",
  );
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByRole("button", { name: "Analyze Model", exact: true }),
  ).toBeDisabled();
  if (isMobile)
    await page
      .getByRole("button", { name: "Open recent analyses" })
      .first()
      .click();
  await page.getByRole("button", { name: /SmolLM2 135M/ }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("SmolLM");
  await expect(page.locator(".model-actions .prepared-label")).toHaveText(
    "Prepared",
  );
  await showQuestions(page, isMobile);
  await expect(page.getByLabel("Ask about model evidence")).toBeDisabled();
  await expect(
    page.getByText("AI service unavailable", { exact: true }),
  ).toBeVisible();
  expect(requests.filter((request) => request.method === "POST")).toHaveLength(
    0,
  );
  expect(requests.some((request) => request.path === "/health")).toBe(true);
  expect(
    requests.some((request) => request.path.includes("capabilities")),
  ).toBe(false);
});

test("new visitor starts empty and chooses a prepared example explicitly", async ({
  page,
  isMobile,
}, testInfo) => {
  const requests = await mockService(page, { noRecent: true });
  await page.goto("/");
  await expect(page.locator(".service-state")).toHaveText("API v1");
  await expect(page.getByText("No analysis selected").first()).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await expect(page).not.toHaveURL(/analysis=/);
  await expect(page.locator(".model-metrics")).toHaveCount(0);
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByRole("button", { name: "Analyze Model", exact: true }),
  ).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath("empty-workspace.png") });
  expect(
    requests.some((request) => /\/analyses\/[^/?]+/.test(request.path)),
  ).toBe(false);
  if (isMobile)
    await page
      .getByRole("button", { name: "Open recent analyses" })
      .first()
      .click();
  await expect(
    page.getByRole("button", { name: /SmolLM2 135M/ }),
  ).toContainText("Prepared");
  await page.getByRole("button", { name: /SmolLM2 135M/ }).click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page).toHaveURL(/analysis=example-smollm/);
});

test("optional AI configuration does not disable healthy model analysis", async ({
  page,
}) => {
  await mockService(page, { ai: false, noRecent: true });
  await page.goto("/");
  await expect(page.locator(".service-state")).toHaveText("API v1");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByRole("button", { name: "Analyze Model", exact: true }),
  ).toBeEnabled();
});

for (const healthError of ["network", "invalid"] as const)
  test(`unknown health (${healthError}) does not claim online or block stored reports`, async ({
    page,
  }) => {
    await mockService(page, { healthError });
    await page.goto("/?analysis=example-gpt2");
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    await expect(page.locator(".service-state")).toHaveText("API unavailable");
    await expect(page.locator(".service-dot")).not.toHaveClass(/online/);
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await expect(
      page.getByRole("button", { name: "Analyze Model", exact: true }),
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
  await page.goto("/");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/new-model");
  await page
    .getByRole("button", { name: "Analyze Model", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Cancel pending job" }),
  ).toBeVisible();
  await expect(page.getByText("Analyzing model")).toBeVisible();
  await expect(page.locator(".job-progress")).toContainText(
    "Resolving artifacts",
  );
  expect(requests.find((request) => request.method === "POST")?.body).toEqual({
    model: "org/new-model",
  });
  await page.getByRole("button", { name: "Cancel pending job" }).click();
  await expect(
    page.getByText("Analysis cancelled", { exact: true }),
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
  await page.goto("/?analysis=job-test");
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
  await page.goto("/?analysis=example-gpt2");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await graphNode(page, embedding.label);
  await showQuestions(page, isMobile);
  await page
    .getByLabel("Ask about model evidence")
    .fill("Which module owns this component?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByText("AI generated", { exact: true })).toBeVisible();
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
      : analysisFor("example-smollm");
    await mockService(page, { metadataCitation: true, job: analysis });
    await page.goto(`/?analysis=${analysis.id}`);
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
    const formatted = (value: number | null | undefined) =>
      value == null ? "Unknown" : value.toLocaleString("en-US");
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
      await page
        .getByRole("navigation", { name: "Workspace views" })
        .getByRole("button", { name: "Graph", exact: true })
        .click();
    await expect(
      page.getByLabel("Export analysis", { exact: true }),
    ).toBeVisible();
  });

test("exports JSON and ZIP through the authenticated API", async ({ page }) => {
  await mockService(page);
  await page.goto("/");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  for (const label of ["Architecture JSON", "Complete ZIP"]) {
    await page.getByLabel("Export analysis", { exact: true }).click();
    await expect
      .poll(async () => {
        const anchor = (await page
          .getByLabel("Export analysis", { exact: true })
          .boundingBox())!;
        const menu = (await page.locator(".export-options").boundingBox())!;
        return Math.abs(menu.y - anchor.y - anchor.height - 6);
      })
      .toBeLessThan(1);
    const downloaded = page.waitForEvent("download");
    await page.getByRole("button", { name: label, exact: true }).click();
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
  await page.goto("/?analysis=job-test");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page.locator(".partial-notice")).toContainText(
    "Partial analysis",
  );
  await expect(page.locator(".model-metrics")).toContainText("0");
  await expect(page.locator(".model-metrics")).toContainText("Unknown");
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
  await page.goto("/?analysis=example-gpt2");
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
    await page.goto("/");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await page
      .getByRole("button", { name: "Analyze Model", exact: true })
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
    await page.goto("/?analysis=job-test");
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
  await page.goto("/");
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
  await mockService(page, { empty: true });
  await page.goto("/");
  await expect(page.getByText("No analysis selected").first()).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
});
