import { expect, test } from "@playwright/test";
import type { ArchitectureGraph } from "../src/api/contract";
import type { AnalysisDetail, Health } from "../src/api/contract";
import {
  expectGraphFramed,
  expectReadableGraph,
  graphNode,
  showDetails,
  showQuestions,
} from "./helpers";

test("live prepared backend contract", async ({ page, isMobile }, testInfo) => {
  test.setTimeout(90_000);
  test.skip(
    !process.env.E2E_LIVE_API,
    "Set E2E_LIVE_API=1 when the service runs on port 8001.",
  );
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(error.message));
  const healthResponse = await page.request.get("/health");
  const health = (await healthResponse.json()) as Health;
  expect(healthResponse.status()).toBe(
    Object.values(health.checks).every(Boolean) ? 200 : 503,
  );
  expect(health.api_version).toBe("v1");
  const exampleResponse = await page.request.get("/api/v1/examples");
  expect(exampleResponse.ok()).toBe(true);
  const examples = (await exampleResponse.json()).examples as {
    id: string;
    model_id: string;
    title: string;
  }[];
  expect(examples.some((example) => example.id === "example-smollm")).toBe(
    true,
  );
  expect(examples.some((example) => example.id === "example-gpt2")).toBe(true);
  await page.goto("/");
  await expect(page.getByText("No analysis selected").first()).toBeVisible();
  await expect(page).not.toHaveURL(/analysis=/);
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await expect(page.locator(".service-state")).toHaveText(
    health.status === "ok" ? "API v1" : "API degraded",
  );
  await page.screenshot({
    path: testInfo.outputPath("live-empty-workspace.png"),
    animations: "disabled",
  });
  for (const id of ["example-smollm", "example-gpt2"]) {
    if (isMobile)
      await page
        .getByRole("button", { name: "Open recent analyses" })
        .first()
        .click();
    await page
      .getByRole("button", {
        name: new RegExp(examples.find((example) => example.id === id)!.title),
      })
      .click();
    const analysisResponse = await page.request.get(`/api/v1/analyses/${id}`);
    expect(analysisResponse.ok()).toBe(true);
    const analysis = await analysisResponse.json();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      analysis.model_id,
    );
    await expect(page.locator(".model-actions .prepared-label")).toHaveText(
      "Prepared",
    );
    const graphResponse = await page.request.get(
      `/api/v1/analyses/${id}/graphs/${encodeURIComponent(analysis.root_graph_id)}`,
    );
    expect(graphResponse.ok()).toBe(true);
    const graph = (await graphResponse.json()) as ArchitectureGraph;
    await expect(page.locator(".react-flow__node")).toHaveCount(
      graph.nodes.length,
    );
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    await expectReadableGraph(page);
    await expectGraphFramed(page);
    await page
      .getByRole("button", { name: "Readable view", exact: true })
      .click();
    await expectReadableGraph(page);
    await page.screenshot({
      path: testInfo.outputPath(`live-${id}-graph.png`),
      animations: "disabled",
    });
    const selected = graph.nodes.find(
      (node) =>
        node.module_path && node.parameter_count && node.evidence.length,
    )!;
    expect(selected).toBeTruthy();
    await graphNode(page, selected.label);
    await showDetails(page, isMobile);
    await expect(
      page
        .getByTestId("inspector")
        .getByRole("heading", { name: selected.label, exact: true }),
    ).toBeVisible();
    await expect(page.locator(".inspector-stat")).toContainText(
      selected.parameter_count!.toLocaleString("en-US"),
    );
    await page.getByRole("tab", { name: /^Tensors/ }).click();
    await expect(page.locator(".tensor-row").first()).toBeVisible();
    await page.locator(".tensor-summary").first().click();
    await expect(page.locator(".tensor-extra").first()).toContainText(
      "artifact",
    );
    await expect
      .poll(async () => (await page.locator(".app-header").boundingBox())!.y)
      .toBe(0);
    await page.screenshot({
      path: testInfo.outputPath(`live-${id}-tensors.png`),
      animations: "disabled",
    });
    await page.getByRole("tab", { name: /^Evidence/ }).click();
    await expect(page.locator(".evidence-item").first()).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath(`live-${id}-evidence.png`),
      animations: "disabled",
    });
    if (isMobile)
      await page
        .getByRole("navigation", { name: "Workspace views" })
        .getByRole("button", { name: "Graph" })
        .click();
    const nested = graph.nodes.find((node) => node.child_graph_id)!;
    await graphNode(page, nested.label);
    await page
      .getByRole("button", { name: `Open ${nested.label}`, exact: true })
      .click();
    await expect(
      page
        .getByRole("navigation", { name: "Graph breadcrumb" })
        .getByRole("button", { name: nested.label }),
    ).toBeVisible();
    const child = await (
      await page.request.get(
        `/api/v1/analyses/${id}/graphs/${encodeURIComponent(nested.child_graph_id!)}`,
      )
    ).json();
    await expect(page.locator(".react-flow__node")).toHaveCount(
      child.nodes.length,
    );
    await expectGraphFramed(page);
    await page
      .getByRole("button", { name: "Readable view", exact: true })
      .click();
    await expectReadableGraph(page);
    await page.screenshot({
      path: testInfo.outputPath(`live-${id}-nested.png`),
      animations: "disabled",
    });
    await page.getByRole("button", { name: "Collapse", exact: true }).click();
    await expect(page.locator(".react-flow__node")).toHaveCount(
      graph.nodes.length,
    );
    for (const label of ["Architecture JSON", "Complete ZIP"]) {
      await page.getByLabel("Export analysis", { exact: true }).click();
      const downloaded = page.waitForEvent("download");
      await page.getByRole("button", { name: label, exact: true }).click();
      expect((await downloaded).suggestedFilename()).toMatch(
        label.endsWith("JSON") ? /\.json$/ : /\.zip$/,
      );
    }
    const geometry = await page.evaluate(() => {
      const siblings = [
        ".app-header",
        ".input-band",
        ".workspace",
        ".mobile-tabs",
      ]
        .map((selector) =>
          document.querySelector(selector)!.getBoundingClientRect(),
        )
        .filter((rect) => rect.height > 0);
      return {
        overflow: document.documentElement.scrollWidth > window.innerWidth,
        overlap: siblings.some(
          (rect, index) =>
            index > 0 && rect.top < siblings[index - 1].bottom - 1,
        ),
      };
    });
    expect(geometry).toEqual({ overflow: false, overlap: false });
  }
  await showQuestions(page, isMobile);
  if (!health.ai_available)
    await expect(page.getByLabel("Ask about model evidence")).toBeDisabled();
  await page.screenshot({
    path: testInfo.outputPath("live-questions.png"),
    animations: "disabled",
  });
  expect(failures).toEqual([]);
});

test("live model submission reaches a stored architecture", async ({
  page,
  isMobile,
}, testInfo) => {
  test.setTimeout(180_000);
  test.skip(
    !process.env.E2E_LIVE_API,
    "Set E2E_LIVE_API=1 with the HTTP service and worker running.",
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("No analysis selected").first()).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("HuggingFaceTB/SmolLM2-360M");
  await expect(
    page.getByRole("button", { name: "Analyze Model", exact: true }),
  ).toBeEnabled({ timeout: 60_000 });
  const createdResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await page
    .getByRole("button", { name: "Analyze Model", exact: true })
    .click();
  const response = await createdResponse;
  expect(response.status()).toBe(202);
  const created = (await response.json()) as AnalysisDetail;
  expect(created.prepared).toBe(false);
  expect(created.id.startsWith("example-")).toBe(false);
  await expect(page).toHaveURL(
    new RegExp(`analysis=${encodeURIComponent(created.id)}`),
  );
  await expect(page.getByTestId("graph-canvas")).toBeVisible({
    timeout: 130_000,
  });
  await expectGraphFramed(page);
  const result = (await (
    await page.request.get(`/api/v1/analyses/${created.id}`)
  ).json()) as AnalysisDetail;
  expect(["completed", "partial"]).toContain(result.status);
  expect(result.model_id).toBe("HuggingFaceTB/SmolLM2-360M");
  expect(result.parameter_count).toBe(361_821_120);
  expect(result.prepared).toBe(false);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    result.model_id,
  );
  await page.screenshot({
    path: testInfo.outputPath("live-created-analysis.png"),
    animations: "disabled",
  });
  if (isMobile)
    await page
      .getByRole("button", { name: "Open recent analyses" })
      .first()
      .click();
  await expect(page.locator(".recent-link.active")).toContainText(
    result.model_id,
  );
  await page.goto("/");
  await expect(page).toHaveURL(
    new RegExp(`analysis=${encodeURIComponent(created.id)}`),
  );
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expectGraphFramed(page);
  await page.screenshot({
    path: testInfo.outputPath("live-restored-analysis.png"),
    animations: "disabled",
  });
  expect(errors).toEqual([]);
});
