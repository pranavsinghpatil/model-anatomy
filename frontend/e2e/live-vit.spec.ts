import { expect, test } from "@playwright/test";
import type { AnalysisDetail, ArchitectureGraph } from "../src/api/contract";
import {
  expectReadableGraph,
  graphNode,
  openNestedGraph,
  showDetails,
} from "./helpers";

test("live ViT patches, CLS selection and tensor inspection", async ({
  page,
  isMobile,
}, testInfo) => {
  test.setTimeout(180_000);
  test.skip(
    !process.env.E2E_LIVE_API,
    "Requires the real API and worker on port 8001.",
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill(
      "https://huggingface.co/google/vit-base-patch16-224/blob/10064cc17b69d623a025c9e90e11230cc8ba6967/model.safetensors",
    );
  const submit = page.getByRole("button", {
    name: "Inspect model",
    exact: true,
  });
  await expect(submit).toBeEnabled({ timeout: 60_000 });
  const createdResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await submit.click();
  const response = await createdResponse;
  expect(response.status()).toBe(202);
  const created = (await response.json()) as AnalysisDetail;
  await expect(page).toHaveURL(new RegExp(`/app/analyses/${created.id}`));
  const path = `/api/v1/analyses/${created.id}`;
  await expect
    .poll(
      async () =>
        (
          (await page.request
            .get(path)
            .then((response) => response.json())) as AnalysisDetail
        ).status,
      { timeout: 130_000 },
    )
    .toBe("completed");
  const result = (await (
    await page.request.get(path)
  ).json()) as AnalysisDetail;
  expect(result.status).toBe("completed");
  expect(result.parameter_count).toBe(86_567_656);
  expect(result.warnings).toEqual([]);
  expect(result.input_names).toEqual(["pixel_values"]);
  await expect(page.getByTestId("model-metric-vocabulary")).toHaveCount(0);
  await expect(page.getByTestId("report-checks")).toHaveCount(0);
  await expect(page.getByTestId("analysis-scope")).toHaveCount(0);
  expect(result.partial_reasons).toEqual([]);
  expect(result.limitations?.length ?? 0).toBeGreaterThan(0);
  expect(result.quality).toMatchObject({
    semantic_coverage: 1,
    module_semantic_coverage: 1,
    checkpoint_parameter_coverage: 1,
    tensor_shape_coverage: 1,
    unknown_nodes: 0,
    unmapped_checkpoint_tensors: 0,
    forward_executed: false,
    runtime_trace_available: false,
  });
  await page.getByRole("link", { name: "Report", exact: true }).click();
  const checks = page.getByTestId("report-checks");
  await expect(checks).toBeVisible();
  await expect(checks.locator("dd").filter({ hasText: "100.0%" })).toHaveCount(
    4,
  );
  const scope = page.getByTestId("analysis-scope");
  await expect(scope).toBeVisible();
  const identity = page.locator(".report-section").filter({
    has: page.getByRole("heading", { name: "Artifact identity" }),
  });
  await expect(identity).toContainText(result.engine_version!);
  await page.screenshot({
    path: testInfo.outputPath("vit-report-checks.png"),
    animations: "disabled",
  });
  const report = await (await page.request.get(`${path}/exports/json`)).json();
  expect(report.metadata.engine_version).toMatch(/^\d{4}\.\d{2}\.\d{2}\.\d+$/);
  expect(report.metadata.engine_version).toBe(result.engine_version);
  expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible({
    timeout: 130_000,
  });
  const root = (await (
    await page.request.get(`${path}/graphs/${result.root_graph_id}`)
  ).json()) as ArchitectureGraph;
  const selection = root.nodes.find((node) => node.kind === "selection")!;
  expect(selection.inputs[0].shape).toEqual(["B", 197, 768]);
  expect(selection.outputs[0].shape).toEqual(["B", 768]);
  await expectReadableGraph(page);
  const classifier = root.nodes.find((node) => node.kind === "classifier")!;
  await graphNode(page, classifier.label);
  await showDetails(page, isMobile);
  await page.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(page.locator(".tensor-row")).toHaveCount(2);
  await page.screenshot({
    path: testInfo.outputPath("vit-classifier.png"),
    animations: "disabled",
  });
  if (isMobile)
    await page.getByRole("button", { name: "Back to graph" }).click();
  let graph = root;
  for (const kind of ["backbone", "embeddings", "patch_embedding"]) {
    const node = graph.nodes.find((item) => item.kind === kind)!;
    expect(node.child_graph_id).toBeTruthy();
    await graphNode(page, node.label);
    await openNestedGraph(page, node.label);
    graph = (await (
      await page.request.get(`${path}/graphs/${node.child_graph_id}`)
    ).json()) as ArchitectureGraph;
    await expect(page.locator(".react-flow__node")).toHaveCount(
      graph.nodes.length,
    );
  }
  expect(
    graph.nodes.find((node) => node.metadata.method === "transpose")!.outputs[0]
      .shape,
  ).toEqual(["B", 196, 768]);
  await expectReadableGraph(page);
  await page.screenshot({
    path: testInfo.outputPath("vit-patches.png"),
    animations: "disabled",
  });
  await page.getByLabel("Export analysis", { exact: true }).click();
  const download = page.waitForEvent("download");
  await page
    .getByRole("menuitem", { name: "Complete ZIP", exact: true })
    .click();
  expect((await download).suggestedFilename()).toMatch(/\.zip$/);
  expect(errors).toEqual([]);
});
