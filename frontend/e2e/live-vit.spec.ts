import { expect, test } from "@playwright/test";
import type {
  AnalysisDetail,
  ArchitectureGraph,
  GraphIndex,
} from "../src/api/contract";

test("live ViT native module hierarchy and checkpoint tensor inspection", async ({
  page,
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
  const submitted = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await submit.click();
  const response = await submitted;
  expect(response.status()).toBe(202);
  const created = (await response.json()) as AnalysisDetail;
  const path = `/api/v1/analyses/${created.id}`;
  await expect
    .poll(async () => (await (await page.request.get(path)).json()).status, {
      timeout: 130_000,
    })
    .toBe("completed");
  const result = (await (
    await page.request.get(path)
  ).json()) as AnalysisDetail;
  expect(result.parameter_count).toBe(86_567_656);
  expect(result.input_names).toContain("pixel_values");
  expect(result.partial_reasons).toEqual([]);
  expect(result.quality).toMatchObject({
    mode: "static_module_architecture",
    checkpoint_parameter_coverage: 1,
    unknown_nodes: 0,
    unmapped_checkpoint_tensors: 0,
    forward_executed: false,
    runtime_trace_available: false,
    architecture_checks: {
      module_inventory: true,
      parameter_mapping: true,
      native_architecture: true,
      module_topology: true,
    },
  });
  await page.getByRole("link", { name: "Report", exact: true }).click();
  const checks = page.getByTestId("report-checks");
  await expect(checks).toBeVisible();
  await expect(checks).toContainText("Native module inventory");
  await expect(checks).toContainText("Native architecture");
  await expect(checks).not.toContainText("Tensor shape coverage");
  await expect(page.getByTestId("analysis-scope")).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("vit-report-checks.png"),
    animations: "disabled",
  });
  const report = await (await page.request.get(`${path}/exports/json`)).json();
  expect(report.metadata.engine_version).toBe(result.engine_version);
  expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
  expect(report.metadata.analysis_report.completion.status).toBe("complete");
  const index = (await (
    await page.request.get(`${path}/graph-index`)
  ).json()) as GraphIndex;
  const root = (await (
    await page.request.get(`${path}/graphs/${result.root_graph_id}`)
  ).json()) as ArchitectureGraph;
  expect(root.nodes.length).toBeGreaterThan(0);
  expect(index.graphs.length).toBeGreaterThan(1);
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible();

  async function inspectModule(modulePath: string) {
    const back = page.getByRole("button", {
      name: "Back to graph",
      exact: true,
    });
    if (await back.isVisible()) await back.click();
    const components = page.getByRole("button", {
      name: "Open components",
      exact: true,
    });
    if (await components.isVisible()) await components.click();
    await page.getByRole("button", { name: "Model tree", exact: true }).click();
    await page
      .getByLabel("Find across model", { exact: true })
      .fill(modulePath);
    const match = page.locator(".model-component-tree .component-select");
    await expect(match).toHaveCount(1);
    await match.click();
    await expect(page.getByTestId("inspector")).toBeVisible();
    return page.getByTestId("inspector");
  }
  const classifier = index.nodes.find((node) => node.kind === "classifier")!;
  expect(classifier.module_path).toBeTruthy();
  expect(classifier.child_graph_id).toBeNull();
  let panel = await inspectModule(classifier.module_path!);
  await expect(
    panel.getByRole("region", { name: "Recorded module properties" }),
  ).toContainText("Linear");
  await expect(
    panel.getByRole("button", { name: "Open nested graph", exact: true }),
  ).toHaveCount(0);
  await panel.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(panel.locator(".tensor-row")).toHaveCount(2);
  expect(report.tensors[`${classifier.module_path}.weight`].shape).toEqual([
    1000, 768,
  ]);
  expect(report.tensors[`${classifier.module_path}.bias`].shape).toEqual([
    1000,
  ]);
  await panel.getByRole("tab", { name: /^Evidence/ }).click();
  await expect(panel.locator(".evidence-item").first()).toBeVisible();

  const patch = index.nodes.find((node) =>
    node.module_path?.endsWith("patch_embeddings.projection"),
  )!;
  expect(patch.child_graph_id).toBeNull();
  panel = await inspectModule(patch.module_path!);
  await panel.getByRole("tab", { name: /^Component/ }).click();
  await expect(
    panel.getByRole("region", { name: "Recorded module properties" }),
  ).toContainText("Conv2d");
  expect(report.tensors[`${patch.module_path}.weight`].shape).toEqual([
    768, 3, 16, 16,
  ]);
  await page.screenshot({
    path: testInfo.outputPath("vit-patch-module.png"),
    animations: "disabled",
  });
  await page.reload();
  await expect(page.getByTestId("inspector")).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Recorded module properties" }),
  ).toContainText("Conv2d");
  const back = page.getByRole("button", { name: "Back to graph", exact: true });
  if (await back.isVisible()) await back.click();
  await expect(
    page.getByRole("navigation", { name: "Graph path" }),
  ).toContainText("ViT");
  await page.getByLabel("Export inspection", { exact: true }).click();
  const download = page.waitForEvent("download");
  await page
    .getByRole("menuitem", { name: "Complete ZIP", exact: true })
    .click();
  expect((await download).suggestedFilename()).toMatch(/\.zip$/);
  expect(errors).toEqual([]);
});
