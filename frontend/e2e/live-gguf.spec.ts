import { expect, test } from "@playwright/test";
import type { AnalysisDetail } from "../src/api/contract";

test("live Bonsai GGUF selection exposes verified tensors without inventing a graph", async ({
  page,
}, info) => {
  test.skip(
    !process.env.E2E_LIVE_API,
    "Requires the real local API and worker.",
  );
  test.setTimeout(240_000);
  const model = "prism-ml/Ternary-Bonsai-2-27B-gguf";
  const revision = "b072e1d3b35a0a630cece372c2127528e0994386";
  const file = "Ternary-Bonsai-2-27B-PTQ1_0.gguf";
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/app");
  await page.getByLabel("Hugging Face model", { exact: true }).fill(model);
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled({ timeout: 30_000 });
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  const inspection = page.getByTestId("artifact-inspection");
  await expect(inspection).toContainText("checkpoint_selection_required", {
    timeout: 130_000,
  });
  await expect(page.getByTestId("artifact-revision")).toHaveText(revision);
  const submitted = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await page
    .getByRole("button", {
      name: `Inspect ${file} (GGUF checkpoint)`,
      exact: true,
    })
    .click();
  const response = await submitted;
  expect(response.status()).toBe(202);
  const job = (await response.json()) as AnalysisDetail;
  await expect(page).toHaveURL(new RegExp(`/app/analyses/${job.id}`));
  await page.getByRole("link", { name: "Artifacts", exact: true }).click();
  const artifactsTab = page.getByRole("tab", { name: /^Tensors/ });
  await artifactsTab.click();
  await expect(inspection.locator("tbody tr")).toHaveCount(50, {
    timeout: 130_000,
  });
  const report = await (
    await page.request.get(`/api/v1/analyses/${job.id}/exports/json`)
  ).json();
  expect(report.metadata.revision).toBe(revision);
  expect(Object.keys(report.tensors).length).toBe(851);
  expect(report.root_graph.nodes).toEqual([]);
  expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
  expect(report.metadata.artifacts.selected_file_scope).toBe(
    "single_selected_gguf_file",
  );
  await expect(inspection).toContainText("Selected file elements");
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await page.screenshot({
    path: info.outputPath("live-gguf.png"),
    animations: "disabled",
  });
  await inspection.locator("table").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: info.outputPath("live-gguf-tensors.png"),
    animations: "disabled",
  });
  expect(errors).toEqual([]);
});
