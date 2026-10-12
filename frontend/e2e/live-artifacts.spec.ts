import { expect, test } from "@playwright/test";
import type { AnalysisDetail } from "../src/api/contract";

const cases = [
  {
    model: "Aleph-Alpha/Kolibri-1",
    revision: "35bc4d3be745502227a67247de77d70e691614ee",
    reason: "metadata_budget_exceeded",
    tensors: 0,
  },
  {
    model: "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA",
    revision: "0d91c4dfa22b3f5aa89c020eadc30d8f42cb51c0",
    reason: "adapter_requires_base_model",
    tensors: 384,
  },
];

for (const item of cases) {
  test(`live artifacts: ${item.model}`, async ({ page, isMobile }, info) => {
    test.skip(
      !process.env.E2E_LIVE_API,
      "Requires the real local API and worker.",
    );
    test.setTimeout(180_000);
    const errors: string[] = [];
    const graphRequests: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (request) => {
      if (request.url().includes("/graphs/")) graphRequests.push(request.url());
    });
    await page.goto("/app");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill(item.model);
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeEnabled({ timeout: 30_000 });
    const submitted = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        response.url().endsWith("/api/v1/analyses"),
    );
    await page
      .getByRole("button", { name: "Inspect model", exact: true })
      .click();
    const response = await submitted;
    expect(response.status()).toBe(202);
    const created = (await response.json()) as AnalysisDetail;
    await expect(page).toHaveURL(new RegExp(`/app/analyses/${created.id}`));
    await page.getByRole("link", { name: "Artifacts", exact: true }).click();
    const view = page.getByTestId("artifact-inspection");
    await expect(view).toBeVisible({ timeout: 130_000 });
    await expect(view).toContainText(item.reason);
    await expect(page.getByTestId("artifact-revision")).toHaveText(
      item.revision,
    );
    await expect(page.getByTestId("artifact-tensor-count")).toHaveText(
      String(item.tensors),
    );
    await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
    expect(graphRequests).toEqual([]);
    const report = await (
      await page.request.get(`/api/v1/analyses/${created.id}/exports/json`)
    ).json();
    expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
    expect(report.root_graph.nodes).toEqual([]);
    expect(report.root_graph.edges).toEqual([]);
    const detail = (await (
      await page.request.get(`/api/v1/analyses/${created.id}`)
    ).json()) as AnalysisDetail;
    expect(detail.status).toBe("inspected");
    expect(detail.parameter_count).toBeNull();
    expect(detail.root_graph_id).toBeNull();
    if (item.tensors) {
      await view.getByRole("tab", { name: /^Tensors/ }).click();
      await expect(view.locator("tbody tr")).toHaveCount(50);
      await page
        .getByRole("button", { name: "Next tensor page", exact: true })
        .click();
      await expect(view).toContainText("Page 2");
    }
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    );
    expect(overflow).toBe(false);
    await page.screenshot({
      path: info.outputPath("live-artifact-inspection.png"),
      animations: "disabled",
    });
    expect(errors).toEqual([]);
  });
}
