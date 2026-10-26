import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type {
  AnalysisDetail,
  ArchitectureGraph,
  NodeDetail,
} from "../src/api/contract";
import { nodeTitle } from "../src/graph/presentation";
import { graphNode } from "./helpers";

const revision = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const reference = `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/${revision}`;

test("real pinned inspection, tensor evidence, export and cached reopen", async ({
  page,
}, testInfo) => {
  test.skip(
    !process.env.UX_LIVE_API,
    "Set UX_LIVE_API=1 with an actual API and worker; no prepared reports are used.",
  );
  test.setTimeout(240000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // Explicit server refresh forces fresh metadata inspection even if another
  // local visitor already inspected this pinned model. Browser cookies are shared
  // with page.request; the second submission exercises the normal UI cache path.
  await page.goto("/app");
  const fresh = await page.request.post("/api/v1/analyses", {
    data: { model: reference, refresh: true },
  });
  expect(fresh.status()).toBe(202);
  const created = (await fresh.json()) as AnalysisDetail;
  await page.goto(`/app/analyses/${created.id}?view=architecture&awaiting=1`);
  let inspected: AnalysisDetail = created;
  await expect
    .poll(
      async () => {
        const response = await page.request.get(
          `/api/v1/analyses/${created.id}`,
        );
        expect(response.ok()).toBe(true);
        inspected = (await response.json()) as AnalysisDetail;
        return inspected.status;
      },
      { timeout: 180000, intervals: [1000, 2000, 3000] },
    )
    .toBe("completed");
  expect(inspected.cached).toBe(false);
  expect(inspected.model_id).toBe("HuggingFaceTB/SmolLM2-135M");
  expect(inspected.revision).toBe(revision);
  expect(inspected.parameter_count).toBe(134515008);
  await expect(page.getByTestId("graph-canvas")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.getByLabel("Inspection outcome")).toContainText(
    "Architecture ready",
  );
  const graphResponse = await page.request.get(
    `/api/v1/analyses/${created.id}/graphs/${inspected.root_graph_id}`,
  );
  expect(graphResponse.ok()).toBe(true);
  const graph = (await graphResponse.json()) as ArchitectureGraph;
  let component: ArchitectureGraph["nodes"][number] | undefined;
  for (const node of graph.nodes) {
    const response = await page.request.get(
      `/api/v1/analyses/${created.id}/nodes/${graph.id}/${node.id}`,
    );
    expect(response.ok()).toBe(true);
    const detail = (await response.json()) as NodeDetail;
    if (detail.tensor_count > 0 && detail.evidence.length > 0) {
      component = node;
      break;
    }
  }
  if (!component)
    throw new Error(
      "Real graph must expose a component with owned tensors and evidence.",
    );
  await graphNode(page, nodeTitle(component));
  const inspector = page.getByTestId("inspector");
  await expect(
    inspector.getByRole("heading", { name: nodeTitle(component), exact: true }),
  ).toBeVisible();
  await inspector.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(
    inspector.getByRole("heading", { name: "Owned tensors", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Export inspection", exact: true })
    .click();
  const downloading = page.waitForEvent("download");
  await page
    .getByRole("menuitem", { name: "Report JSON", exact: true })
    .click();
  const download = await downloading;
  const file = await download.path();
  if (!file)
    throw new Error("The browser must expose the actual downloaded report.");
  const report = JSON.parse(await readFile(file, "utf8"));
  expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
  await expect(page.getByTestId("feedback-toast")).toContainText(
    "Download requested: Report JSON",
  );
  await page.screenshot({
    path: testInfo.outputPath("real-pinned-inspection.png"),
    animations: "disabled",
  });
  await page.getByRole("link", { name: "New inspection", exact: true }).click();
  await page.getByLabel("Hugging Face model", { exact: true }).fill(reference);
  const submitted = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await page
    .getByRole("button", { name: /^(Inspect model|Open saved inspection)$/ })
    .click();
  const cachedResponse = await submitted;
  expect(cachedResponse.status()).toBe(202);
  const cached = (await cachedResponse.json()) as AnalysisDetail;
  expect(cached.id).not.toBe(created.id);
  expect(cached.cached).toBe(true);
  expect(cached.revision).toBe(revision);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  expect(errors).toEqual([]);
  await testInfo.attach("real-inspection-evidence", {
    contentType: "application/json",
    body: JSON.stringify({
      model: inspected.model_id,
      revision,
      parameter_count: inspected.parameter_count,
      cold_id: created.id,
      cold_cached: inspected.cached,
      reopened_id: cached.id,
      reopened_cached: cached.cached,
      weight_payload_bytes_read:
        report.metadata.artifacts.weight_payload_bytes_read,
    }),
  });
});
