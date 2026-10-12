import { expect, test } from "@playwright/test";
import type { ArtifactInspectionResponse } from "../src/api/contract";
import { analysisFor, mockService } from "./service";

const revision = "0123456789abcdef0123456789abcdef01234567";
const longTensorName =
  "diffusion_model.transformer_blocks.16.attn.to_out.0.lora_B.default.weight".padEnd(
    384,
    "_",
  );
const longArtifactUrl = `https://huggingface.co/acme/vision-model/resolve/${revision}/${"shard/".repeat(9)}checkpoint-00001-of-00002.safetensors`;
const tensor = (index: number) => ({
  name: index === 0 ? longTensorName : `encoder.layer.${index}.weight`,
  dtype: "F32",
  shape: [768, 768],
  parameter_count: 589824,
  artifact:
    index === 0
      ? longArtifactUrl
      : `https://huggingface.co/acme/vision-model/resolve/${revision}/model-00001-of-00002.safetensors`,
});
const page = (
  offset: number,
  files: string[],
  tensors: ArtifactInspectionResponse["tensors"],
): ArtifactInspectionResponse => ({
  kind: "artifact_inspection",
  reason_code: "architecture_not_supported",
  message:
    "Artifact metadata is available; no architecture graph was produced.",
  revision,
  files,
  file_count: 3,
  tensors,
  tensor_count: 51,
  checkpoint_tensor_element_count: 589824,
  checkpoint_available: true,
  source_config: null,
  evidence: [
    {
      source: "safetensors_header",
      kind: "direct",
      locator: `https://huggingface.co/acme/vision-model/resolve/${revision}/model-00001-of-00002.safetensors`,
      details: {},
    },
  ],
  offset,
  limit: 50,
});

test("inspected artifacts stay graph-free and paginate verified metadata", async ({
  page: browser,
  isMobile,
}) => {
  const base = analysisFor();
  const job = {
    ...base,
    model_id: "acme/vision-model",
    architecture: null,
    revision,
    status: "inspected" as const,
    stage: "inspected",
    parameter_count: null,
    root_graph_id: null,
    dimensions: { hidden: null, heads: null, layers: null, vocabulary: null },
    inspection: {
      kind: "artifact_inspection" as const,
      reason_code: "architecture_not_supported",
      message:
        "Artifact metadata is available; no architecture graph was produced.",
      file_count: 3,
      tensor_count: 51,
      checkpoint_tensor_element_count: 589824,
      checkpoint_available: true,
    },
  };
  const requests = await mockService(browser, {
    job,
    artifactPages: [
      page(
        0,
        [
          "config.json",
          "model.safetensors.index.json",
          "model-00001-of-00002.safetensors",
        ],
        Array.from({ length: 50 }, (_, index) => tensor(index)),
      ),
      page(50, [], [tensor(50)]),
    ],
  });
  await browser.goto(`/?analysis=${job.id}`);

  const inspection = browser.getByTestId("artifact-inspection");
  await expect(
    inspection.getByRole("heading", { name: "Architecture unavailable" }),
  ).toBeVisible();
  await expect(
    browser
      .getByRole("main")
      .getByRole("status", { name: "Analysis status: inspected" }),
  ).toBeVisible();
  await expect(
    browser.getByText("Artifacts inspected", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    inspection.getByText("architecture_not_supported"),
  ).toBeVisible();
  await expect(inspection.getByTestId("artifact-revision")).toHaveText(
    revision,
  );
  await expect(inspection.getByText("Checkpoint", { exact: true })).toHaveCount(
    0,
  );
  await expect(browser.getByTestId("graph-canvas")).toHaveCount(0);
  await expect(browser.locator(".model-metrics")).toHaveCount(0);
  await expect(
    inspection.getByRole("link", { name: "config.json" }),
  ).toHaveAttribute(
    "href",
    `https://huggingface.co/acme/vision-model/blob/${revision}/config.json`,
  );

  const tensorRow = inspection
    .getByRole("row")
    .filter({ hasText: longTensorName });
  const tensorLink = tensorRow.getByRole("link");
  await expect(tensorLink).toHaveAttribute("href", longArtifactUrl);
  await expect(inspection.getByText("0 of 3")).toHaveCount(0);

  if (isMobile) {
    await expect(
      browser
        .getByRole("navigation", { name: "Workspace views" })
        .getByRole("button", { name: "Questions" }),
    ).toBeDisabled();
  } else {
    await expect(
      browser.getByRole("button", {
        name: "Questions unavailable for artifact inspection",
      }),
    ).toBeDisabled();
    await browser.getByRole("button", { name: "Open analysis panel" }).click();
    await expect(
      browser.getByRole("tab", { name: "Questions" }),
    ).toBeDisabled();
  }
  expect(requests.some((request) => request.path.includes("/graphs/"))).toBe(
    false,
  );
  expect(requests.some((request) => request.path.includes("/questions"))).toBe(
    false,
  );

  await inspection.getByRole("button", { name: "Next artifact page" }).click();
  await expect(inspection.getByText("encoder.layer.50.weight")).toBeVisible();
  await expect(inspection.getByText("0 of 3")).toBeVisible();
  await expect(
    inspection.getByRole("button", { name: "Previous artifact page" }),
  ).toBeEnabled();
  expect(
    requests.some((request) =>
      request.path.includes("/artifacts?offset=50&limit=50"),
    ),
  ).toBe(true);
  expect(requests.some((request) => request.path.includes("/graphs/"))).toBe(
    false,
  );
  if (isMobile) {
    const metrics = await inspection
      .locator(".artifact-table-wrap")
      .evaluate((wrap) => {
        const table = wrap.querySelector("table")!;
        const row = table.tBodies[0].rows[0];
        const cells = Array.from(
          row.cells,
          (cell) => cell.getBoundingClientRect().width,
        );
        return {
          clientWidth: wrap.clientWidth,
          scrollWidth: wrap.scrollWidth,
          rowHeight: row.getBoundingClientRect().height,
          columns: cells.map(
            (width) => width / table.getBoundingClientRect().width,
          ),
          tableWidth: table.getBoundingClientRect().width,
        };
      });
    expect(metrics.scrollWidth).toBeGreaterThan(metrics.clientWidth);
    expect(metrics.tableWidth).toBeGreaterThanOrEqual(1000);
    expect(metrics.columns[0]).toBeGreaterThan(0.35);
    expect(metrics.columns[1]).toBeGreaterThan(0.1);
    expect(metrics.columns[2]).toBeGreaterThan(0.06);
    expect(metrics.columns[3]).toBeGreaterThan(0.1);
    expect(metrics.columns[4]).toBeGreaterThan(0.27);
    expect(metrics.rowHeight).toBeLessThan(150);
  }
});

test("inspected report keeps supported exports and omits Mermaid", async ({
  page: browser,
}) => {
  const base = analysisFor();
  const job = {
    ...base,
    architecture: null,
    status: "inspected" as const,
    parameter_count: null,
    root_graph_id: null,
    inspection: {
      kind: "artifact_inspection" as const,
      reason_code: "architecture_not_supported",
      message: "Artifact metadata is available.",
      file_count: 0,
      tensor_count: 0,
      checkpoint_tensor_element_count: null,
      checkpoint_available: false,
    },
  };
  const requests = await mockService(browser, {
    job,
    artifactPages: [page(0, [], [])],
  });
  await browser.goto(`/?analysis=${job.id}`);
  await expect(browser.getByTestId("artifact-inspection")).toBeVisible();

  const formats = [
    ["Report JSON", "json"],
    ["Complete ZIP", "zip"],
    ["Markdown", "markdown"],
    ["Diagnostics", "diagnostics"],
    ["Audit", "audit"],
  ] as const;
  for (const [label, format] of formats) {
    await browser.getByLabel("Export analysis", { exact: true }).click();
    const download = browser.waitForEvent("download");
    await browser.getByRole("button", { name: label, exact: true }).click();
    await download;
    expect(
      requests.some((request) => request.path.endsWith(`/exports/${format}`)),
    ).toBe(true);
  }
  await browser.getByLabel("Export analysis", { exact: true }).click();
  await expect(
    browser.getByRole("button", { name: "Mermaid", exact: true }),
  ).toHaveCount(0);
});
