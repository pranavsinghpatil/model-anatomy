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
  shape_convention: "tensor_axes" as const,
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

test("GGUF selection submits the exact pinned file rather than guessing a variant", async ({
  page: browser,
}) => {
  const file = "Bonsai-PTQ1_0.gguf";
  const base = analysisFor();
  const job = {
    ...base,
    model_id: "acme/vision-model",
    revision,
    status: "inspected" as const,
    stage: "inspected",
    root_graph_id: null,
    architecture: null,
    parameter_count: null,
  };
  const requests = await mockService(browser, {
    job,
    artifactPages: [
      {
        ...page(0, [file, "Bonsai-F16.gguf"], []),
        tensor_count: 0,
        reason_code: "checkpoint_selection_required",
      },
    ],
  });
  await browser.goto(`/?analysis=${job.id}`);
  await browser
    .getByRole("button", {
      name: `Inspect ${file} (GGUF checkpoint)`,
      exact: true,
    })
    .click();
  await expect
    .poll(() =>
      requests
        .filter((request) => request.method === "POST")
        .map((request) => request.body),
    )
    .toContainEqual({
      model: `https://huggingface.co/acme/vision-model/blob/${revision}/${file}`,
    });
  expect(requests.some((request) => request.path.includes("/graphs/"))).toBe(
    false,
  );
});

test("declared repository profile and nested Safetensors index remain selectable", async ({
  page: browser,
}) => {
  const indexFile = "text_encoder/model.safetensors.index.json";
  const componentFile = "text_encoder/model-00001-of-00002.safetensors";
  const job = {
    ...analysisFor(),
    model_id: "acme/compound-model",
    revision,
    status: "inspected" as const,
    stage: "inspected",
    root_graph_id: null,
    architecture: null,
    parameter_count: null,
  };
  const profilePage = {
    ...page(0, [indexFile, componentFile], []),
    reason_code: "checkpoint_selection_required",
    model_profile: {
      repository_kind: "compound_pipeline",
      declared_task: "text-to-image",
      declared_library: "diffusers",
      config_model_type: null,
      declared_architectures: [],
      formats: ["safetensors"],
      selected_file: null,
      selected_config: "model_index.json",
      architecture_blockers: [
        "native_architecture_unavailable",
        "pipeline_components_require_selection",
      ],
      components: [
        {
          path: "text_encoder",
          library: "transformers",
          declared_class: "CLIPTextModel",
          checkpoint_files: [componentFile, indexFile],
        },
      ],
      metadata_basis:
        "Declared by repository metadata; components are not a verified graph.",
    },
  };
  const requests = await mockService(browser, {
    job,
    artifactPages: [profilePage],
  });
  await browser.goto(`/?analysis=${job.id}`);

  const inspection = browser.getByTestId("artifact-inspection");
  await expect(inspection.getByTestId("model-profile")).toContainText(
    "text-to-image",
  );
  await expect(inspection.getByTestId("model-profile")).toContainText(
    "Declared components",
  );
  await expect(inspection.getByTestId("model-profile")).toHaveClass(
    /artifact-model-profile/,
  );
  await expect(inspection.getByTestId("model-profile")).toContainText(
    "native_architecture_unavailable",
  );
  await expect(inspection.getByTestId("model-profile")).toContainText(
    "pipeline_components_require_selection",
  );
  await expect(
    inspection.getByTestId("checkpoint-selection-callout"),
  ).toContainText("pipeline_components_require_selection");
  await expect(
    inspection.getByRole("button", {
      name: `Inspect ${indexFile} (Safetensors index)`,
      exact: true,
    }),
  ).toBeVisible();
  await inspection
    .getByRole("button", {
      name: `Inspect ${indexFile} (Safetensors index)`,
      exact: true,
    })
    .click();

  await expect
    .poll(() =>
      requests
        .filter((request) => request.method === "POST")
        .map((request) => request.body),
    )
    .toContainEqual({
      model: `https://huggingface.co/acme/compound-model/blob/${revision}/${indexFile}`,
    });
  expect(requests.some((request) => request.path.includes("/graphs/"))).toBe(
    false,
  );
});

test("selected split GGUF reports file elements and native axis order", async ({
  page: browser,
}) => {
  const file = "part-00001-of-00002.gguf";
  const job = {
    ...analysisFor(),
    model_id: "acme/vision-model",
    revision,
    status: "inspected" as const,
    stage: "inspected",
    root_graph_id: null,
    architecture: null,
    parameter_count: null,
  };
  await mockService(browser, {
    job,
    artifactPages: [
      {
        ...page(
          0,
          [file],
          [
            {
              ...tensor(1),
              artifact: `https://huggingface.co/acme/vision-model/resolve/${revision}/${file}`,
              shape_convention: "ggml_dimensions",
              dtype: "GGML_TYPE_0",
              tensor_type_id: 0,
            },
          ],
        ),
        tensor_count: 1,
        file_count: 1,
        reason_code: "gguf_header_only",
        checkpoint_tensor_element_count: null,
        checkpoint_format: "gguf",
        selected_file_tensor_element_count: 589824,
        selected_file_split_metadata: {
          "split.no": 0,
          "split.count": 2,
          "split.tensors.count": 2,
        },
      },
    ],
  });
  await browser.goto(`/?analysis=${job.id}`);
  const view = browser.getByTestId("artifact-inspection");
  await expect(view).toContainText("Selected file elements");
  await expect(view).toContainText("only the selected file");
  await expect(view).toContainText("GGML axis order");
  await expect(view).toContainText("589,824");
  await expect(browser.getByTestId("graph-canvas")).toHaveCount(0);
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
