import { expect, test } from "@playwright/test";
import type { ArtifactInspectionResponse } from "../src/api/contract";
import { analysisFor, mockService } from "./service";
import { openAnalysis } from "./helpers";

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
  source_config: { model_type: "vision-model", hidden_size: 768 },
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
  await openAnalysis(browser, job.id, "artifacts");
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
  await inspection
    .getByText("Repository profile and declared components", { exact: true })
    .click();
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
  await openAnalysis(browser, job.id, "artifacts");
  const view = browser.getByTestId("artifact-inspection");
  await view.getByText("Split checkpoint metadata", { exact: true }).click();
  await view.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(view).toContainText("Selected file elements");
  await expect(view).toContainText("only the explicitly selected file");
  await expect(view).toContainText("GGML native axis order");
  await expect(
    view.getByTestId("artifact-inspection-stats").locator("dd").nth(2),
  ).toHaveAttribute("title", "589,824");
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
  await openAnalysis(browser, job.id, "artifacts");

  const inspection = browser.getByTestId("artifact-inspection");
  await expect(
    inspection.getByRole("heading", { name: "Artifacts" }),
  ).toBeVisible();
  await inspection.getByText("Inspection reason", { exact: true }).click();
  await expect(
    inspection.getByText("architecture_not_supported"),
  ).toBeVisible();
  await inspection.locator(".artifact-inspection-identity summary").click();
  await expect(inspection.getByTestId("artifact-revision")).toHaveText(
    revision,
  );
  await expect(inspection.getByText("Checkpoint", { exact: true })).toHaveCount(
    0,
  );
  await expect(browser.getByTestId("graph-canvas")).toHaveCount(0);
  await expect(browser.locator(".model-metrics")).toHaveCount(0);
  await expect(browser.getByTestId("report-checks")).toHaveCount(0);
  await expect(browser.getByTestId("analysis-scope")).toHaveCount(0);
  await expect(
    browser.getByRole("link", { name: "Overview", exact: true }),
  ).toBeVisible();
  await expect(
    browser.getByRole("link", { name: "Architecture", exact: true }),
  ).toBeVisible();
  await expect(
    browser.getByRole("link", { name: "Artifacts", exact: true }),
  ).toBeVisible();
  await expect(
    browser.getByRole("link", { name: "Report", exact: true }),
  ).toBeVisible();
  await expect(
    inspection.getByRole("link", { name: "config.json" }),
  ).toHaveAttribute(
    "href",
    `https://huggingface.co/acme/vision-model/blob/${revision}/config.json`,
  );

  const filesTab = inspection.getByRole("tab", { name: /^Files/ });
  await expect(filesTab).toHaveAttribute("data-state", "active");
  await expect(inspection.getByText("Source configuration")).toBeVisible();
  await expect(inspection.getByText("Artifact evidence (1)")).toBeVisible();
  await inspection.getByRole("tab", { name: /^Tensors/ }).click();
  const tensorLink = inspection
    .getByRole("row")
    .filter({ hasText: longTensorName })
    .getByRole("link");
  await expect(tensorLink).toHaveAttribute("href", longArtifactUrl);
  await expect(inspection.getByText("0 of 3")).toHaveCount(0);

  // Long names and source URLs must scroll within the table at every supported
  // width; they must never push the surrounding workspace beyond the viewport.
  for (const width of [320, 390, 768, 1024, 1440, 1920]) {
    await browser.setViewportSize({ width, height: 900 });
    for (const theme of ["Light", "Dark"]) {
      await browser.getByRole("button", { name: "Theme", exact: true }).click();
      await browser
        .getByRole("menuitemradio", { name: theme, exact: true })
        .click();
      await expect(tensorLink).toHaveAttribute("href", longArtifactUrl);
      await expect
        .poll(
          () =>
            browser.evaluate(
              () =>
                Math.max(
                  document.body.scrollWidth,
                  document.documentElement.scrollWidth,
                ) <= innerWidth,
            ),
          `${width}px ${theme} long tensor`,
        )
        .toBe(true);
    }
  }
  await browser.setViewportSize(
    isMobile ? { width: 390, height: 844 } : { width: 1440, height: 960 },
  );

  expect(requests.some((request) => request.path.includes("/graphs/"))).toBe(
    false,
  );
  expect(requests.some((request) => request.path.includes("/questions"))).toBe(
    false,
  );

  await inspection.getByRole("tab", { name: /^Tensors/ }).click();
  let releaseTensorPage!: () => void;
  let signalTensorRequest!: () => void;
  const tensorPageGate = new Promise<void>((resolve) => {
    releaseTensorPage = resolve;
  });
  const tensorRequestStarted = new Promise<void>((resolve) => {
    signalTensorRequest = resolve;
  });
  const tensorPageRoute = /\/api\/v1\/analyses\/job-test\/artifacts\?offset=50/;
  await browser.route(tensorPageRoute, async (route) => {
    signalTensorRequest();
    await tensorPageGate;
    await route.fallback();
  });
  await inspection.getByRole("button", { name: "Next tensor page" }).click();
  await tensorRequestStarted;
  try {
    await expect(
      inspection.getByText("1–50 of 51", { exact: true }),
    ).toBeVisible();
  } finally {
    releaseTensorPage();
  }
  await browser.unroute(tensorPageRoute);
  await expect(inspection.getByText("encoder.layer.50.weight")).toBeVisible();
  await expect(
    inspection.getByText("51–51 of 51", { exact: true }),
  ).toBeVisible();
  await expect(
    inspection.getByRole("button", { name: "Previous tensor page" }),
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
        return {
          clientWidth: wrap.clientWidth,
          scrollWidth: wrap.scrollWidth,
        };
      });
    expect(metrics.scrollWidth).toBeGreaterThan(metrics.clientWidth);
    expect(metrics.clientWidth).toBeGreaterThan(0);
    const pageWidth = await browser.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(pageWidth).toBeLessThanOrEqual(1);
  }
  await inspection.getByRole("tab", { name: /^Files/ }).click();
  await expect(
    inspection.getByRole("link", { name: "config.json" }),
  ).toBeVisible();
  await expect(
    inspection.getByRole("button", { name: "Next file page" }),
  ).toBeDisabled();
});

test("inspected report keeps supported exports and omits Mermaid", async ({
  page: browser,
}) => {
  const base = analysisFor();
  const job = {
    ...base,
    architecture: null,
    engine_version: "2030.01.01.1",
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
  await openAnalysis(browser, job.id, "report");
  const reportChecks = browser.getByTestId("report-checks");
  const scope = browser.getByTestId("analysis-scope");
  await expect(reportChecks).toBeVisible();
  await expect(scope).toBeVisible();
  await expect(reportChecks.locator("details")).toHaveCount(0);
  await expect(scope.locator("details")).toHaveCount(0);
  const identity = browser.locator(".report-identity-details");
  await identity.locator("summary").click();
  await expect(
    identity.getByText("2030.01.01.1", { exact: true }),
  ).toBeVisible();

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
    await browser.getByRole("menuitem", { name: label, exact: true }).click();
    await download;
    expect(
      requests.some((request) => request.path.endsWith(`/exports/${format}`)),
    ).toBe(true);
  }
  await browser.getByLabel("Export analysis", { exact: true }).click();
  await expect(
    browser.getByRole("menuitem", { name: "Mermaid", exact: true }),
  ).toHaveCount(0);
});
