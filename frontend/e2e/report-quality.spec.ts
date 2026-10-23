import { expect, test } from "@playwright/test";
import { openAnalysis } from "./helpers";
import { analysisFor, mockService } from "./service";

test("overview preserves staged head dimensions from the current API", async ({
  page,
}) => {
  await mockService(page, {
    job: {
      ...analysisFor(),
      architecture: "SwinForImageClassification",
      input_names: ["pixel_values"],
      dimensions: {
        hidden: null,
        heads: [3, 6, 12, 24],
        layers: 4,
        vocabulary: null,
      },
    },
  });
  await openAnalysis(page, "job-test", "overview");
  const heads = page.getByTestId("model-metric-heads");
  await expect(heads).toContainText("Heads per stage");
  await expect(heads).toContainText("3 / 6 / 12 / 24");
});

test("selected report status stays consistent while recent jobs are still stale", async ({
  page,
}) => {
  const job = {
    ...analysisFor("job-test"),
    status: "completed" as const,
    stage: "completed",
  };
  await mockService(page, {
    job,
    recent: [{ ...job, status: "running", stage: "analyzing_components" }],
  });
  await openAnalysis(page, job.id, "report");
  await expect(page.getByTestId("report-checks")).toBeVisible();
  const selected = page.locator(".workspace-header");
  await expect(
    selected.getByRole("status", { name: "Analysis status: completed" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Switch model" }).click();
  const selectedRecent = page
    .getByRole("menuitem")
    .filter({ hasText: job.model_id });
  await expect(selectedRecent).toContainText("completed");
  await expect(selectedRecent).not.toContainText("running");
  await page.keyboard.press("Escape");
  await expect(
    selected.getByRole("status", { name: "Analysis status: completed" }),
  ).toBeVisible();
});

test("text report shows established vocabulary and keeps missing dimensions unknown", async ({
  page,
}) => {
  const base = analysisFor("job-test");
  const analysis = {
    ...base,
    dimensions: { ...base.dimensions, heads: null },
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "overview");
  await expect(page.locator(".model-metrics")).toBeVisible();

  const metrics = page.locator(".model-metrics");
  await expect(metrics.getByTestId("model-metric-vocabulary")).toContainText(
    "50,257",
  );
  await expect(
    metrics.getByTestId("model-metric-heads").locator("dd"),
  ).toHaveText("Unknown");
});

test("vision report omits text vocabulary", async ({ page }) => {
  const base = analysisFor("job-test");
  const analysis = {
    ...base,
    architecture: "ViTForImageClassification",
    dimensions: { hidden: 768, heads: 12, layers: 12, vocabulary: null },
    input_names: ["pixel_values"],
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "overview");
  await expect(page.locator(".model-metrics")).toBeVisible();

  const metrics = page.locator(".model-metrics");
  await expect(metrics.getByTestId("model-metric-hidden")).toBeVisible();
  await expect(metrics.getByTestId("model-metric-vocabulary")).toHaveCount(0);
});

test("text input evidence establishes vocabulary for classification wrappers", async ({
  page,
}) => {
  const analysis = {
    ...analysisFor("job-test"),
    architecture: "BertForSequenceClassification",
    dimensions: { hidden: 768, heads: 12, layers: 12, vocabulary: 30522 },
    input_names: ["input_ids"],
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "overview");
  await expect(page.locator(".model-metrics")).toBeVisible();
  await expect(
    page.locator(".model-metrics").getByTestId("model-metric-vocabulary"),
  ).toContainText("30,522");
});

test("text input evidence keeps an unreported vocabulary honestly unknown", async ({
  page,
}) => {
  const analysis = {
    ...analysisFor("job-test"),
    architecture: null,
    dimensions: { hidden: 768, heads: 12, layers: 12, vocabulary: null },
    input_names: ["input_ids"],
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "overview");
  await expect(page.locator(".model-metrics")).toBeVisible();
  await expect(
    page
      .locator(".model-metrics")
      .getByTestId("model-metric-vocabulary")
      .locator("dd"),
  ).toHaveText("Unknown");
});

test("completed report separates revision, scope, checks, and warnings", async ({
  page,
}) => {
  const base = analysisFor("job-test");
  const analysis = {
    ...base,
    status: "completed" as const,
    warnings: [],
    engine_version: "1.2.3",
    limitations: ["Forward execution was not requested."],
    partial_reasons: [],
    quality: {
      semantic_coverage: 1,
      module_semantic_coverage: 1,
      checkpoint_parameter_coverage: 1,
      tensor_shape_coverage: 1,
      unknown_nodes: 0,
      unmapped_checkpoint_tensors: 0,
      forward_executed: false,
      runtime_trace_available: false,
      mode: "static",
    },
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "report");
  await expect(page.getByTestId("report-checks")).toBeVisible();
  await expect(
    page.locator(".workspace-header").getByLabel("Analysis status: completed"),
  ).toBeVisible();
  await page.locator(".report-identity-details summary").click();
  await expect(page.getByTestId("model-revision")).toHaveText(base.revision!);

  const checks = page.getByTestId("report-checks");
  await expect(
    checks.getByRole("heading", { name: "Coverage and checks" }),
  ).toBeVisible();
  await expect(checks.locator("dd").filter({ hasText: "100.0%" })).toHaveCount(
    4,
  );
  await expect(checks).toContainText("Unknown nodes");
  await expect(checks).toContainText("Forward executed");
  await expect(checks).toContainText("No");

  const scope = page.getByTestId("analysis-scope");
  await expect(
    scope.getByRole("heading", { name: "Analysis scope" }),
  ).toBeVisible();
  const identity = page.locator(".report-identity-details");
  await identity.locator("summary").click();
  await expect(identity).toContainText("1.2.3");
  await expect(scope).toContainText("Forward execution was not requested.");
  await expect(page.locator(".report-warning-section")).toContainText(
    "No incomplete-check details returned.",
  );
  await expect(
    page.getByTestId("report-checks").locator("details"),
  ).toHaveCount(0);
});

test("partial report shows specific missing checks and unknown quality", async ({
  page,
}) => {
  const analysis = {
    ...analysisFor("job-test"),
    status: "partial" as const,
    partial_reasons: ["Tensor shape comparison could not be completed."],
    limitations: ["Custom module details are unavailable."],
    quality: {
      semantic_coverage: null,
      module_semantic_coverage: null,
      checkpoint_parameter_coverage: null,
      tensor_shape_coverage: null,
      unknown_nodes: 0,
      unmapped_checkpoint_tensors: null,
      forward_executed: false,
      runtime_trace_available: null,
      mode: "static",
    },
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "report");
  await expect(page.getByTestId("report-checks")).toBeVisible();
  await expect(
    page.locator(".workspace-header").getByLabel("Analysis status: partial"),
  ).toBeVisible();
  await page.locator(".report-identity-details summary").click();
  await expect(page.getByTestId("model-revision")).toHaveText(
    analysis.revision!,
  );

  const checks = page.getByTestId("report-checks");
  await expect(
    checks.getByRole("heading", { name: "Coverage and checks" }),
  ).toBeVisible();
  await expect(checks).toContainText("Unknown");
  await expect(page.locator(".report-warning-section")).toContainText(
    "Tensor shape comparison could not be completed.",
  );
  await expect(checks).not.toContainText("100.0%");

  const scope = page.getByTestId("analysis-scope");
  await expect(
    scope.getByRole("heading", { name: "Analysis scope" }),
  ).toBeVisible();
  await expect(scope).toContainText("Custom module details are unavailable.");
  await expect(page.locator(".report-warning-section")).toContainText(
    "Tensor shape comparison could not be completed.",
  );
});

test("actual extraction warnings remain visible separately from scope", async ({
  page,
}) => {
  const analysis = {
    ...analysisFor("job-test"),
    status: "partial" as const,
    warnings: ["Opaque operation: output transformation is unverified."],
    limitations: ["Weights were not downloaded."],
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "report");
  await expect(page.getByTestId("report-checks")).toBeVisible();
  await expect(
    page.locator(".report-section").filter({
      has: page.getByRole("heading", {
        name: "Warnings and incomplete checks",
      }),
    }),
  ).toContainText(analysis.warnings[0]);
  await expect(
    page.locator(".report-section").filter({
      has: page.getByRole("heading", {
        name: "Warnings and incomplete checks",
      }),
    }),
  ).not.toContainText(analysis.limitations[0]);
});

test("artifact report explains index declarations and native blockers without graph claims", async ({
  page,
}) => {
  const analysis = {
    ...analysisFor("job-test"),
    status: "inspected" as const,
    stage: "inspected",
    architecture: null,
    root_graph_id: null,
    parameter_count: null,
    warnings: [],
    inspection: {
      kind: "artifact_inspection" as const,
      reason_code: "metadata_budget_exceeded",
      message: "Checkpoint metadata exceeds the configured inspection budget.",
      file_count: 83,
      tensor_count: 0,
      checkpoint_tensor_element_count: null,
      checkpoint_available: true,
      architecture_blockers: [
        "No native Transformers config for model_type=deepseek_v41",
      ],
      declared_checkpoint_index: {
        url:
          "https://huggingface.co/example/model/resolve/" +
          "a".repeat(40) +
          "/model.safetensors.index.json",
        canonical_sha256: "b".repeat(64),
        declared_tensor_names: 96085,
        declared_shards: 48,
        tensor_record_limit: 20000,
      },
    },
  };
  await mockService(page, { job: analysis });
  await openAnalysis(page, analysis.id, "report");
  const report = page.locator(".report-page");
  await expect(report).toContainText("96,085");
  await expect(report).toContainText(
    "No native Transformers config for model_type=deepseek_v41",
  );
  await expect(report).toContainText("Shard headers were not verified");
  await expect(report).not.toContainText(
    "Reconstructed directly from verified Safetensors headers",
  );
  await expect(report).not.toContainText(
    "No incomplete-check details returned.",
  );
});
