import { expect, test } from "@playwright/test";
import { openAnalysis } from "./helpers";
import { analysisFor, mockService } from "./service";

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
  await expect(page.getByTestId("model-revision")).toHaveText(base.revision!);

  const checks = page.getByTestId("report-checks");
  await expect(
    checks.getByRole("heading", { name: "Report checks" }),
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
  await expect(
    page.locator(".report-section").filter({
      has: page.getByRole("heading", { name: "Artifact identity" }),
    }),
  ).toContainText("1.2.3");
  await expect(scope).toContainText("Forward execution was not requested.");
  await expect(page.getByTestId("report-checks")).toContainText(
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
  await expect(page.getByTestId("model-revision")).toHaveText(
    analysis.revision!,
  );

  const checks = page.getByTestId("report-checks");
  await expect(
    checks.getByRole("heading", { name: "Report checks" }),
  ).toBeVisible();
  await expect(checks).toContainText("Unknown");
  await expect(checks).toContainText(
    "Tensor shape comparison could not be completed.",
  );
  await expect(checks).not.toContainText("100.0%");

  const scope = page.getByTestId("analysis-scope");
  await expect(
    scope.getByRole("heading", { name: "Analysis scope" }),
  ).toBeVisible();
  await expect(scope).toContainText("Custom module details are unavailable.");
  await expect(checks).toContainText(
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
    page
      .locator(".report-section")
      .filter({ has: page.getByRole("heading", { name: "Warnings" }) }),
  ).toContainText(analysis.warnings[0]);
  await expect(
    page
      .locator(".report-section")
      .filter({ has: page.getByRole("heading", { name: "Warnings" }) }),
  ).not.toContainText(analysis.limitations[0]);
});
