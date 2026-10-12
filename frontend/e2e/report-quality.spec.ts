import { expect, test } from "@playwright/test";
import { analysisFor, mockService } from "./service";

test("selected report status stays consistent while recent jobs are still stale", async ({
  page,
  isMobile,
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
  await page.goto(`/?analysis=${job.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  if (isMobile)
    await page
      .getByRole("button", { name: "Open recent analyses" })
      .first()
      .click();
  const recent = page.locator(".recent-link.active");
  await expect(
    recent.getByRole("status", { name: "Analysis status: completed" }),
  ).toBeVisible();
  await expect(
    recent.getByRole("status", { name: "Analysis status: running" }),
  ).toHaveCount(0);
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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();

  const metrics = page.locator(".model-metrics");
  await expect(metrics.getByTestId("model-metric-vocabulary")).toContainText(
    "50,257",
  );
  await expect(
    metrics.getByTestId("model-metric-heads").locator("strong"),
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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();

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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(
    page
      .locator(".model-metrics")
      .getByTestId("model-metric-vocabulary")
      .locator("strong"),
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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(
    page.locator(".model-title-row").getByLabel("Analysis status: completed"),
  ).toBeVisible();
  await expect(page.getByTestId("model-revision")).toHaveText(base.revision!);

  const checks = page.getByTestId("report-checks");
  await checks.locator("summary").click();
  await expect(checks.locator("dd").filter({ hasText: "100.0%" })).toHaveCount(
    4,
  );
  await expect(checks).toContainText("Unknown nodes");
  await expect(checks).toContainText("Forward executed");
  await expect(checks).toContainText("No");

  const scope = page.getByTestId("analysis-scope");
  await scope.locator("summary").click();
  await expect(scope).toContainText("1.2.3");
  await expect(scope).toContainText("Forward execution was not requested.");
  await expect(page.locator(".partial-notice")).toHaveCount(0);
  await expect(page.locator(".warnings")).toHaveCount(0);
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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(page.locator(".partial-notice")).toContainText(
    "Partial analysis",
  );
  await expect(
    page.locator(".model-title-row").getByLabel("Analysis status: partial"),
  ).toBeVisible();
  await expect(page.getByTestId("model-revision")).toHaveText(
    analysis.revision!,
  );

  const checks = page.getByTestId("report-checks");
  await checks.locator("summary").click();
  await expect(checks).toContainText("Unknown");
  await expect(checks).toContainText(
    "Tensor shape comparison could not be completed.",
  );
  await expect(checks).not.toContainText("100.0%");

  const scope = page.getByTestId("analysis-scope");
  await scope.locator("summary").click();
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
  await page.goto(`/?analysis=${analysis.id}`);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await page.locator(".analysis-notes .warnings summary").click();
  await expect(page.locator(".analysis-notes .warnings")).toContainText(
    analysis.warnings[0],
  );
  await expect(page.locator(".analysis-notes .warnings")).not.toContainText(
    analysis.limitations[0],
  );
});
