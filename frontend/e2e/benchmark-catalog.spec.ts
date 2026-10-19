import { expect, test } from "@playwright/test";

function report(count = 50) {
  return {
    schema_version: 1,
    generated_at: "2026-10-20T12:00:00Z",
    engine_version: "engine-test",
    manifest_sha256: "a".repeat(64),
    results: Array.from({ length: count }, (_, index) => {
      const status = [
        "completed",
        "partial",
        "inspected",
        "artifact_error",
        "http_access",
      ][index % 5];
      const revision = index.toString(16).padStart(40, "0");
      const isGraphResult = status === "completed" || status === "partial";
      const warnings =
        status === "partial" ? ["Some layers were not traced."] : [];
      return {
        model_id: `org/model-${String(index).padStart(3, "0")}`,
        category: index % 2 ? "text-generation" : "image-classification",
        revision,
        status,
        architecture: isGraphResult
          ? index % 2
            ? "LlamaForCausalLM"
            : "ViTForImageClassification"
          : null,
        reference: `https://huggingface.co/org/model-${String(index).padStart(3, "0")}/tree/${revision}`,
        declared_task: index % 2 ? "text-generation" : "image-classification",
        started_at: "2026-10-20T11:00:00Z",
        finished_at: "2026-10-20T11:00:02Z",
        engine_version: "engine-test",
        warning_count: warnings.length,
        warnings,
        reason_codes: status === "partial" ? ["graph_partial"] : [],
        graph_count: isGraphResult ? 1 : null,
        tensor_count: status === "inspected" ? 12 + index : null,
        parameter_counts: { unique: null },
        coverage:
          status === "completed"
            ? { module_tree: 1, graph: 1 }
            : status === "partial"
              ? { module_tree: 0.9, graph: 0.8 }
              : undefined,
        error:
          status === "artifact_error"
            ? { message: "No supported checkpoint artifact was found." }
            : status === "http_access"
              ? { message: "The pinned repository could not be accessed." }
              : undefined,
      };
    }),
  };
}

async function serveReport(
  page: import("@playwright/test").Page,
  options: { status?: number; body?: unknown } = {},
) {
  let requests = 0;
  await page.route("**/benchmark-catalog.json", async (route) => {
    requests++;
    await route.fulfill({
      status: options.status ?? 200,
      contentType: "application/json",
      body: JSON.stringify(options.body ?? report()),
    });
  });
  return () => requests;
}

test("benchmark search, filters, counts, and 24-row pagination use measured rows", async ({
  page,
}) => {
  await serveReport(page);
  await page.goto("/catalog");
  await expect(
    page.getByRole("heading", { name: "Measured compatibility attempts" }),
  ).toBeVisible();
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  await expect(
    page.getByText("Showing 1–24 of 50 matching results"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  await expect(page.getByText("Page 2 of 3")).toBeVisible();

  await page
    .getByRole("searchbox", { name: "Search benchmark results" })
    .fill("model-001");
  await expect(page.locator(".benchmark-card")).toHaveCount(1);
  await expect(
    page.getByText("Showing 1–1 of 1 matching results"),
  ).toBeVisible();
  await page.locator(".benchmark-details summary").click();
  await expect(page.getByText("Some layers were not traced.")).toBeVisible();

  await page
    .getByRole("searchbox", { name: "Search benchmark results" })
    .fill("");
  await page
    .getByRole("combobox", { name: "Filter by task category" })
    .selectOption("image-classification");
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  await expect(
    page.getByText("Showing 1–24 of 25 matching results"),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: "Filter by outcome" })
    .selectOption("partial");
  await expect(page.locator(".benchmark-card")).toHaveCount(5);
  await expect(
    page.getByText("Partial graph", { exact: true }).first(),
  ).toBeVisible();
});

test("pinned action opens the normal visitor-owned inspection flow with its reference", async ({
  page,
}) => {
  await serveReport(page, { body: report(1) });
  await page.goto("/catalog");
  await page.getByRole("button", { name: "Inspect pinned model" }).click();
  await expect(page).toHaveURL(/\/app\?model=/);
  await expect(page.locator("#model-input")).toHaveValue(
    `https://huggingface.co/org/model-000/tree/${"0".repeat(40)}`,
  );
});

test("report loading failure offers retry and does not show sample results", async ({
  page,
}) => {
  let allowSuccess = false;
  let requests = 0;
  await page.route("**/benchmark-catalog.json", async (route) => {
    requests++;
    await route.fulfill({
      status: allowSuccess ? 200 : 503,
      contentType: "application/json",
      body: allowSuccess ? JSON.stringify(report(1)) : "unavailable",
    });
  });
  await page.goto("/catalog");
  await expect(
    page.getByRole("heading", { name: "Could not load benchmark results" }),
  ).toBeVisible();
  await expect(page.locator(".benchmark-card")).toHaveCount(0);
  allowSuccess = true;
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.locator(".benchmark-card")).toHaveCount(1);
  expect(requests).toBeGreaterThan(1);
});

test("loading, malformed reports, and empty measured reports stay explicit", async ({
  page,
}) => {
  let release: (() => void) | undefined;
  await page.route("**/benchmark-catalog.json", async (route) => {
    await new Promise<void>((resolve) => {
      release = resolve;
    });
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(report(0)),
    });
  });
  await page.goto("/catalog");
  await expect(page.getByRole("status")).toContainText(
    "Loading measured benchmark results",
  );
  release?.();
  await expect(
    page.getByText("No measured results match these filters"),
  ).toBeVisible();
  await page.unrouteAll();
  await serveReport(page, { body: { schema_version: 9, results: [] } });
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Could not load benchmark results" }),
  ).toBeVisible();
  await expect(page.locator(".benchmark-card")).toHaveCount(0);
});

test("artifact and access failures keep unavailable graph fields unknown", async ({
  page,
}) => {
  await serveReport(page, { body: report(5) });
  await page.goto("/catalog");
  await page
    .getByRole("combobox", { name: "Filter by outcome" })
    .selectOption("artifact_error");
  await expect(
    page.getByRole("heading", { name: "org/model-003" }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("article")
      .getByText("Artifact processing failed", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".benchmark-card .benchmark-facts")).toContainText(
    "Unknown",
  );
  await expect(
    page.locator(".benchmark-card").getByText("Complete static report"),
  ).toHaveCount(0);
  await page
    .getByRole("combobox", { name: "Filter by outcome" })
    .selectOption("http_access");
  await expect(
    page.getByRole("heading", { name: "org/model-004" }),
  ).toBeVisible();
  await expect(
    page.getByRole("article").getByText("HF request failed", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("The pinned repository could not be accessed."),
  ).toHaveCount(1);
});

test("benchmark cards fit a narrow viewport without horizontal page overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await serveReport(page, { body: report(5) });
  await page.goto("/catalog");
  await expect(page.locator(".benchmark-card")).toHaveCount(5);
  const widths = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth,
    viewport: document.documentElement.clientWidth,
  }));
  expect(widths.document).toBeLessThanOrEqual(widths.viewport);
});
