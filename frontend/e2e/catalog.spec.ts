import { expect, test } from "@playwright/test";
import { analysisFor, mockService } from "./service";

const records = [
  {
    model_id: "HuggingFaceTB/SmolLM2-135M",
    category: "text-generation",
    revision: "abc123",
    reference: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/abc123",
    stored_report_id: "stored-smol",
    status: "completed",
    architecture: "LlamaForCausalLM",
    declared_task: "text-generation",
    started_at: "2026-10-19T09:59:00Z",
    finished_at: "2026-10-19T10:00:00Z",
    engine_version: "engine-1",
    warning_count: 0,
    graph_count: 1,
    tensor_count: 182,
    parameter_counts: { unique: 134515008 },
    dimensions: { hidden: 576, heads: [9, 3], layers: 30, vocabulary: null },
    artifact: "model.safetensors",
    explicit_artifact: false,
    source: "inspection" as const,
    category_source: "model configuration",
    declared_task_source: "model configuration",
  },
  {
    model_id: "org/custom-model",
    category: "unclassified",
    revision: null,
    reference: null,
    stored_report_id: null,
    status: "artifact_error",
    architecture: null,
    declared_task: null,
    started_at: null,
    finished_at: null,
    engine_version: null,
    warning_count: null,
    graph_count: null,
    tensor_count: null,
    parameter_counts: { unique: null },
    dimensions: { hidden: null, heads: null, layers: null, vocabulary: null },
    artifact: "weights.bin",
    explicit_artifact: true,
    source: "benchmark" as const,
    benchmark_observations: [
      {
        status: "unsupported",
        engine_version: "engine-old",
        finished_at: "2026-10-18T10:00:00Z",
      },
    ],
  },
];

function response(results: any[] = records) {
  const status_counts = Object.fromEntries(
    [...new Set(results.map((item) => item.status))].map((status) => [
      status,
      results.filter((item) => item.status === status).length,
    ]),
  );
  return {
    results,
    total: results.length,
    status_counts,
    generated_at: "2026-10-21T10:00:00Z",
    benchmark_publication_id: "a".repeat(64),
  };
}

async function catalogApi(
  page: import("@playwright/test").Page,
  options: { empty?: boolean; failOnce?: boolean } = {},
) {
  await mockService(page);
  let failures = options.failOnce ? 2 : 0;
  const requests: string[] = [];
  await page.route("**/api/v1/catalog/discovery", async (route) => {
    requests.push(
      `${route.request().method()} ${new URL(route.request().url()).pathname}`,
    );
    if (failures) {
      failures--;
      return route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({
          error: { code: "unavailable", message: "Try again" },
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(response(options.empty ? [] : undefined)),
    });
  });
  await page.route("**/api/v1/catalog/stored-smol/inspect", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(analysisFor("saved-analysis")),
    }),
  );
  return requests;
}

test("one catalog combines inspection and benchmark provenance with honest metrics", async ({
  page,
}) => {
  const requests = await catalogApi(page);
  await page.goto("/catalog?view=benchmark");
  await expect(
    page.getByRole("heading", { name: "Unified model evidence" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Benchmark results" }),
  ).toHaveCount(0);
  await expect(page.locator(".benchmark-card")).toHaveCount(2);
  await expect(page.getByText("134,515,008", { exact: true })).toBeVisible();
  await expect(page.getByRole("article").nth(1)).toContainText("Unknown");
  await expect(page.getByRole("article").nth(1)).toContainText(
    "repository reference only",
  );
  await expect(
    page.getByText("Saved inspection", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Benchmark attempt", { exact: true }),
  ).toBeVisible();

  await page
    .getByRole("searchbox", { name: "Search model catalog" })
    .fill("custom");
  await expect(page.locator(".benchmark-card")).toHaveCount(1);
  await page.locator(".benchmark-details summary").click();
  await expect(page.getByText("Benchmark history")).toBeVisible();
  await expect(
    page.getByText("Reconstruction unavailable · engine engine-old · finished"),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Retry inspection for org/custom-model" })
    .click();
  await expect(page).toHaveURL(
    /\/app\?model=https%3A%2F%2Fhuggingface.co%2Forg%2Fcustom-model$/,
  );
  expect(requests).toContain("GET /api/v1/catalog/discovery");
});

test("saved reports reopen through inspectCatalog and discovery entries refresh into the same view", async ({
  page,
}) => {
  await mockService(page);
  let includeNew = false;
  const discoveryRequests: string[] = [];
  await page.route("**/api/v1/catalog/discovery", async (route) => {
    discoveryRequests.push(route.request().method());
    const rows = includeNew
      ? [
          ...records,
          {
            ...records[0],
            model_id: "org/new-inspection",
            stored_report_id: "stored-new",
          },
        ]
      : records;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(response(rows)),
    });
  });
  await page.route("**/api/v1/catalog/stored-smol/inspect", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(analysisFor("saved-analysis")),
    }),
  );
  await page.goto("/catalog");
  await page
    .getByRole("button", {
      name: "Open saved inspection for HuggingFaceTB/SmolLM2-135M",
    })
    .click();
  await expect(page).toHaveURL(
    /\/app\/analyses\/saved-analysis\/architecture$/,
  );
  await page.goto("/catalog?view=stored");
  await expect(page.locator(".benchmark-card")).toHaveCount(2);
  includeNew = true;
  await page.getByRole("button", { name: "Refresh model catalog" }).click();
  await expect(page.getByText("org/new-inspection")).toBeVisible();
  expect(discoveryRequests.length).toBeGreaterThan(1);
});

test("recorded inspection failure keeps its prior saved report and retry reference", async ({
  page,
}) => {
  await mockService(page);
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        response([
          {
            ...records[1],
            source: "inspection",
            stored_report_id: null,
            benchmark_observations: [],
            category_source: "model configuration",
            previous_observations: [
              {
                status: "inspected",
                engine_version: "engine-previous",
                finished_at: "2026-10-17T09:00:00Z",
                stored_report_id: "previous-custom-report",
              },
            ],
          },
        ]),
      ),
    }),
  );
  await page.route(
    "**/api/v1/catalog/previous-custom-report/inspect",
    (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(analysisFor("previous-analysis")),
      }),
  );
  await page.goto("/catalog");
  await expect(
    page.getByText("Recorded failure · Artifact processing failed"),
  ).toBeVisible();
  await expect(
    page.getByText("Inspection attempt", { exact: true }),
  ).toBeVisible();
  await page.locator(".benchmark-details summary").click();
  await expect(
    page.getByText("Recorded failure; previous saved report available"),
  ).toBeVisible();
  await expect(page.getByText("Category source").locator("..")).toContainText(
    "model configuration",
  );
  await page
    .getByRole("button", { name: "Open previous saved report" })
    .click();
  await expect(page).toHaveURL(
    /\/app\/analyses\/previous-analysis\/architecture$/,
  );
});

test("unverified explicit artifacts stay withheld and retry at the pinned repository", async ({
  page,
}) => {
  await mockService(page);
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        response([
          {
            ...records[1],
            model_id: "org/private-model",
            revision: "abc123",
            reference: "https://huggingface.co/org/private-model/tree/abc123",
            artifact: null,
            artifact_verified: false,
            explicit_artifact: true,
            selection_id: "opaque-selection-42",
          },
        ]),
      ),
    }),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/catalog");
  await expect(page.getByText("Requested artifact withheld")).toBeVisible();
  await expect(page.getByText("weights.bin", { exact: true })).toHaveCount(0);
  await page
    .getByRole("button", {
      name: "Inspect pinned repository for org/private-model",
    })
    .click();
  await expect(page.locator("#model-input")).toHaveValue(
    "https://huggingface.co/org/private-model/tree/abc123",
  );
});

test("status and category filters show global totals and paginate at 24", async ({
  page,
}) => {
  const many = Array.from({ length: 50 }, (_, index) => ({
    ...records[index % records.length],
    model_id: `org/model-${String(index).padStart(3, "0")}`,
    revision: String(index).padStart(40, "0"),
    status: index % 2 ? "partial" : "completed",
    category: index % 2 ? "text-generation" : "image-classification",
    source: "benchmark" as const,
  }));
  await mockService(page);
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ...response(many),
        status_counts: { completed: 25, partial: 25 },
      }),
    }),
  );
  await page.goto("/catalog");
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  await expect(page.getByText("Page 1 of 3")).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  await page
    .getByRole("combobox", { name: "Filter by task category" })
    .selectOption("image-classification");
  await page
    .getByRole("combobox", { name: "Filter by outcome" })
    .selectOption("completed");
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  await expect(
    page.getByText("Showing 1–24 of 25 matching records"),
  ).toBeVisible();
});

test("background refresh error keeps last data and privacy excludes static benchmark fetches", async ({
  page,
}) => {
  await mockService(page);
  let failNext = false;
  let calls = 0;
  const unexpected: string[] = [];
  page.on("request", (request) => {
    if (
      request.url().includes("benchmark-catalog.json") ||
      request.url().includes("/benchmarks/catalog")
    )
      unexpected.push(request.url());
  });
  await page.route("**/api/v1/catalog/discovery", async (route) => {
    calls++;
    await route.fulfill({
      status: failNext ? 503 : 200,
      contentType: "application/json",
      body: failNext
        ? JSON.stringify({
            error: { code: "unavailable", message: "Catalog offline" },
          })
        : JSON.stringify(response()),
    });
  });
  await page.goto("/catalog");
  await expect(page.locator(".benchmark-card")).toHaveCount(2);
  failNext = true;
  await page.getByRole("button", { name: "Refresh model catalog" }).click();
  await expect(page.getByRole("status")).toContainText(
    "Showing the last loaded catalog",
  );
  await expect(page.locator(".benchmark-card")).toHaveCount(2);
  expect(calls).toBeGreaterThan(1);
  expect(unexpected).toEqual([]);
});

test("initial failure offers retry and empty discovery stays honest", async ({
  page,
}) => {
  await mockService(page);
  let allowSuccess = false;
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      status: allowSuccess ? 200 : 503,
      contentType: "application/json",
      body: allowSuccess
        ? JSON.stringify(response())
        : JSON.stringify({
            error: { code: "unavailable", message: "Try again" },
          }),
    }),
  );
  await page.goto("/catalog");
  await expect(
    page.getByRole("heading", { name: "Could not load the model catalog" }),
  ).toBeVisible();
  allowSuccess = true;
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.locator(".benchmark-card")).toHaveCount(2);

  await page.unrouteAll();
  await catalogApi(page, { empty: true });
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "No model evidence is available yet" }),
  ).toBeVisible();
});

test("catalog cards fit a narrow viewport", async ({ page }) => {
  await catalogApi(page);
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/catalog");
  await expect(page.locator(".benchmark-card")).toHaveCount(2);
  expect(
    await page.evaluate(() => {
      const main = document.querySelector("#main")!;
      const disclaimer = document.querySelector(".benchmark-disclaimer")!;
      return {
        documentFits: document.documentElement.scrollWidth <= innerWidth,
        mainFits: main.scrollWidth <= main.clientWidth,
        disclaimerFits: disclaimer.scrollWidth <= disclaimer.clientWidth,
      };
    }),
  ).toEqual({ documentFits: true, mainFits: true, disclaimerFits: true });
});
