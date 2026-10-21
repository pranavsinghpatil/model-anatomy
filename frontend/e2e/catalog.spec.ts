import { expect, test } from "@playwright/test";
import { analysisFor, mockService } from "./service";

const items = [
  {
    id: "registry-smol",
    model_id: "HuggingFaceTB/SmolLM2-135M",
    revision: "abc123",
    architecture: "LlamaForCausalLM",
    status: "completed" as const,
    parameter_count: 134515008,
    dimensions: { hidden: 576, heads: [9, 3], layers: 30, vocabulary: null },
    engine_version: "engine-1",
    schema_version: "1",
    artifact: "model.safetensors",
    explicit_artifact: false,
    inspected_at: "2026-10-19T10:00:00Z",
  },
  {
    id: "registry-unknown",
    model_id: "org/custom-model",
    revision: "def456",
    architecture: null,
    status: "inspected" as const,
    parameter_count: null,
    dimensions: { hidden: null, heads: null, layers: null, vocabulary: null },
    engine_version: "engine-1",
    schema_version: "1",
    artifact: "weights.bin",
    explicit_artifact: true,
    inspected_at: "2026-10-18T10:00:00Z",
  },
];

async function catalogApi(
  page: import("@playwright/test").Page,
  options: { empty?: boolean; failOnce?: boolean; paginated?: boolean } = {},
) {
  await mockService(page);
  let failures = options.failOnce ? 50 : 0;
  const requests: string[] = [];
  await page.route("**/api/v1/catalog**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    requests.push(`${request.method()} ${url.pathname}${url.search}`);
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
    if (url.pathname.endsWith("/inspect"))
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(analysisFor("saved-analysis")),
      });
    const q = url.searchParams.get("q")?.toLowerCase() ?? "";
    const filtered = (options.empty ? [] : items).filter((item) =>
      item.model_id.toLowerCase().includes(q),
    );
    const offset = Number(url.searchParams.get("offset") ?? 0);
    const limit = Number(url.searchParams.get("limit") ?? 24);
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: filtered.slice(offset, offset + limit),
        total: options.paginated && !q ? 30 : filtered.length,
        offset,
        limit,
      }),
    });
  });
  return requests;
}

test("catalog renders saved API facts and opens the selected registry record", async ({
  page,
}) => {
  const requests = await catalogApi(page);
  await page.goto("/catalog");
  await expect(page).toHaveURL(/\/catalog$/);
  await expect(
    page.getByRole("button", { name: "Stored inspections", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("heading", { name: "Model Catalog" }),
  ).toBeVisible({ timeout: 25_000 });
  await expect(page.locator(".catalog-detail-card")).toHaveCount(2);
  await expect(page.getByText("134,515,008 parameters")).toBeVisible();
  await expect(
    page.locator(".catalog-detail-card").nth(0).locator(".spec-metric").nth(3),
  ).toContainText("Unknown");
  await expect(page.getByText("org/custom-model")).toBeVisible();

  await page
    .getByRole("searchbox", { name: "Search stored model catalog" })
    .fill("custom");
  await expect(page.locator(".catalog-detail-card")).toHaveCount(1);
  await expect(
    page.getByText("Unknown", { exact: true }).first(),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Open stored inspection for org/custom-model",
    })
    .click();
  await expect(page).toHaveURL(
    /\/app\/analyses\/saved-analysis\/architecture$/,
  );
  expect(requests).toContain("POST /api/v1/catalog/registry-unknown/inspect");
});

test("catalog refresh reveals a new inspection and keeps the selected view in the URL", async ({
  page,
}) => {
  await mockService(page);
  let includeNew = false;
  const catalogRequests: string[] = [];
  await page.route("**/api/v1/catalog**", async (route) => {
    const url = new URL(route.request().url());
    catalogRequests.push(
      `${route.request().method()} ${url.pathname}${url.search}`,
    );
    const models = includeNew
      ? [
          ...items,
          {
            ...items[0],
            id: "registry-new",
            model_id: "org/new-inspection",
            inspected_at: "2026-10-21T10:00:00Z",
          },
        ]
      : items;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: models.slice(
          Number(url.searchParams.get("offset") ?? 0),
          Number(url.searchParams.get("offset") ?? 0) + 24,
        ),
        total: 30,
        offset: Number(url.searchParams.get("offset") ?? 0),
        limit: Number(url.searchParams.get("limit") ?? 24),
      }),
    });
  });
  await page.goto("/catalog");
  await expect(page.locator(".catalog-detail-card")).toHaveCount(2);
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(0);
  includeNew = true;
  const pageZeroRequestsBeforeRefresh = catalogRequests.filter((request) =>
    request.includes("offset=0"),
  ).length;
  await page
    .getByRole("button", { name: "Refresh stored inspections" })
    .click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(3);
  await expect(page.getByText("org/new-inspection")).toBeVisible();
  expect(
    catalogRequests.filter((request) => request.includes("offset=0")).length,
  ).toBeGreaterThan(pageZeroRequestsBeforeRefresh);

  await page.getByRole("button", { name: "Benchmark results" }).click();
  await expect(page).toHaveURL(/\/catalog\?view=benchmark$/);
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Benchmark results", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.goBack();
  await expect(page).toHaveURL(/\/catalog$/);
  await expect(page.locator(".catalog-detail-card")).toHaveCount(3);
});

test("catalog supports server pagination and shows honest empty state", async ({
  page,
}) => {
  const requests = await catalogApi(page, { paginated: true });
  await page.goto("/catalog");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page).toHaveURL(/\/catalog$/);
  expect(requests.some((request) => request.includes("offset=24"))).toBe(true);
  await page
    .getByRole("searchbox", { name: "Search stored model catalog" })
    .fill("absent");
  await expect(
    page.getByRole("heading", { name: /No stored models match/ }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Inspect a model" }),
  ).toBeVisible();
});

test("saved model names and checkpoint paths fit a 320px viewport", async ({
  page,
}) => {
  await mockService(page);
  const item = {
    ...items[0],
    model_id:
      "organization/a-long-model-name-with-an-unbroken-architecture-identifier",
    revision: "a".repeat(40),
    artifact: "component-directory/model-00001-of-00099.safetensors",
    explicit_artifact: true,
  };
  await page.route("**/api/v1/catalog?**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ models: [item], total: 1, offset: 0, limit: 24 }),
    }),
  );
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/catalog");
  await expect(page.locator(".card-model-title")).toHaveText(item.model_id);
  await expect(page.locator(".tech-class")).toContainText(item.artifact);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("catalog retries API errors and explains a truly empty database", async ({
  page,
}) => {
  await catalogApi(page, { failOnce: true });
  await page.goto("/catalog");
  await expect(
    page.getByRole("heading", { name: "Could not load the model catalog" }),
  ).toBeVisible();
  await page.unrouteAll();
  await catalogApi(page);
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(2);

  await page.unrouteAll();
  await catalogApi(page, { empty: true });
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "No inspections are stored yet" }),
  ).toBeVisible();
});

test("entry links into the database-backed catalog without model count claims", async ({
  page,
}) => {
  await catalogApi(page);
  await page.goto("/");
  await expect(
    page.getByText("Saved inspections", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("8 Pre-verified", { exact: true })).toHaveCount(
    0,
  );
  await page.getByRole("link", { name: "Browse Full Catalog" }).click();
  await expect(page).toHaveURL(/\/catalog$/);
});
