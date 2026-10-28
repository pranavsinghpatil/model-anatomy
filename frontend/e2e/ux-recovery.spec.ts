import { expect, test } from "@playwright/test";
import { analysisFor, mockService } from "./service";
import { openAnalysis, showQuestions } from "./helpers";
import type { Health } from "../src/api/contract";

test.setTimeout(60000);

const health = (checks: {
  database: boolean;
  storage: boolean;
  worker: boolean;
}): Health => ({
  status: "ok" as const,
  api_version: "v1",
  checks,
  analysis_available: checks.database && checks.storage && checks.worker,
  ai_available: true,
  storage_backend: "local",
  reason: null,
});

test("availability states distinguish healthy runs, stored-only lookup, and unavailable storage", async ({
  page,
}) => {
  await mockService(page, { noRecent: true });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  const status = page.getByRole("status").filter({
    hasText: "New inspections available",
  });
  await expect(status).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled();

  await page.unrouteAll();
  await mockService(page, {
    noRecent: true,
    health: health({ database: true, storage: true, worker: false }),
  });
  await page.reload();
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(
    page.getByText("Saved inspections available; new runs unavailable"),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Find saved inspection", exact: true }),
  ).toBeEnabled();

  for (const checks of [
    { database: false, storage: true, worker: true },
    { database: true, storage: false, worker: true },
  ]) {
    await page.unrouteAll();
    await mockService(page, { health: health(checks), noRecent: true });
    await page.goto("/app");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await expect(
      page.getByText("Inspection storage unavailable", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeDisabled();
  }
});

test("checking availability waits for health before enabling submission", async ({
  page,
}) => {
  await mockService(page, { noRecent: true });
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => (release = resolve));
  let healthSeen!: () => void;
  const observed = new Promise<void>((resolve) => (healthSeen = resolve));
  await page.route("**/health", async (route) => {
    healthSeen();
    await waiting;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        health({ database: true, storage: true, worker: true }),
      ),
    });
  });
  await page.goto("/app");
  await observed;
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await expect(page.getByText("Checking inspection service")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeDisabled();
  release();
  await expect(
    page.getByRole("button", { name: "Inspect model", exact: true }),
  ).toBeEnabled();
});

test("worker offline still opens a matching saved inspection", async ({
  page,
}) => {
  await mockService(page, {
    health: health({ database: true, storage: true, worker: false }),
  });
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        results: [
          {
            model_id: "org/saved-model",
            status: "completed",
            stored_report_id: "stored-1",
            source: "inspection",
            revision: "abc123",
            reference: "https://huggingface.co/org/saved-model/tree/abc123",
            architecture: "GPT2LMHeadModel",
            category: "text-generation",
            declared_task: "text-generation",
            started_at: null,
            finished_at: null,
            engine_version: "test",
            warning_count: 0,
            graph_count: 1,
            tensor_count: 1,
            parameter_counts: { unique: 124439808 },
            dimensions: {
              hidden: 768,
              heads: 12,
              layers: 12,
              vocabulary: 50257,
            },
            artifact: "model.safetensors",
            explicit_artifact: false,
          },
        ],
        total: 1,
        status_counts: { completed: 1 },
        generated_at: "2026-10-26T00:00:00Z",
        benchmark_publication_id: "a".repeat(64),
      }),
    }),
  );
  await page.route("**/api/v1/catalog/stored-1/inspect", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(analysisFor()),
    }),
  );
  await page.goto("/catalog");
  await page
    .getByRole("button", { name: "Open saved inspection for org/saved-model" })
    .click();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test\?view=architecture$/);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
});

for (const [code, copy] of [
  ["queue_full", "Queue is full right now."],
  ["rate_limited", "Daily limit reached."],
] as const) {
  test(`${code} shows the service-specific recovery copy`, async ({ page }) => {
    await mockService(page, { noRecent: true });
    await page.route("**/api/v1/analyses", (route) =>
      route.request().method() === "POST"
        ? route.fulfill({
            status: 429,
            contentType: "application/json",
            body: JSON.stringify({
              error: {
                code,
                message:
                  code === "queue_full"
                    ? "Queue is full right now."
                    : "Daily limit reached.",
                request_id: "limit-test",
              },
            }),
          })
        : route.fallback(),
    );
    await page.goto("/app");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await page
      .getByRole("button", { name: "Inspect model", exact: true })
      .click();
    await expect(page.getByRole("alert")).toContainText(copy);
  });
}

test("a pending submission cannot be submitted twice", async ({ page }) => {
  await mockService(page, { noRecent: true });
  let postCount = 0;
  let release!: () => void;
  const waiting = new Promise<void>((resolve) => (release = resolve));
  let postSeen!: () => void;
  const observed = new Promise<void>((resolve) => (postSeen = resolve));
  await page.route("**/api/v1/analyses", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    postCount++;
    postSeen();
    await waiting;
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify(analysisFor("new-job")),
    });
  });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  const submit = page.getByRole("button", {
    name: "Inspect model",
    exact: true,
  });
  await submit.click();
  await observed;
  await expect(page.getByRole("button", { name: "Starting…" })).toBeDisabled();
  await expect(submit).toHaveCount(0);
  expect(postCount).toBe(1);
  release();
  await expect(page).toHaveURL(/\/app\/analyses\/new-job/);
});

test("empty and unmatched catalog states offer a direct recovery action", async ({
  page,
}) => {
  await mockService(page);
  let rows: unknown[] = [];
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        results: rows,
        total: rows.length,
        status_counts: {},
        generated_at: "2026-10-26T00:00:00Z",
        benchmark_publication_id: "a".repeat(64),
      }),
    }),
  );
  await page.goto("/catalog");
  await expect(
    page.getByRole("heading", { name: "No model evidence is available yet" }),
  ).toBeVisible();
  await expect(
    page.getByRole("searchbox", { name: "Search model catalog" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("combobox", { name: "Filter by outcome" }),
  ).toHaveCount(0);
  await page.getByRole("link", { name: /inspect a model/i }).click();
  await expect(page).toHaveURL(/\/#submit-title$/);
  await expect(
    page.getByRole("heading", { name: "Inspect a model" }),
  ).toBeVisible();

  await page.goto("/catalog");
  rows = [
    {
      model_id: "org/available",
      status: "completed",
      stored_report_id: "saved-2",
      source: "inspection",
      revision: "abc123",
      reference: "https://huggingface.co/org/available/tree/abc123",
      architecture: "GPT2LMHeadModel",
      category: "text-generation",
      declared_task: "text-generation",
      started_at: null,
      finished_at: null,
      engine_version: "test",
      warning_count: 0,
      graph_count: 1,
      tensor_count: 1,
      parameter_counts: { unique: 1 },
      dimensions: {},
      artifact: "model.safetensors",
      explicit_artifact: false,
    },
  ];
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "org/available", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("searchbox", { name: "Search model catalog" })
    .fill("no-match");
  await expect(page.getByText(/No records match these filters/i)).toBeVisible();
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(
    page.getByRole("heading", { name: "org/available", exact: true }),
  ).toBeVisible();
});

test("question failure preserves the draft and failed citations can be retried", async ({
  page,
  isMobile,
}) => {
  await mockService(page, { questionError: true });
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await showQuestions(page, isMobile);
  const field = page.getByLabel("Ask about model evidence");
  await field.fill("What evidence supports this component?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Explanation provider unavailable",
  );
  await expect(field).toHaveValue("What evidence supports this component?");

  await page.unrouteAll();
  await mockService(page);
  let fail = true;
  await page.route("**/api/v1/analyses/job-test/graph-index", (route) =>
    fail
      ? route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({
            error: {
              code: "graph_unavailable",
              message: "Citation index unavailable",
            },
          }),
        })
      : route.fallback(),
  );
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".citation-button")).toBeVisible();
  await page.locator(".citation-button").click();
  await expect(page.getByRole("alert")).toContainText(
    "Citation index unavailable",
  );
  fail = false;
  await page.getByRole("button", { name: /retry/i }).click();
  await expect(
    page.getByTestId("inspector").getByRole("tab", { name: /^Evidence/ }),
  ).toHaveAttribute("aria-selected", "true");
});

test("help and entry navigation move focus to the requested content", async ({
  page,
}) => {
  await mockService(page);
  await page.route("**/api/v1/catalog/discovery", (route) =>
    route.fulfill({
      json: {
        results: [],
        total: 0,
        status_counts: {},
        generated_at: "2026-10-27T00:00:00Z",
        benchmark_publication_id: "a".repeat(64),
      },
    }),
  );
  await page.goto("/catalog");
  await page
    .getByRole("link", { name: "Support boundaries", exact: true })
    .click();
  const help = page.locator(".entry-boundaries");
  await expect(help).toHaveAttribute("open", "");
  await expect(help.locator("summary")).toBeFocused();
  await page
    .getByRole("link", { name: "Your inspections", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your inspections", exact: true }),
  ).toBeFocused();
  await page.goto("/catalog");
  await page
    .getByRole("link", { name: "Inspect a model", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Inspect a model", exact: true }),
  ).toBeFocused();
});
