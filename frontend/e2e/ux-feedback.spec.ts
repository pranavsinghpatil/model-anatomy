import { expect, test, type Page } from "@playwright/test";
import { analysisFor, mockService } from "./service";
import { openAnalysis } from "./helpers";
import type { AnalysisDetail } from "../src/api/contract";

test.setTimeout(60000);

async function openModelCopy(page: Page) {
  const copy = page.getByRole("button", { name: "Copy model ID", exact: true });
  await expect(
    page.getByRole("button", { name: "Export inspection", exact: true }),
  ).toBeVisible();
  if (!(await copy.isVisible()))
    await page.locator(".model-facts > summary").click();
  await expect(copy).toBeVisible();
  return copy;
}

test("copy confirms only a resolved clipboard write and recovers from denial", async ({
  page,
}) => {
  await mockService(page);
  await page.addInitScript(() => {
    let calls = 0;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async () => {
          if (++calls === 1) throw new Error("Clipboard denied");
        },
      },
    });
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openAnalysis(page);
  await (await openModelCopy(page)).click();
  await expect(
    page.getByText("Copy failed. Select and copy the text.", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Copied", { exact: true })).toHaveCount(0);
  await page
    .getByRole("button", {
      name: "Copy failed. Select and copy the text.",
      exact: true,
    })
    .click();
  await expect(page.getByText("Copied", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("a pending clipboard operation never reports success early", async ({
  page,
}) => {
  await mockService(page);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: () =>
          new Promise<void>((resolve) => {
            (window as unknown as { finishCopy: () => void }).finishCopy =
              resolve;
          }),
      },
    });
  });
  await openAnalysis(page);
  await (await openModelCopy(page)).click();
  await expect(
    page.getByRole("button", { name: "Copying...", exact: true }),
  ).toBeDisabled();
  await expect(page.getByText("Copied", { exact: true })).toHaveCount(0);
  await page.evaluate(() =>
    (window as unknown as { finishCopy: () => void }).finishCopy(),
  );
  await expect(page.getByText("Copied", { exact: true })).toBeVisible();
});

test("export confirms a real download request and replaces duplicate notices", async ({
  page,
}) => {
  await mockService(page);
  await openAnalysis(page, "job-test", "report");
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await page
      .getByRole("button", { name: "Export inspection", exact: true })
      .click();
    const download = page.waitForEvent("download");
    await page
      .getByRole("menuitem", { name: "Report JSON", exact: true })
      .click();
    expect((await download).suggestedFilename()).toMatch(/\.json$/);
    await expect(page.getByTestId("feedback-toast")).toHaveCount(1);
    await expect(page.getByTestId("feedback-toast")).toContainText(
      "Download requested: Report JSON",
    );
  }
  const toast = page.getByTestId("feedback-toast");
  const bounds = await toast.boundingBox();
  const header = await page.locator(".workspace-viewbar").boundingBox();
  expect(bounds).not.toBeNull();
  expect(header).not.toBeNull();
  if (!bounds || !header)
    throw new Error("Notification geometry is unavailable");
  expect(bounds.y).toBeGreaterThanOrEqual(header.y + header.height);
  expect(bounds.x).toBeGreaterThanOrEqual(0);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(
    page.viewportSize()!.width,
  );
  await toast.focus();
  await page.waitForTimeout(6500);
  await expect(toast).toBeVisible();
  await toast.getByRole("button", { name: "Dismiss notification" }).click();
  await expect(page.getByTestId("feedback-toast")).toHaveCount(0);
});

test("export failure stays actionable and retry produces the download", async ({
  page,
}) => {
  await mockService(page);
  let failing = true;
  await page.route("**/api/v1/analyses/job-test/exports/json", (route) =>
    route.fulfill({
      status: failing ? 503 : 200,
      contentType: "application/json",
      body: JSON.stringify(
        failing
          ? {
              error: {
                code: "service_unavailable",
                message: "Export service is temporarily unavailable.",
                request_id: "export-test",
              },
            }
          : { model_id: "org/model" },
      ),
    }),
  );
  await openAnalysis(page);
  await page
    .getByRole("button", { name: "Export inspection", exact: true })
    .click();
  await page
    .getByRole("menuitem", { name: "Report JSON", exact: true })
    .click();
  const alert = page.getByRole("alert");
  await expect(alert).toContainText(
    "Export service is temporarily unavailable.",
  );
  await expect(page.getByTestId("feedback-toast")).toHaveCount(0);
  await alert.getByText("Technical details", { exact: true }).click();
  await expect(alert).toContainText("export-test");
  failing = false;
  const download = page.waitForEvent("download");
  await alert.getByRole("button", { name: /retry/i }).click();
  await download;
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByTestId("feedback-toast")).toContainText(
    "Download requested: Report JSON",
  );
});

test("failed cancellation preserves the running inspection and can be retried", async ({
  page,
}) => {
  let job: AnalysisDetail = {
    ...analysisFor(),
    status: "running",
    stage: "building_structure",
  };
  await mockService(page, { job });
  let cancellations = 0;
  await page.route("**/api/v1/analyses/job-test", async (route) => {
    if (route.request().method() === "DELETE") {
      cancellations += 1;
      if (cancellations === 1)
        return route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({
            error: {
              code: "service_unavailable",
              message: "Cancellation could not be confirmed.",
            },
          }),
        });
      job = { ...job, status: "cancelled", stage: "cancelled" };
      return route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(job),
      });
    }
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(job),
    });
  });
  await openAnalysis(page);
  await page
    .getByRole("button", { name: "Cancel inspection", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "Cancellation could not be confirmed.",
  );
  await expect(page.getByRole("progressbar")).toBeVisible();
  await page
    .getByRole("button", { name: "Cancel inspection", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Inspection cancelled", exact: true }),
  ).toBeVisible();
  expect(cancellations).toBe(2);
});

test("historical completions are silent and background transitions are bounded and deduplicated", async ({
  page,
}) => {
  await mockService(page);
  let rows = [
    analysisFor("history"),
    ...["one", "two", "three"].map((id) => ({
      ...analysisFor(id),
      status: "running" as const,
      stage: "building_structure",
    })),
  ];
  let reads = 0;
  await page.route("**/api/v1/analyses", async (route) => {
    reads += 1;
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ analyses: rows }),
    });
  });
  await page.goto("/");
  await expect.poll(() => reads).toBeGreaterThan(0);
  await expect(page.getByTestId("feedback-toast")).toHaveCount(0);
  rows = rows.map((row) =>
    row.id === "three"
      ? row
      : { ...row, status: "completed", stage: "completed" },
  );
  await expect(page.getByTestId("feedback-toast")).toHaveCount(2, {
    timeout: 10000,
  });
  const afterTransition = reads;
  await expect
    .poll(() => reads, { timeout: 6000 })
    .toBeGreaterThan(afterTransition);
  await expect(page.getByTestId("feedback-toast")).toHaveCount(2);
  for (const toast of await page.getByTestId("feedback-toast").all()) {
    await expect(
      toast.getByRole("link", { name: "View inspection" }),
    ).toBeVisible();
    await expect(toast).not.toContainText("history");
  }
});

test("completion already visible in the active workspace has no duplicate toast", async ({
  page,
}) => {
  await mockService(page, {
    job: {
      ...analysisFor(),
      status: "queued",
      stage: "queued",
      root_graph_id: null,
    },
    pollComplete: true,
  });
  await page.goto("/app/analyses/job-test?view=architecture&awaiting=1");
  await expect(page.getByTestId("graph-canvas")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.getByTestId("feedback-toast")).toHaveCount(0);
});

test("invalid references stay associated with the field and clear when edited", async ({
  page,
}) => {
  const requests = await mockService(page, {
    postStatus: 422,
    postError: "invalid_model",
  });
  await page.goto("/app");
  const field = page.getByLabel("Hugging Face model", { exact: true });
  await field.fill("invalid reference");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  await expect(field).toHaveAttribute("aria-invalid", "true");
  await expect(field).toHaveValue("invalid reference");
  await expect(field).toHaveAttribute("aria-describedby", "model-input-error");
  await expect(page.getByRole("alert")).toContainText("Hugging Face");
  await field.fill("org/corrected");
  await expect(field).toHaveAttribute("aria-invalid", "false");
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect(requests.filter((item) => item.method === "POST")).toHaveLength(1);
});

test("an uncertain submission points to inspections without retrying the mutation", async ({
  page,
}) => {
  await mockService(page, { noRecent: true });
  let submissions = 0;
  await page.route("**/api/v1/analyses", async (route) => {
    if (route.request().method() === "POST") {
      submissions += 1;
      return route.abort("failed");
    }
    return route.fallback();
  });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("confirm");
  await expect(
    alert.getByRole("link", { name: "Your inspections", exact: true }),
  ).toBeVisible();
  await expect(alert.getByRole("button", { name: /retry/i })).toHaveCount(0);
  await page.waitForTimeout(1200);
  expect(submissions).toBe(1);
  await expect(
    page.getByLabel("Hugging Face model", { exact: true }),
  ).toHaveValue("org/model");
});

test("expired inspections retain their original checkpoint reference for a new inspection", async ({
  page,
}) => {
  const reference =
    "https://huggingface.co/org/model/blob/0123456789abcdef0123456789abcdef01234567/fp16.safetensors";
  const job = { ...analysisFor(), model_input: reference };
  await mockService(page, { getStatus: 410, recent: [job] });
  await openAnalysis(page);
  await expect(page.getByRole("alert")).toContainText("Inspection expired");
  await page
    .getByRole("link", { name: "Start a new inspection", exact: true })
    .click();
  await expect(
    page.getByLabel("Hugging Face model", { exact: true }),
  ).toHaveValue(reference);
});
