import { expect, test } from "@playwright/test";
import type { AnalysisDetail } from "../src/api/contract";
import { expectGraphFramed } from "./helpers";

test("live removed examples endpoint and IDs are unavailable", async ({
  page,
}, testInfo) => {
  test.skip(
    !process.env.E2E_LIVE_API,
    "Set E2E_LIVE_API=1 when the service runs on port 8001.",
  );
  const appRequests: string[] = [];
  page.on("request", (request) => appRequests.push(request.url()));
  expect((await page.request.get("/api/v1/examples")).status()).toBe(404);

  await page.goto("/");
  await expect(page.getByText("No analysis selected").first()).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await page.screenshot({
    path: testInfo.outputPath("live-empty-workspace.png"),
    animations: "disabled",
  });

  await page.goto("/?analysis=example-gpt2");
  await expect(page.getByRole("alert")).toContainText("Analysis not found");
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(
    appRequests.some((url) => new URL(url).pathname === "/api/v1/examples"),
  ).toBe(false);
  expect(
    appRequests.some((url) => new URL(url).pathname.includes("/graphs/")),
  ).toBe(false);
});

test("live model submission reaches a stored architecture", async ({
  page,
  isMobile,
}, testInfo) => {
  test.setTimeout(180_000);
  test.skip(
    !process.env.E2E_LIVE_API,
    "Set E2E_LIVE_API=1 with the HTTP service and worker running.",
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("No analysis selected").first()).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("HuggingFaceTB/SmolLM2-360M");
  await expect(
    page.getByRole("button", { name: "Analyze Model", exact: true }),
  ).toBeEnabled({ timeout: 60_000 });
  const createdResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await page
    .getByRole("button", { name: "Analyze Model", exact: true })
    .click();
  const response = await createdResponse;
  expect(response.status()).toBe(202);
  const created = (await response.json()) as AnalysisDetail;
  await expect(page).toHaveURL(
    new RegExp(`analysis=${encodeURIComponent(created.id)}`),
  );
  await expect(page.getByTestId("graph-canvas")).toBeVisible({
    timeout: 130_000,
  });
  await expectGraphFramed(page);
  const result = (await (
    await page.request.get(`/api/v1/analyses/${created.id}`)
  ).json()) as AnalysisDetail;
  expect(result.status).toBe("completed");
  expect(result.warnings).toEqual([]);
  expect(result.input_names).toEqual(["input_ids"]);
  expect(result.model_id).toBe("HuggingFaceTB/SmolLM2-360M");
  expect(result.parameter_count).toBe(361_821_120);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    result.model_id,
  );
  await page.screenshot({
    path: testInfo.outputPath("live-created-analysis.png"),
    animations: "disabled",
  });
  if (isMobile)
    await page
      .getByRole("button", { name: "Open recent analyses" })
      .first()
      .click();
  await expect(page.locator(".recent-link.active")).toContainText(
    result.model_id,
  );
  await page.goto("/");
  await expect(page).toHaveURL(
    new RegExp(`analysis=${encodeURIComponent(created.id)}`),
  );
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expectGraphFramed(page);
  await page.screenshot({
    path: testInfo.outputPath("live-restored-analysis.png"),
    animations: "disabled",
  });
  expect(errors).toEqual([]);
});
