import { expect, test } from "@playwright/test";
import { analysisFor, mockService } from "./service";

test("root and app routes open the same immediately usable entry", async ({
  page,
}, testInfo) => {
  await mockService(page);

  for (const path of ["/", "/app"]) {
    await page.goto(path);
    await expect(
      page.getByRole("heading", {
        name: "See how a model is built.",
      }),
    ).toBeVisible();
    const intro = page.locator(".entry-heading");
    const formStage = page.locator(".entry-submit");
    const introBox = await intro.boundingBox();
    const formBox = await formStage.boundingBox();
    expect(introBox).not.toBeNull();
    expect(formBox).not.toBeNull();
    expect(introBox!.y + introBox!.height).toBeLessThanOrEqual(formBox!.y);
    expect(formBox!.y).toBeLessThan(page.viewportSize()!.height);
    expect(
      await page
        .locator(".entry-main")
        .evaluate((element) => element.clientWidth),
    ).toBeLessThanOrEqual(800);
    await expect(
      page.getByLabel("Hugging Face model", { exact: true }),
    ).toBeVisible();
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeEnabled();
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeInViewport();
    await expect(
      page.getByRole("link", { name: "Your analyses" }),
    ).toBeVisible();
    await expect(page.locator(".entry-header .entry-brand-name")).toHaveCSS(
      "white-space",
      "nowrap",
    );
    await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
    await expect(
      page.getByRole("heading", { name: /real inspections/ }),
    ).toBeVisible();
    await expect(
      page.getByText("SafeTensors Verified", { exact: true }),
    ).toHaveCount(0);
    await expect(page.getByText("134,515,008", { exact: true })).toHaveCount(0);
  }
  await page.screenshot({
    path: testInfo.outputPath("entry-fixture-preview.png"),
    animations: "disabled",
  });
});

test("model examples fill the input without submitting", async ({ page }) => {
  const requests = await mockService(page);
  await page.goto("/");

  await page
    .getByRole("button", {
      name: "Fill model field with HuggingFaceTB/SmolLM2-135M",
    })
    .click();

  await expect(
    page.getByLabel("Hugging Face model", { exact: true }),
  ).toHaveValue("HuggingFaceTB/SmolLM2-135M");
  await expect(
    page.getByRole("button", {
      name: "Fill model field with HuggingFaceTB/SmolLM2-135M",
    }),
  ).toHaveAttribute("aria-pressed", "true");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/another-model");
  await expect(
    page.getByRole("button", {
      name: "Fill model field with HuggingFaceTB/SmolLM2-135M",
    }),
  ).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(
    requests.some(
      (request) =>
        request.path === "/api/v1/analyses" && request.method === "POST",
    ),
  ).toBe(false);
});

test("a new inspection opens Architecture in the unified workspace", async ({
  page,
}) => {
  await mockService(page, { job: analysisFor() });
  await page.goto("/app");

  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();

  await expect(page).toHaveURL(/\/app\/analyses\/job-test\?view=architecture$/);
  await expect(
    page.getByRole("link", { name: "Architecture", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Export analysis" }),
  ).toBeVisible();
});

test("progress hands off a partial result to the graph and keeps findings secondary", async ({
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
    pollResult: "partial",
  });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  await expect(
    page.getByRole("progressbar", { name: "Inspection in progress" }),
  ).toBeVisible();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test\?view=architecture$/);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "View findings", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "View findings", exact: true }).click();
  await expect(page).toHaveURL(/view=report$/);
  await expect(page.getByTestId("report-checks")).toBeVisible();
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
});

test("progress hands off an inspected result to artifacts", async ({
  page,
}) => {
  await mockService(page, {
    job: {
      ...analysisFor(),
      status: "queued",
      stage: "queued",
      root_graph_id: null,
      architecture: null,
      parameter_count: null,
    },
    pollComplete: true,
    pollResult: "inspected",
    artifactPages: [
      {
        kind: "artifact_inspection",
        reason_code: "architecture_not_supported",
        message:
          "Artifact metadata is available; no architecture graph was produced.",
        revision: "0123456789abcdef0123456789abcdef01234567",
        files: ["config.json"],
        file_count: 1,
        tensors: [],
        tensor_count: 0,
        checkpoint_tensor_element_count: null,
        checkpoint_available: false,
        source_config: null,
        evidence: [],
        offset: 0,
        limit: 50,
      },
    ],
  });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  await expect(
    page.getByRole("progressbar", { name: "Inspection in progress" }),
  ).toBeVisible();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test\?view=artifacts$/);
  await expect(page.getByTestId("artifact-inspection")).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "No architecture graph", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("architecture_not_supported")).toBeVisible();
  await page.getByRole("link", { name: "Explore artifacts" }).click();
  await expect(page.getByTestId("artifact-inspection")).toBeVisible();
});
