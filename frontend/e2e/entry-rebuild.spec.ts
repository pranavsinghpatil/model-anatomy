import { expect, test } from "@playwright/test";
import { analysisFor, mockService } from "./service";

test("root and app routes open the same immediately usable entry", async ({
  page,
}) => {
  await mockService(page);

  for (const path of ["/", "/app"]) {
    await page.goto(path);
    await expect(
      page.getByRole("heading", {
        name: "Open a model. Understand its structure.",
      }),
    ).toBeVisible();
    await expect(
      page.getByLabel("Hugging Face model", { exact: true }),
    ).toBeVisible();
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("org/model");
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeEnabled();
    await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  }
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
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(
    requests.some(
      (request) =>
        request.path === "/api/v1/analyses" && request.method === "POST",
    ),
  ).toBe(false);
});

test("a new Learn inspection opens Architecture with its reading guide", async ({
  page,
}) => {
  await mockService(page, { job: analysisFor() });
  await page.goto("/app");

  await page
    .getByRole("group", { name: "Experience mode" })
    .getByRole("button", { name: "Learn", exact: true })
    .click();
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
  await expect(page.locator("details.learn-guide")).toBeVisible();
});
