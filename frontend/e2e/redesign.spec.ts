import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { graphNode, hasNoHorizontalOverflow, openAnalysis } from "./helpers";
import { analysisFor, embedding, mockService } from "./service";

const widths = [320, 390, 768, 1024, 1440, 1920];

async function setTheme(
  page: import("@playwright/test").Page,
  theme: "Light" | "Dark",
) {
  await page.getByRole("button", { name: "Theme" }).click();
  await page.getByRole("menuitemradio", { name: theme, exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-theme",
    theme.toLowerCase(),
  );
}

test("theme preference persists across workspace contexts", async ({
  page,
}) => {
  await mockService(page);
  await page.goto("/app");
  await setTheme(page, "Dark");

  for (const view of [
    "overview",
    "architecture",
    "artifacts",
    "report",
  ] as const) {
    await openAnalysis(page, "job-test", view);
    const title = view[0].toUpperCase() + view.slice(1);
    await expect(
      page.getByRole("link", { name: title, exact: true }),
    ).toHaveAttribute("aria-current", "page");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  }

  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await setTheme(page, "Light");
  await page.goto("/app");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

test("responsive workspace stays in bounds and accessible across themes", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await mockService(page);
  await page.setViewportSize({ width: widths[0], height: 900 });
  await openAnalysis(page, "job-test", "architecture");
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  for (const [widthIndex, width] of widths.entries()) {
    await page.setViewportSize({ width, height: 900 });
    if (width < 1280) {
      const components = page.getByRole("button", {
        name: "Open components",
        exact: true,
      });
      await expect(components).toBeVisible();
      await components.click();
      await expect(
        page.getByRole("dialog").getByRole("heading", { name: "Components" }),
      ).toBeVisible();
      await page.keyboard.press("Escape");
    } else {
      const openComponents = page.getByRole("button", {
        name: "Open components",
        exact: true,
      });
      if (await openComponents.isVisible()) await openComponents.click();
      await expect(
        page.getByRole("complementary", { name: "Component navigator" }),
      ).toBeVisible();
    }
    for (const [themeIndex, theme] of (["Light", "Dark"] as const).entries()) {
      await setTheme(page, theme);
      await expect(
        page.getByRole("link", { name: "Overview", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Architecture", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Artifacts", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Report", exact: true }),
      ).toBeVisible();
      await expect
        .poll(() => page.evaluate(hasNoHorizontalOverflow))
        .toBe(true);
      if (themeIndex === widthIndex % 2) {
        const results = await new AxeBuilder({ page }).analyze();
        expect(results.violations, `${width}px ${theme}`).toEqual([]);
      }
    }
  }
});

test("opening the workspace does not auto-select a recent analysis", async ({
  page,
}) => {
  const requests = await mockService(page);
  await page.goto("/app");
  await expect(
    page.getByLabel("Hugging Face model", { exact: true }),
  ).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  expect(
    requests.some((request) =>
      /\/analyses\/job-test(?:\?|$)/.test(request.path),
    ),
  ).toBe(false);
});

test("copy failure gives a manual fallback", async ({ page }) => {
  await mockService(page);
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: () => Promise.reject(new Error("Clipboard denied")) },
    });
  });
  await openAnalysis(page, "job-test", "overview");
  await page.getByRole("button", { name: "Copy revision" }).click();
  await expect(
    page.locator(".summary-revision .copy-control [role='status']"),
  ).toContainText("Copy failed. Select and copy the text.");
});

test("copy feedback resets and keyboard navigation remains usable with reduced motion", async ({
  page,
}) => {
  await mockService(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: () => Promise.resolve() },
    });
  });
  await openAnalysis(page, "job-test", "overview");
  await page.getByRole("button", { name: "Copy revision" }).focus();
  await page.getByRole("button", { name: "Copy revision" }).press("Enter");
  await expect(
    page.locator(".summary-revision .copy-control [role='status']"),
  ).toHaveText("Copied");
  await expect(
    page.getByRole("button", { name: "Copy revision" }),
  ).toBeEnabled();
  const architecture = page
    .getByRole("navigation", { name: "Workspace views" })
    .getByRole("link", { name: "Architecture", exact: true });
  await architecture.focus();
  await architecture.press("Enter");
  await expect(page).toHaveURL(/view=architecture/);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
});

test("failed report export leaves an actionable error", async ({ page }) => {
  await mockService(page);
  await page.route("**/api/v1/analyses/job-test/exports/json", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        error: {
          code: "unavailable",
          message: "Export service unavailable",
          request_id: "export-failure-test",
        },
      }),
    }),
  );
  await openAnalysis(page, "job-test", "report");
  await page.getByRole("button", { name: "Export inspection" }).click();
  await page.getByRole("menuitem", { name: "Report JSON" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Export service unavailable",
  );
});

test("view changes and keyboard panel resizing preserve inspection context", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await page
    .getByRole("button", { name: "Open components", exact: true })
    .click();
  await graphNode(page, embedding.label);
  const inspector = page.getByTestId("inspector");
  await inspector.getByRole("tab", { name: /^Tensors/ }).click();
  await page
    .getByRole("button", { name: "Readable view", exact: true })
    .click();
  const viewport = page.locator(".react-flow__viewport");
  // Search focuses at up to 1.2x. Readable view settles at 1x in this desktop
  // canvas; capture that result before asserting workspace navigation preserves it.
  await expect(viewport).toHaveAttribute("style", /scale\(1\)/);
  const beforeNavigation = await viewport.getAttribute("style");
  await expect(
    inspector.getByRole("tab", { name: /^Tensors/ }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(
    page.locator(`[data-component-id="${embedding.id}"]`),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(viewport).toHaveAttribute("style", beforeNavigation!);
  const panel = page.getByRole("complementary", {
    name: "Inspector",
    exact: true,
  });
  const beforeWidth = (await panel.boundingBox())!.width;
  const separator = page.getByRole("separator", { name: "Resize inspector" });
  await separator.focus();
  await separator.press("ArrowLeft");
  await expect
    .poll(async () => (await panel.boundingBox())!.width)
    .not.toBe(beforeWidth);
  await expect(
    inspector.getByRole("tab", { name: /^Tensors/ }),
  ).toHaveAttribute("aria-selected", "true");
  const afterResize = await viewport.getAttribute("style");
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await page.getByRole("link", { name: "Architecture", exact: true }).click();
  await expect(viewport).toHaveAttribute("style", afterResize!);
  await expect(
    inspector.getByRole("tab", { name: /^Tensors/ }),
  ).toHaveAttribute("aria-selected", "true");
});

test("late submission response does not redirect a visitor who continued an existing analysis", async ({
  page,
}) => {
  await mockService(page);
  let release!: () => void;
  let postStarted!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started = new Promise<void>((resolve) => {
    postStarted = resolve;
  });
  await page.route("**/api/v1/analyses", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    postStarted();
    await gate;
    await route.fulfill({ json: analysisFor("late-created"), status: 202 });
  });
  await page.goto("/app");
  await page
    .getByLabel("Hugging Face model", { exact: true })
    .fill("org/model");
  await page
    .getByRole("button", { name: "Inspect model", exact: true })
    .click();
  const response = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await started;
  await page.getByRole("link", { name: /^Open inspection of / }).click();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test/);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();
  release();
  await response;
  await expect(
    page.getByRole("link", { name: "Architecture", exact: true }),
  ).toBeVisible();
  await expect(page).toHaveURL(/\/app\/analyses\/job-test/);
});

test("entry pages and long report identifiers fit every target width in both themes", async ({
  page,
}) => {
  test.setTimeout(120_000);
  const identifier = `acme/${"technical-model-identifier-".repeat(12)}revision`;
  await mockService(page, {
    job: {
      ...analysisFor(),
      model_id: identifier,
      architecture: "LongArchitectureClass".repeat(10),
      warnings: [identifier],
    },
  });
  for (const width of widths) {
    await page.setViewportSize({ width, height: 900 });
    for (const theme of ["Light", "Dark"] as const) {
      await page.goto("/");
      await setTheme(page, theme);
      for (const path of [
        "/",
        "/app",
        "/app/analyses/job-test?view=overview",
        "/app/analyses/job-test?view=report",
      ]) {
        if (new URL(page.url()).pathname + new URL(page.url()).search !== path)
          await page.goto(path);
        await expect(page.getByRole("main")).toBeVisible();
        await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
        await expect
          .poll(
            () => page.evaluate(hasNoHorizontalOverflow),
            `${width}px ${theme} ${path}`,
          )
          .toBe(true);
      }
    }
  }
});
