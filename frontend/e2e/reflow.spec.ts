import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mockService, embedding } from "./service";
import { graphNode, openAnalysis } from "./helpers";

test.use({
  viewport: { width: 720, height: 480 },
  deviceScaleFactor: 2,
  reducedMotion: "reduce",
});

test("workspace reflows at 200-percent-equivalent CSS dimensions", async ({
  page,
}) => {
  // A 1440×960 physical viewport at 200% browser zoom has a 720×480
  // CSS viewport. Exercise that layout at 2× raster scale; this checks
  // equivalent reflow, not a browser-chrome zoom preference.
  test.setTimeout(60_000);
  await mockService(page);
  await openAnalysis(page);
  for (const theme of ["Light", "Dark"]) {
    await page.getByRole("button", { name: "Theme", exact: true }).click();
    await page.getByRole("menuitemradio", { name: theme, exact: true }).click();
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    await graphNode(page, embedding.label);
    const inspector = page.getByTestId("inspector");
    await expect(
      inspector.getByRole("heading", { name: embedding.label, exact: true }),
    ).toBeVisible();
    await inspector.getByRole("tab", { name: /^Evidence/ }).click();
    await page
      .getByRole("button", { name: "Back to graph", exact: true })
      .click();
    await page.getByRole("button", { name: "Anatomy", exact: true }).click();
    await expect(page.getByTestId("anatomy-drawing")).toBeVisible();
    await page
      .getByRole("button", { name: "Reset illustration view", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Separate parts", exact: true })
      .click();
    await expect(page.getByTestId("anatomy-drawing")).toHaveAttribute(
      "data-pose",
      "1.000",
    );
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    await page
      .getByRole("button", { name: "Inspect component", exact: true })
      .click();
    await expect(
      inspector.getByRole("tab", { name: /^Evidence/ }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Back to graph", exact: true })
      .click();
    await page.getByRole("button", { name: "Graph", exact: true }).click();
  }
});
