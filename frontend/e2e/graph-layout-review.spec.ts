import { expect, test } from "@playwright/test";
import { block, embedding, mockService } from "./service";
import {
  graphNode,
  MIN_READABLE_NODE_FONT_SIZE,
  openAnalysis,
} from "./helpers";

test("short initial workspaces open on a readable representative component", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 720, height: 480 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  const repeatedNode = page.locator(`.model-node[data-node-id="${block.id}"]`);
  await expect(repeatedNode).toBeVisible();
  await expect
    .poll(() =>
      repeatedNode.evaluate((node) => {
        const title = node.querySelector<HTMLElement>(".node-title")!;
        const rect = node.getBoundingClientRect();
        const flow = node.closest(".react-flow")!.getBoundingClientRect();
        return {
          titleSize:
            Number.parseFloat(getComputedStyle(title).fontSize) *
            (rect.width / (node as HTMLElement).offsetWidth),
          visible:
            rect.left >= flow.left &&
            rect.right <= flow.right &&
            rect.top >= flow.top &&
            rect.bottom <= flow.bottom,
        };
      }),
    )
    .toMatchObject({ visible: true });
  await expect
    .poll(() =>
      repeatedNode.evaluate((node) => {
        const title = node.querySelector<HTMLElement>(".node-title")!;
        return (
          Number.parseFloat(getComputedStyle(title).fontSize) *
          (node.getBoundingClientRect().width /
            (node as HTMLElement).offsetWidth)
        );
      }),
    )
    .toBeGreaterThanOrEqual(MIN_READABLE_NODE_FONT_SIZE);
  await page.screenshot({
    path: testInfo.outputPath("short-window-readable-graph.png"),
    animations: "disabled",
  });
});

test("technical graph keeps a clear drawing area and controls across responsive widths", async ({
  page,
}, testInfo) => {
  test.setTimeout(60_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
    await expect(page.getByTestId("graph-canvas")).toBeVisible();
    await expect(
      page.getByRole("navigation", { name: "Graph path" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Fit graph", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Readable view", exact: true }),
    ).toBeVisible();

    const geometry = await page
      .getByTestId("graph-canvas")
      .evaluate((canvas) => {
        const box = (selector: string) => {
          const element = canvas.querySelector<HTMLElement>(selector);
          if (!element) return null;
          const rect = element.getBoundingClientRect();
          return { top: rect.top, bottom: rect.bottom, height: rect.height };
        };
        const bounds = canvas.getBoundingClientRect();
        return {
          height: bounds.height,
          header: box(".graph-header"),
          viewport: box(".react-flow"),
          dock: box(".graph-dock"),
          duplicateToolbar: Boolean(canvas.querySelector(".canvas-toolbar")),
        };
      });
    expect(geometry.header).not.toBeNull();
    expect(geometry.viewport).not.toBeNull();
    expect(geometry.dock).not.toBeNull();
    expect(geometry.duplicateToolbar).toBe(false);
    expect(geometry.viewport!.top).toBeGreaterThanOrEqual(
      geometry.header!.bottom - 1,
    );
    expect(geometry.viewport!.bottom).toBeLessThanOrEqual(
      geometry.dock!.top + 1,
    );
    expect(geometry.viewport!.height / geometry.height).toBeGreaterThanOrEqual(
      width >= 1024 ? 0.7 : 0.62,
    );
    await page.screenshot({
      path: testInfo.outputPath(`graph-layout-${width}.png`),
      animations: "disabled",
    });

    await page
      .getByRole("textbox", { name: "Find component" })
      .fill("embedding");
    await expect(page.locator(".search-results")).toBeVisible();
    const searchBounds = await page.locator(".search-results").boundingBox();
    expect(searchBounds).not.toBeNull();
    expect(searchBounds!.x + searchBounds!.width).toBeLessThanOrEqual(
      width + 1,
    );
    await page.getByRole("textbox", { name: "Find component" }).fill("");

    await page.getByText("Options", { exact: true }).click();
    const orientationToggle = page.getByRole("button", {
      name: /^Switch to (horizontal|vertical) layout$/,
    });
    const initialOrientationAction =
      await orientationToggle.getAttribute("aria-label");
    await orientationToggle.click();
    const nextOrientationAction = initialOrientationAction?.includes(
      "horizontal",
    )
      ? "vertical"
      : "horizontal";
    await expect(
      page.getByRole("button", {
        name: `Switch to ${nextOrientationAction} layout`,
      }),
    ).toBeVisible();
    await page
      .getByRole("button", {
        name: `Switch to ${nextOrientationAction} layout`,
      })
      .click();
    const densityToggle = page.getByRole("button", {
      name: /^Show (detailed|compact) node cards$/,
    });
    const initialDensityAction = await densityToggle.getAttribute("aria-label");
    await densityToggle.click();
    const nextDensityAction = initialDensityAction?.includes("detailed")
      ? "compact"
      : "detailed";
    await expect(
      page.getByRole("button", {
        name: `Show ${nextDensityAction} node cards`,
      }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: `Show ${nextDensityAction} node cards` })
      .click();
    await page.getByText("Options", { exact: true }).click();

    await page
      .getByRole("button", { name: "Readable view", exact: true })
      .click();
    await graphNode(page, embedding.label);
    await expect(page.getByTestId("inspector")).toBeVisible();
    await expect(
      page
        .getByTestId("inspector")
        .getByRole("heading", { name: embedding.label, exact: true }),
    ).toBeVisible();
    if (width < 1024)
      await page.getByRole("button", { name: "Back to graph" }).click();
    await page.getByRole("button", { name: "Graph", exact: true }).click();
    await page.getByRole("button", { name: "Anatomy", exact: true }).click();
    await expect(page.getByTestId("anatomy-drawing")).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath(`anatomy-layout-${width}.png`),
      animations: "disabled",
    });
    await page.getByRole("button", { name: "Graph", exact: true }).click();

    if (width >= 1024) {
      // Scene switching restores the focused viewport. Request the overview
      // before opening a different component farther down the graph.
      await page
        .getByRole("button", { name: "Fit graph", exact: true })
        .click();
      await page
        .getByRole("button", { name: `Open ${block.label}`, exact: true })
        .click();
      await expect(
        page
          .getByRole("navigation", { name: "Graph path" })
          .getByRole("button", {
            name: block.label,
            exact: true,
          }),
      ).toBeVisible();
      await page
        .getByRole("navigation", { name: "Graph path" })
        .getByRole("button")
        .first()
        .click();
      await expect(page.getByTestId("graph-canvas")).toBeVisible();
    }
  }
});
