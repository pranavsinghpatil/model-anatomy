import { expect, test } from "@playwright/test";
import { block, mockService } from "./service";
import { graphNode, openAnalysis, UI_SCALE } from "./helpers";

test("entry keeps the form visible and recent analysis reachable at 90 percent zoom", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await mockService(page);
  await page.goto("/app");
  await expect(
    page.getByRole("heading", { name: "Your inspections" }),
  ).toBeVisible();
  const geometry = await page.evaluate(() => {
    const input = document
      .querySelector(".entry-input-row")!
      .getBoundingClientRect();
    return {
      inputBottom: input.bottom,
      zoom: getComputedStyle(document.documentElement).zoom,
      overflow: document.documentElement.scrollWidth > innerWidth,
    };
  });
  expect(geometry.inputBottom).toBeLessThan(480);
  expect(geometry.zoom).toBe(String(UI_SCALE));
  expect(geometry.overflow).toBe(false);
  const recent = page.locator(".entry-table tbody tr").first();
  await expect(recent).toBeVisible();
  await recent.scrollIntoViewIfNeeded();
  const bounds = await recent.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds!.y).toBeGreaterThanOrEqual(0);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(800);
});

test("docked inspection keeps graph actions clear of search at laptop width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1158, height: 828 });
  await mockService(page);
  await openAnalysis(page);
  await graphNode(page, block.label);
  await expect(page.getByTestId("inspector")).toBeVisible();
  const geometry = await page.evaluate(() => {
    const bounds = (selector: string) => {
      const box = document.querySelector(selector)!.getBoundingClientRect();
      return {
        left: box.left,
        right: box.right,
        top: box.top,
        bottom: box.bottom,
      };
    };
    return {
      actions: bounds(".graph-panel-actions"),
      search: bounds(".graph-search"),
      header: bounds(".graph-header"),
      canvas: bounds(".graph-canvas > .react-flow"),
    };
  });
  expect(geometry.actions.right).toBeLessThanOrEqual(geometry.search.left);
  expect(geometry.search.right).toBeLessThanOrEqual(geometry.header.right);
  expect(geometry.header.bottom).toBeLessThanOrEqual(geometry.canvas.top + 1);
});

test("workspace scale is 90 percent and graph controls honor reduced motion", async ({
  page,
}) => {
  await mockService(page);
  await page.setViewportSize({ width: 1440, height: 960 });
  await openAnalysis(page);
  await expect(page.locator("html")).toHaveCSS("zoom", String(UI_SCALE));
  const workspaceScale = await page
    .locator(".workspace-shell")
    .evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      return bounds.height / (element as HTMLElement).offsetHeight;
    });
  expect(workspaceScale).toBeCloseTo(UI_SCALE, 2);
  await expect(
    page.getByRole("button", { name: "Zoom in", exact: true }),
  ).toBeVisible();
  const themeButton = page.getByRole("button", { name: "Theme", exact: true });
  await themeButton.focus();
  await themeButton.press("Enter");
  const darkTheme = page.getByRole("menuitemradio", {
    name: "Dark",
    exact: true,
  });
  await expect(darkTheme).toBeVisible();
  await darkTheme.focus();
  await darkTheme.press("Enter");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.emulateMedia({ reducedMotion: "reduce" });
  const transitionSeconds = await page
    .locator(".react-flow__viewport")
    .evaluate((element) =>
      getComputedStyle(element)
        .transitionDuration.split(",")
        .map((duration) => {
          const value = Number.parseFloat(duration);
          return duration.trim().endsWith("ms") ? value / 1000 : value;
        }),
    );
  expect(transitionSeconds.every((duration) => duration <= 0.00001)).toBe(true);
  const before = await page
    .locator(".react-flow__viewport")
    .getAttribute("style");
  await page.getByRole("button", { name: "Zoom in", exact: true }).focus();
  await page
    .getByRole("button", { name: "Zoom in", exact: true })
    .press("Enter");
  await expect(page.locator(".react-flow__viewport")).not.toHaveAttribute(
    "style",
    before!,
  );
});
