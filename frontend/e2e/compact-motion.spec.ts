import { expect, test } from "@playwright/test";
import { block, mockService } from "./service";
import { graphNode, openAnalysis } from "./helpers";

test("entry exposes the form and recent analysis together at normal zoom", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await mockService(page);
  await page.goto("/app");
  await expect(
    page.getByRole("heading", { name: "Recent analyses" }),
  ).toBeVisible();
  const geometry = await page.evaluate(() => {
    const input = document
      .querySelector(".entry-input-row")!
      .getBoundingClientRect();
    const row = document
      .querySelector(".entry-table tbody tr")!
      .getBoundingClientRect();
    return {
      inputBottom: input.bottom,
      historyBottom: row.bottom,
      zoom: getComputedStyle(document.documentElement).zoom,
      overflow: document.documentElement.scrollWidth > innerWidth,
    };
  });
  expect(geometry.inputBottom).toBeLessThan(400);
  expect(geometry.historyBottom).toBeLessThan(760);
  expect(geometry.zoom).toBe("1");
  expect(geometry.overflow).toBe(false);
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

test("mode indicator follows keyboard selection and honors reduced motion", async ({
  page,
}) => {
  await mockService(page);
  await page.goto("/app");
  const modes = page.getByRole("group", { name: "Experience mode" });
  const indicator = modes.locator(".mode-indicator");
  for (const name of ["Learn", "Research"]) {
    const choice = modes.getByRole("button", { name, exact: true });
    await choice.focus();
    await choice.press("Space");
    await expect(choice).toHaveAttribute("aria-pressed", "true");
    await expect
      .poll(async () => {
        const [pill, button] = await Promise.all([
          indicator.boundingBox(),
          choice.boundingBox(),
        ]);
        return Math.abs(
          pill!.x + pill!.width / 2 - button!.x - button!.width / 2,
        );
      })
      .toBeLessThan(3);
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(indicator).toHaveCSS("transition-duration", "0s");
  await modes.getByRole("button", { name: "Learn", exact: true }).click();
  await expect(
    modes.getByRole("button", { name: "Learn", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
});
