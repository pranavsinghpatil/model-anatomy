import type { Page } from "@playwright/test";
import { expect } from "@playwright/test";

export async function expectReadableGraph(page: Page) {
  await expect
    .poll(() =>
      page
        .locator(".model-node")
        .first()
        .evaluate((node) => {
          const title = node.querySelector(".node-title")!;
          const scale =
            node.getBoundingClientRect().width /
            (node as HTMLElement).offsetWidth;
          return Number.parseFloat(getComputedStyle(title).fontSize) * scale;
        }),
    )
    .toBeGreaterThanOrEqual(11.8);
  await expect
    .poll(() =>
      page.getByTestId("graph-canvas").evaluate((canvas) => {
        const bounds = canvas.getBoundingClientRect();
        return Array.from(canvas.querySelectorAll(".model-node")).some(
          (node) => {
            const rect = node.getBoundingClientRect();
            return (
              rect.left >= bounds.left &&
              rect.right <= bounds.right &&
              rect.top >= bounds.top + 50 &&
              rect.bottom <= bounds.bottom
            );
          },
        );
      }),
    )
    .toBe(true);
}

export async function expectGraphFramed(page: Page) {
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await expect
    .poll(async () =>
      page.getByTestId("graph-canvas").evaluate((canvas) => {
        const bounds = canvas.getBoundingClientRect();
        const nodes = Array.from(canvas.querySelectorAll(".react-flow__node"));
        if (!nodes.length) return ["No rendered nodes"];
        return nodes
          .filter((node) => {
            const rect = node.getBoundingClientRect();
            return (
              rect.width < 1 ||
              rect.height < 1 ||
              rect.left < bounds.left - 1 ||
              rect.right > bounds.right + 1 ||
              rect.top < bounds.top - 1 ||
              rect.bottom > bounds.bottom + 1
            );
          })
          .map((node) => node.getAttribute("data-id"));
      }),
    )
    .toEqual([]);
}

export async function graphNode(page: Page, label: string) {
  const back = page.getByRole("button", { name: "Back to graph", exact: true });
  if (await back.isVisible()) await back.click();
  await page.getByRole("textbox", { name: /find.*component/i }).fill(label);
  await page
    .locator(".search-results")
    .getByRole("button", { name: label, exact: true })
    .click();
}

export async function openNestedGraph(page: Page, label: string) {
  const componentTab = page.getByRole("tab", { name: /^Component/ });
  if (await componentTab.isVisible()) await componentTab.click();
  const inspectorAction = page.getByRole("button", {
    name: "Open nested graph",
    exact: true,
  });
  if (await inspectorAction.isVisible()) await inspectorAction.click();
  else
    await page
      .getByRole("button", { name: `Open ${label}`, exact: true })
      .click();
}

export async function openAnalysis(
  page: Page,
  analysisId = "job-test",
  view: "overview" | "architecture" | "artifacts" | "report" = "architecture",
) {
  await page.goto(`/app/analyses/${analysisId}?view=${view}`);
}

export async function showDetails(page: Page, _mobile?: boolean) {
  await expect(page.getByTestId("inspector")).toBeVisible();
}

export async function showQuestions(page: Page, _mobile?: boolean) {
  const tab = page.getByRole("tab", { name: "Questions", exact: true });
  if (await tab.isVisible()) {
    await tab.click();
    return;
  }
  await page
    .getByRole("button", { name: "Model questions", exact: true })
    .click();
}
