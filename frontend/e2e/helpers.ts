import type { Page } from "@playwright/test";
import { expect } from "@playwright/test";

export async function expectGraphFramed(page: Page) {
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
  await page.getByLabel("Find component").fill(label);
  await page
    .locator(".search-results")
    .getByRole("button", { name: label, exact: true })
    .click();
}
export async function showDetails(page: Page, mobile: boolean) {
  if (mobile)
    await page
      .getByRole("navigation", { name: "Workspace views" })
      .getByRole("button", { name: "Details" })
      .click();
}
export async function showQuestions(page: Page, mobile: boolean) {
  if (mobile)
    await page
      .getByRole("navigation", { name: "Workspace views" })
      .getByRole("button", { name: "Questions" })
      .click();
  else await page.getByRole("tab", { name: "Questions", exact: true }).click();
}
