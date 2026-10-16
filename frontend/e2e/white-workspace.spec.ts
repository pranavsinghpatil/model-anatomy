import { expect, test } from "@playwright/test";
import { graphNode, openAnalysis } from "./helpers";
import { embedding, mockService } from "./service";

test("new visitors get white appearance and a canvas without an open navigator", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await mockService(page);
  await openAnalysis(page);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator("html")).toHaveCSS(
    "background-color",
    "rgb(248, 249, 251)",
  );
  await expect(page.locator(".graph-canvas > .react-flow")).toHaveCSS(
    "background-color",
    "rgb(255, 255, 255)",
  );
  await expect(
    page.getByRole("button", { name: "Open components", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("complementary", {
      name: "Component navigator",
      exact: true,
    }),
  ).toHaveCount(0);
  if (page.viewportSize()!.width >= 1024) {
    const drawing = await page
      .locator(".graph-canvas > .react-flow")
      .boundingBox();
    expect(drawing).not.toBeNull();
    expect(drawing!.height).toBeGreaterThan(page.viewportSize()!.height * 0.7);
  }
  await page.getByRole("button", { name: "Theme", exact: true }).click();
  await page
    .getByRole("menuitemradio", { name: "System", exact: true })
    .click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});

test("model facts and type guidance disclose without losing inspection context", async ({
  page,
}) => {
  await mockService(page);
  await openAnalysis(page);
  const facts = page.locator("details.model-facts");
  await facts.locator("summary").click();
  await expect(
    page.getByRole("region", { name: "Model context" }),
  ).toBeVisible();
  await expect(facts.getByText("Parameters", { exact: true })).toBeVisible();
  await facts.locator("summary").press("Escape");
  await expect(facts).not.toHaveAttribute("open");
  await expect(facts.locator("summary")).toBeFocused();

  await graphNode(page, embedding.label);
  const guide = page.getByTestId("inspector").locator(".inspector-definition");
  await expect(guide).not.toHaveAttribute("open");
  const mode = page.getByRole("group", { name: "Experience mode" });
  await mode.getByRole("button", { name: "Learn", exact: true }).click();
  await expect(guide).toHaveAttribute("open");
  await expect(
    page
      .getByTestId("inspector")
      .getByRole("heading", { name: embedding.label, exact: true }),
  ).toBeVisible();
  await mode.getByRole("button", { name: "Research", exact: true }).click();
  await expect(guide).not.toHaveAttribute("open");
});
