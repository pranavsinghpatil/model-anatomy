import { expect, test } from "@playwright/test";
import { mockService } from "./service";

test("catalog page renders model cards, filters, search, and routes to inspection", async ({
  page,
}) => {
  await mockService(page);

  // 1. Visit catalog page
  await page.goto("/catalog");

  // Verify page title and header
  await expect(
    page.getByRole("heading", { name: "Verified Model Catalog" }),
  ).toBeVisible();

  // Verify navbar highlights Catalog on desktop viewports
  const isMobile = (page.viewportSize()?.width ?? 1000) <= 820;
  if (!isMobile) {
    const catalogNav = page.getByRole("link", { name: "Catalog", exact: true });
    await expect(catalogNav).toBeVisible();
    await expect(catalogNav).toHaveAttribute("aria-current", "page");
  }

  // Verify cards grid renders all initial 8 models
  const cards = page.locator(".catalog-detail-card");
  await expect(cards).toHaveCount(8);

  // 2. Test category filtering: Sliding Window
  const slidingWindowTab = page.getByRole("tab", { name: /Sliding Window/ });
  await slidingWindowTab.click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "Mistral · 7B" })).toBeVisible();

  // Test category filtering: Decoder-Only
  const decoderTab = page.getByRole("tab", { name: /Decoder-Only/ });
  await decoderTab.click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(5);

  // Reset to All Models
  const allTab = page.getByRole("tab", { name: /All Models/ });
  await allTab.click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(8);

  // 3. Test Search filtering
  const searchInput = page.getByPlaceholder(
    "Search by model name, family, or Hugging Face ID...",
  );
  await searchInput.fill("bert");
  await expect(page.locator(".catalog-detail-card")).toHaveCount(2);
  await expect(page.getByRole("heading", { name: "BERT · Base" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "DistilBERT" })).toBeVisible();

  // Clear search via clear button
  await page.getByLabel("Clear search query").click();
  await expect(page.locator(".catalog-detail-card")).toHaveCount(8);

  // 4. Test inspecting a model from catalog
  const firstCardInspect = page
    .locator(".catalog-detail-card")
    .first()
    .getByRole("button", { name: /Inspect.*architecture/i });
  await firstCardInspect.click();

  // Should navigate to /app with model parameter in query and pre-filled in input
  await expect(page).toHaveURL(/\/app\?model=HuggingFaceTB%2FSmolLM2-135M/);
  await expect(
    page.getByLabel("Hugging Face model", { exact: true }),
  ).toHaveValue("HuggingFaceTB/SmolLM2-135M");

  // 5. Test landing page catalog banner link
  await page.goto("/");
  const browseCatalogBtn = page.getByRole("link", {
    name: "Browse Full Catalog",
  });
  await expect(browseCatalogBtn).toBeVisible();
  await browseCatalogBtn.click();
  await expect(page).toHaveURL(/\/catalog$/);

  // 6. Test rich 4-column footer
  const footer = page.locator(".entry-footer");
  await expect(footer).toBeVisible();
  await expect(footer.getByText("Inspection Views")).toBeVisible();
  await expect(footer.getByRole("heading", { name: "Model Registry" })).toBeVisible();
  await expect(footer.getByText("Provenance")).toBeVisible();
  await expect(footer.getByRole("link", { name: "Back to top of page" })).toBeVisible();
});
