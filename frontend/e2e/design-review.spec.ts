import { expect, test, type Page } from "@playwright/test";
import type { ArtifactInspectionResponse } from "../src/api/contract";
import { expectGraphFramed, graphNode, openAnalysis } from "./helpers";
import { embedding, mockService } from "./service";

const revision = "0123456789abcdef0123456789abcdef01234567";

function artifactPage(): ArtifactInspectionResponse {
  return {
    kind: "artifact_inspection",
    reason_code: "architecture_not_supported",
    message: "Artifact metadata is available for inspection.",
    revision,
    files: ["config.json", "model.safetensors", "tokenizer.json"],
    file_count: 3,
    tensors: [],
    tensor_count: 0,
    checkpoint_tensor_element_count: null,
    checkpoint_available: true,
    source_config: { model_type: "gpt2", n_embd: 768, n_layer: 12 },
    evidence: [],
    offset: 0,
    limit: 50,
  };
}

async function chooseTheme(page: Page, theme: "Light" | "Dark") {
  const expected = theme.toLowerCase();
  if ((await page.locator("html").getAttribute("data-theme")) !== expected) {
    await page.getByRole("button", { name: "Theme", exact: true }).click();
    await page.getByRole("menuitemradio", { name: theme, exact: true }).click();
  }
  await expect(page.locator("html")).toHaveAttribute("data-theme", expected);
}

async function expectNoHorizontalOverflow(page: Page) {
  const result = await page.evaluate(() => {
    const root = document.documentElement;
    const header = document.querySelector<HTMLElement>(".workspace-header");
    const views = document.querySelector<HTMLElement>(".primary-views");
    const children = [...(header?.children ?? [])].map((element) => {
      const rect = element.getBoundingClientRect();
      return { x: rect.x, right: rect.right, y: rect.y, bottom: rect.bottom };
    });
    const overlaps: number[] = [];
    for (let first = 0; first < children.length; first += 1) {
      for (let second = first + 1; second < children.length; second += 1) {
        const a = children[first];
        const b = children[second];
        const width = Math.min(a.right, b.right) - Math.max(a.x, b.x);
        const height = Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y);
        if (width > 1 && height > 1) overlaps.push(width * height);
      }
    }
    return {
      viewport: root.clientWidth,
      document: root.scrollWidth,
      headerWidth: header?.clientWidth ?? 0,
      headerScroll: header?.scrollWidth ?? 0,
      viewsWidth: views?.clientWidth ?? 0,
      viewsScroll: views?.scrollWidth ?? 0,
      overlaps,
    };
  });
  expect(result.document, JSON.stringify(result)).toBeLessThanOrEqual(
    result.viewport,
  );
  expect(result.headerScroll, JSON.stringify(result)).toBeLessThanOrEqual(
    result.headerWidth,
  );
  expect(result.viewsScroll, JSON.stringify(result)).toBeLessThanOrEqual(
    result.viewsWidth,
  );
  expect(result.overlaps, JSON.stringify(result)).toEqual([]);
}

test("workspace layout review at desktop, laptop, and phone widths", async ({
  page,
}, testInfo) => {
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await expect(page.getByTestId("graph-canvas")).toBeVisible();

  for (const width of [1440, 768, 390]) {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
    for (const theme of ["Light", "Dark"] as const) {
      await chooseTheme(page, theme);
      await expect(
        page.getByRole("navigation", { name: "Workspace views" }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Architecture", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Overview", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Artifacts", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Report", exact: true }),
      ).toBeVisible();
      await expect(page.getByTestId("graph-canvas")).toBeVisible();
      await expectGraphFramed(page);
      await expectNoHorizontalOverflow(page);
      await page.screenshot({
        path: testInfo.outputPath(`graph-${width}-${theme.toLowerCase()}.png`),
        animations: "disabled",
      });
    }

    await graphNode(page, embedding.label);
    await expect(page.getByTestId("inspector")).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await page.screenshot({
      path: testInfo.outputPath(`selected-${width}.png`),
      animations: "disabled",
    });
    if (width >= 1024) {
      await expect(
        page.getByRole("complementary", { name: "Inspector" }),
      ).toBeVisible();
      await page
        .getByRole("button", { name: "Close analysis panel" })
        .last()
        .click();
    } else {
      await page.getByRole("button", { name: "Back to graph" }).click();
    }
  }
});

test("artifacts and report remain readable without horizontal overflow", async ({
  page,
}, testInfo) => {
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await mockService(page, { artifactPages: [artifactPage()] });
  await openAnalysis(page);

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
    await page.getByRole("link", { name: "Artifacts", exact: true }).click();
    await expect(page.getByTestId("artifact-inspection")).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Artifacts", exact: true }),
    ).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await page.screenshot({
      path: testInfo.outputPath(`artifacts-${width}.png`),
      animations: "disabled",
    });

    await page.getByRole("link", { name: "Report", exact: true }).click();
    await expect(page.getByTestId("report-checks")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await expectNoHorizontalOverflow(page);
    await page.screenshot({
      path: testInfo.outputPath(`report-${width}.png`),
      animations: "disabled",
    });
    await page.getByRole("link", { name: "Architecture", exact: true }).click();
  }
});
