import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import type {
  ArchitectureGraph,
  ArtifactInspectionResponse,
  GraphNode,
} from "../src/api/contract";
import { count, shape } from "../src/format";
import { nodeTitle } from "../src/graph/presentation";
import {
  analysisFor,
  block,
  embedding,
  graphs,
  mockService,
  rootGraph,
} from "./service";
import {
  expectGraphFramed,
  graphNode,
  hasNoHorizontalOverflow,
  openAnalysis,
  UI_SCALE,
} from "./helpers";

const drawing = (page: import("@playwright/test").Page) =>
  page.getByTestId("anatomy-drawing");

async function openAnatomy(page: import("@playwright/test").Page) {
  await page
    .getByRole("group", { name: "Architecture presentation" })
    .getByRole("button", { name: "Anatomy", exact: true })
    .click();
  await expect(drawing(page)).toBeVisible();
}

async function selectFacet(
  page: import("@playwright/test").Page,
  node: GraphNode,
  isMobile: boolean,
) {
  const faces = page.locator(`.assembly-face[data-part-id="${node.id}"]`);
  let point: { x: number; y: number } | null = null;
  for (let index = 0; index < (await faces.count()) && !point; index += 1) {
    const face = faces.nth(index);
    if (!(await face.isVisible())) continue;
    point = await face.evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      for (let row = 1; row < 10; row += 1) {
        for (let column = 1; column < 10; column += 1) {
          const x = bounds.left + (bounds.width * column) / 10;
          const y = bounds.top + (bounds.height * row) / 10;
          if (document.elementFromPoint(x, y) === element) return { x, y };
        }
      }
      return null;
    });
  }
  expect(
    point,
    `Facet ${node.id} should expose a hit-test point`,
  ).not.toBeNull();
  if (isMobile) await page.touchscreen.tap(point!.x, point!.y);
  else await page.mouse.click(point!.x, point!.y);
  await expect(page.locator(".anatomy-part-summary strong")).toHaveText(
    nodeTitle(node),
  );
}

async function renderedPaint(page: import("@playwright/test").Page) {
  return drawing(page)
    .locator("svg")
    .evaluate(async (svg) => {
      const clone = svg.cloneNode(true) as SVGSVGElement;
      const source = [svg, ...svg.querySelectorAll("*")];
      const target = [clone, ...clone.querySelectorAll("*")];
      const properties = [
        "fill",
        "fill-opacity",
        "stroke",
        "stroke-opacity",
        "stroke-width",
        "stroke-linejoin",
        "stroke-linecap",
        "opacity",
        "display",
        "visibility",
      ];
      source.forEach((element, index) => {
        const computed = getComputedStyle(element);
        for (const property of properties) {
          (target[index] as SVGElement).style.setProperty(
            property,
            computed.getPropertyValue(property),
          );
        }
      });

      const width = 640;
      const height = 360;
      clone.setAttribute("width", String(width));
      clone.setAttribute("height", String(height));
      const xml = new XMLSerializer().serializeToString(clone);
      const url = URL.createObjectURL(
        new Blob([xml], { type: "image/svg+xml;charset=utf-8" }),
      );
      try {
        const image = new Image();
        image.src = url;
        await image.decode();
        const canvas = new OffscreenCanvas(width, height);
        const context = canvas.getContext("2d", { willReadFrequently: true });
        if (!context) throw new Error("Offscreen canvas is unavailable");
        context.drawImage(image, 0, 0, width, height);
        const pixels = context.getImageData(0, 0, width, height).data;
        const colors = new Set<string>();
        let painted = 0;
        for (let index = 0; index < pixels.length; index += 4) {
          if (pixels[index + 3] <= 16) continue;
          painted += 1;
          colors.add(
            `${pixels[index]},${pixels[index + 1]},${pixels[index + 2]}`,
          );
        }
        return { coverage: painted / (width * height), colors: colors.size };
      } finally {
        URL.revokeObjectURL(url);
      }
    });
}

async function setTheme(
  page: import("@playwright/test").Page,
  theme: "Light" | "Dark",
) {
  await page.getByRole("button", { name: "Theme", exact: true }).click();
  await page.getByRole("menuitemradio", { name: theme, exact: true }).click();
}

test("anatomy renders the real assembly and preserves graph context", async ({
  page,
  isMobile,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await expectGraphFramed(page);

  await page.locator(`.model-node[data-node-id="${embedding.id}"]`).click();
  const evidenceTab = page.getByRole("tab", { name: /Evidence/ });
  await evidenceTab.click();
  await expect(evidenceTab).toHaveAttribute("aria-selected", "true");
  const backToGraph = page.getByRole("button", {
    name: "Back to graph",
    exact: true,
  });
  if (await backToGraph.isVisible()) await backToGraph.click();

  const zoom = page.getByRole("button", { name: "Zoom in", exact: true });
  await zoom.click();
  const viewport = page.locator(".react-flow__viewport");
  await expect
    .poll(() => viewport.getAttribute("style"))
    .not.toBe("transform: translate(0px, 0px) scale(1);");
  const savedTransform = await viewport.getAttribute("style");

  await openAnatomy(page);
  await selectFacet(page, embedding, isMobile);
  const summary = page.locator(".anatomy-part-summary");
  await expect(summary).toContainText(embedding.kind.replaceAll("_", " "));
  await expect(summary.locator("dl")).toContainText(
    count(embedding.parameter_count, true),
  );
  await expect(summary.locator("dl")).toContainText(
    shape(embedding.outputs[0]?.shape ?? null),
  );

  const pixels = await renderedPaint(page);
  expect(pixels.coverage).toBeGreaterThan(0.01);
  expect(pixels.colors).toBeGreaterThan(4);
  await expect(drawing(page).locator("svg")).toHaveAttribute("role", "group");
  await expect(drawing(page).locator("svg")).toHaveAttribute(
    "aria-label",
    "Interactive model assembly",
  );

  if (await backToGraph.isVisible()) await backToGraph.click();
  await page
    .getByRole("group", { name: "Architecture presentation" })
    .getByRole("button", { name: "Graph", exact: true })
    .click();
  const restoredViewport = page.locator(".react-flow__viewport");
  await expect(restoredViewport).toHaveAttribute("style", savedTransform!);
  const selected = page.locator(
    `.model-node[data-node-id="${embedding.id}"].is-selected`,
  );
  await expect(selected).toBeVisible();
  if (isMobile) await selected.click();
  await expect(page.getByRole("tab", { name: /Evidence/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

test("anatomy controls, facets, keyboard inspection, and reduced motion work", async ({
  page,
  isMobile,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await openAnatomy(page);
  const svg = drawing(page).locator("svg");
  const initialViewBox = await svg.getAttribute("viewBox");
  const initialPaths = await svg
    .locator(".assembly-face")
    .evaluateAll((paths) => paths.map((path) => path.getAttribute("d")));
  await page.getByRole("button", { name: "Separate parts" }).click();
  await expect(drawing(page)).toHaveAttribute("data-motion", "reduced");
  await expect(drawing(page)).toHaveAttribute("data-pose", "1.000");
  const separatedPaths = await svg
    .locator(".assembly-face")
    .evaluateAll((paths) => paths.map((path) => path.getAttribute("d")));
  expect(separatedPaths).not.toEqual(initialPaths);
  expect(await svg.getAttribute("viewBox")).toBe(initialViewBox);
  await page.getByRole("button", { name: "Reassemble" }).click();
  await expect(drawing(page)).toHaveAttribute("data-pose", "0.000");
  expect(
    await svg
      .locator(".assembly-face")
      .evaluateAll((paths) => paths.map((path) => path.getAttribute("d"))),
  ).toEqual(initialPaths);
  expect(await svg.getAttribute("viewBox")).toBe(initialViewBox);

  const initialGeometry = await svg
    .locator(".assembly-face")
    .first()
    .getAttribute("d");
  await page.getByRole("button", { name: "Rotate assembly right" }).click();
  await expect
    .poll(() => svg.locator(".assembly-face").first().getAttribute("d"))
    .not.toBe(initialGeometry);
  const beforeZoom = await svg.getAttribute("viewBox");
  await page.getByRole("button", { name: "Zoom illustration in" }).click();
  await expect(svg).not.toHaveAttribute("viewBox", beforeZoom!);
  await page.getByRole("button", { name: "Fit assembly" }).click();
  await expect(svg).toHaveAttribute("viewBox", initialViewBox!);
  await page.getByRole("button", { name: "Reset illustration view" }).click();
  await expect(svg).toHaveAttribute("viewBox", initialViewBox!);
  await expect
    .poll(() => svg.locator(".assembly-face").first().getAttribute("d"))
    .toBe(initialGeometry);

  await selectFacet(page, block, isMobile);
  await expect(page.locator(".anatomy-readout")).toContainText(
    `×${count(block.repeat)} instances`,
  );
  const backToGraph = page.getByRole("button", {
    name: "Back to graph",
    exact: true,
  });
  if (await backToGraph.isVisible()) await backToGraph.click();
  const assembly = page.getByRole("group", {
    name: "Interactive model assembly",
  });
  await assembly.focus();
  const beforeArrow = await svg
    .locator(".assembly-face.is-selected[data-part-id]")
    .first()
    .getAttribute("data-part-id");
  await assembly.press("ArrowRight");
  await expect
    .poll(() =>
      svg
        .locator(".assembly-face.is-selected[data-part-id]")
        .first()
        .getAttribute("data-part-id"),
    )
    .not.toBe(beforeArrow);
  const keyboardSelected = await svg
    .locator(".assembly-face.is-selected[data-part-id]")
    .first()
    .getAttribute("data-part-id");
  const keyboardNode = rootGraph.nodes.find(
    (node) => node.id === keyboardSelected,
  )!;
  await assembly.press("Enter");
  await expect(page.getByTestId("inspector").locator("h2")).toHaveText(
    nodeTitle(keyboardNode),
  );
  if (await backToGraph.isVisible()) await backToGraph.click();
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(drawing(page)).toHaveAttribute("data-motion", "visible");
  const drawingElement = drawing(page);
  const originalStyle = await drawingElement.getAttribute("style");
  await page.getByRole("button", { name: "Separate parts" }).click();
  await drawingElement.evaluate((element, scale) => {
    (element as HTMLElement).style.position = "fixed";
    (element as HTMLElement).style.top = `calc(120vh / ${scale})`;
  }, UI_SCALE);
  await expect(drawing(page)).toHaveAttribute("data-motion", "paused");
  const offscreenPose = await drawing(page).getAttribute("data-pose");
  await page.waitForTimeout(200);
  await expect(drawing(page)).toHaveAttribute("data-pose", offscreenPose!);
  await drawingElement.evaluate((element, style) => {
    (element as HTMLElement).setAttribute("style", style ?? "");
  }, originalStyle);
});

test("nested anatomy loads reported components lazily and restores the parent pose", async ({
  page,
  isMobile,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const requests = await mockService(page);
  await openAnalysis(page);
  await openAnatomy(page);

  const childId = block.child_graph_id!;
  const child = graphs.get(childId)!;
  const attention = child.nodes.find((node) => node.kind === "self_attention")!;
  const mlp = child.nodes.find((node) => node.kind === "mlp")!;
  expect(attention).toBeTruthy();
  expect(mlp).toBeTruthy();
  await expect(
    drawing(page).locator(`.assembly-face[data-part-id="${attention.id}"]`),
  ).toHaveCount(0);
  await expect(
    drawing(page).locator(`.assembly-face[data-part-id="${mlp.id}"]`),
  ).toHaveCount(0);
  expect(
    requests.some((request) => request.path.includes(`/graphs/${childId}`)),
  ).toBe(false);

  await selectFacet(page, block, isMobile);
  const backToGraph = page.getByRole("button", {
    name: "Back to graph",
    exact: true,
  });
  if (await backToGraph.isVisible()) await backToGraph.click();
  await page.getByRole("button", { name: "Separate parts" }).click();
  await expect(drawing(page)).toHaveAttribute("data-pose", "1.000");
  const parentViewBox = await drawing(page)
    .locator("svg")
    .getAttribute("viewBox");
  await page
    .getByRole("button", { name: "Explore inside", exact: true })
    .click();
  await expect(
    page.getByRole("navigation", { name: "Graph path" }).getByRole("button", {
      name: block.label,
      exact: true,
    }),
  ).toBeVisible();
  await expect
    .poll(() =>
      requests.some((request) => request.path.includes(`/graphs/${childId}`)),
    )
    .toBe(true);
  await expect
    .poll(() =>
      drawing(page)
        .locator(`.assembly-face[data-part-id="${attention.id}"]`)
        .count(),
    )
    .toBeGreaterThan(0);
  await expect
    .poll(() =>
      drawing(page).locator(`.assembly-face[data-part-id="${mlp.id}"]`).count(),
    )
    .toBeGreaterThan(0);

  const rootCrumb = page
    .getByRole("navigation", { name: "Graph path" })
    .getByRole("button")
    .first();
  await rootCrumb.click();
  await expect(drawing(page)).toHaveAttribute("data-pose", "1.000");
  await expect(drawing(page).locator("svg")).toHaveAttribute(
    "viewBox",
    parentViewBox!,
  );
});

test("narrow anatomy keeps a long parent breadcrumb clickable", async ({
  page,
  isMobile,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  const job = {
    ...analysisFor("job-test"),
    model_id: "HuggingFaceTB/SmolLM2-135M",
  };
  await mockService(page, { job });
  await openAnalysis(page);
  await openAnatomy(page);
  await selectFacet(page, block, isMobile);
  const backToGraph = page.getByRole("button", {
    name: "Back to graph",
    exact: true,
  });
  if (await backToGraph.isVisible()) await backToGraph.click();
  await page
    .getByRole("button", { name: "Explore inside", exact: true })
    .click();
  const path = page.getByRole("navigation", { name: "Graph path" });
  await expect(
    path.getByRole("button", { name: block.label, exact: true }),
  ).toBeVisible();
  await path.getByRole("button", { name: "SmolLM2-135M", exact: true }).click();
  await expect(path.getByRole("button")).toHaveCount(1);
  await expect(
    drawing(page)
      .locator(`.assembly-face[data-part-id="${embedding.id}"]`)
      .first(),
  ).toBeVisible();
});

test("partial shape facts stay unknown and reported repeats have bounded geometry", async ({
  page,
  isMobile,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const stressGraph: ArchitectureGraph = {
    ...rootGraph,
    nodes: rootGraph.nodes.map((node) =>
      node.id === block.id ? { ...node, repeat: 1000 } : node,
    ),
  };
  await mockService(page);
  await page.route("**/api/v1/analyses/job-test/graphs/**", async (route) => {
    if (new URL(route.request().url()).pathname.endsWith(`/${rootGraph.id}`))
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(stressGraph),
      });
    return route.fallback();
  });
  await openAnalysis(page);
  await openAnatomy(page);
  await selectFacet(page, block, isMobile);
  await expect(page.locator(".anatomy-readout")).toContainText(
    "×1,000 instances",
  );
  expect(await drawing(page).locator(".assembly-face").count()).toBeLessThan(
    2000,
  );

  const partialJob = {
    ...analysisFor(),
    status: "partial" as const,
    stage: "partial",
  };
  await page.goto("about:blank");
  await mockService(page, { job: partialJob });
  const partialGraph: ArchitectureGraph = {
    ...rootGraph,
    nodes: rootGraph.nodes.map((node) =>
      node.id === embedding.id
        ? {
            ...node,
            outputs: node.outputs.map((tensor, index) =>
              index === 0 ? { ...tensor, shape: null } : tensor,
            ),
          }
        : node,
    ),
  };
  await page.route("**/api/v1/analyses/job-test/graphs/**", async (route) => {
    if (new URL(route.request().url()).pathname.endsWith(`/${rootGraph.id}`))
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(partialGraph),
      });
    return route.fallback();
  });
  await openAnalysis(page);
  await openAnatomy(page);
  await expect(page.locator(".anatomy-boundary")).toContainText(
    "Some architecture checks remain incomplete. Open the Report view for the recorded reasons. Activation shapes are outside these checks.",
  );
  await selectFacet(page, embedding, isMobile);
  await expect(page.locator(".anatomy-part-summary dl")).toContainText(
    "Parameters",
  );
  await expect(page.locator(".anatomy-part-summary dl")).not.toContainText(
    "First output shape",
  );
});

test("artifact-only reports do not expose the anatomy scene", async ({
  page,
}) => {
  const revision = "0123456789abcdef0123456789abcdef01234567";
  const job = {
    ...analysisFor(),
    architecture: null,
    status: "inspected" as const,
    stage: "inspected",
    root_graph_id: null,
    parameter_count: null,
  };
  const artifacts: ArtifactInspectionResponse = {
    kind: "artifact_inspection",
    reason_code: "architecture_not_supported",
    message:
      "Artifact metadata is available; no architecture graph was produced.",
    revision,
    files: ["config.json"],
    file_count: 1,
    tensors: [],
    tensor_count: 0,
    checkpoint_tensor_element_count: null,
    checkpoint_available: false,
    source_config: null,
    evidence: [],
    offset: 0,
    limit: 50,
  };
  await mockService(page, { job, artifactPages: [artifacts] });
  await openAnalysis(page, "job-test", "artifacts");
  await expect(page.getByTestId("artifact-inspection")).toBeVisible();
  await expect(page.locator(".scene-switcher")).toHaveCount(0);
  await expect(page.getByTestId("anatomy-scene")).toHaveCount(0);
  await expect(page.getByTestId("anatomy-drawing")).toHaveCount(0);
});

test("anatomy stays usable across viewport widths and themes", async ({
  page,
}) => {
  // Axe scans the detailed SVG scene twice; allow enough time under parallel CI load.
  test.setTimeout(120_000);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);
  await openAnatomy(page);

  for (const width of [320, 390, 768, 1024, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    for (const theme of ["Light", "Dark"] as const) {
      await setTheme(page, theme);
      await expect
        .poll(
          () => page.evaluate(hasNoHorizontalOverflow),
          `${width}px ${theme} anatomy width`,
        )
        .toBe(true);
      const sceneBounds = await page.getByTestId("anatomy-scene").boundingBox();
      expect(sceneBounds).not.toBeNull();
      expect(sceneBounds!.x).toBeGreaterThanOrEqual(0);
      expect(sceneBounds!.x + sceneBounds!.width).toBeLessThanOrEqual(
        width + 1,
      );
    }
  }

  await page.setViewportSize({ width: 1440, height: 900 });
  await setTheme(page, "Light");
  const light = await new AxeBuilder({ page })
    .include('[data-testid="anatomy-scene"]')
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(light.violations).toEqual([]);
  await setTheme(page, "Dark");
  const dark = await new AxeBuilder({ page })
    .include('[data-testid="anatomy-scene"]')
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(dark.violations).toEqual([]);
});
