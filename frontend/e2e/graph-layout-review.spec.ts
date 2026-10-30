import { expect, test } from "@playwright/test";
import { block, embedding, mockService, rootGraph } from "./service";
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
        const title = node.querySelector<HTMLElement>(".node-title");
        const flowElement = node.closest<HTMLElement>(".react-flow");
        if (!(node instanceof HTMLElement) || !title || !flowElement) {
          throw new Error(
            "graph node title and React Flow container must exist",
          );
        }
        const rect = node.getBoundingClientRect();
        const flow = flowElement.getBoundingClientRect();
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
        const title = node.querySelector<HTMLElement>(".node-title");
        if (!(node instanceof HTMLElement) || !title) {
          throw new Error("graph node title must exist");
        }
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

    const readGeometry = () =>
      page.getByTestId("graph-canvas").evaluate((canvas) => {
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
    let geometry = await readGeometry();
    // A responsive resize can briefly leave React Flow with zero height while
    // its parent layout settles. Wait only for valid geometry before checking
    // the same layout thresholds below.
    await expect
      .poll(async () => {
        geometry = await readGeometry();
        return (
          Number.isFinite(geometry.height) &&
          geometry.height > 0 &&
          Number.isFinite(geometry.viewport?.height) &&
          (geometry.viewport?.height ?? 0) > 0
        );
      })
      .toBe(true);
    if (!geometry.header || !geometry.viewport || !geometry.dock) {
      throw new Error("graph header, viewport, and dock geometry must exist");
    }
    expect(geometry.duplicateToolbar).toBe(false);
    expect(geometry.viewport.top).toBeGreaterThanOrEqual(
      geometry.header.bottom - 1,
    );
    expect(geometry.viewport.bottom).toBeLessThanOrEqual(geometry.dock.top + 1);
    expect(geometry.viewport.height / geometry.height).toBeGreaterThanOrEqual(
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
    if (!searchBounds) {
      throw new Error("component search results must have visible bounds");
    }
    expect(searchBounds.x + searchBounds.width).toBeLessThanOrEqual(width + 1);
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

test("keyboard selection exposes readable edge labels and reported endpoints", async ({
  page,
}, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);

  const edge = page.locator(".react-flow__edge").first();
  // A perfectly vertical SVG path has zero bounding-box width, so Playwright
  // reports its group as hidden even while the stroke is painted and focusable.
  await expect(edge).toBeAttached();
  await expect(edge.locator(".react-flow__edge-path")).toHaveAttribute(
    "d",
    /^M/,
  );
  await expect(edge).toHaveAttribute(
    "aria-label",
    /Token IDs.*Token Embedding.*Data flow.*output.*input/,
  );
  await edge.focus();
  await page.keyboard.press("Enter");

  const inspector = page.locator(".edge-inspector");
  await expect(inspector).toBeVisible();
  await expect(inspector).toContainText("Token IDs");
  await expect(inspector).toContainText("input");
  await expect(inspector).toContainText("Token Embedding");
  await expect(inspector).toContainText("embedding");
  await expect(inspector).toContainText("Data flow");
  await expect(inspector).toContainText("dataflow");

  // Mobile navigation replaces the graph with the inspector after selection.
  if (testInfo.project.name !== "desktop") return;
  await expect(page.locator(".graph-edge-label.is-selected")).toBeVisible();
  // Selecting another edge while a component remains selected must keep that
  // connection fully visible.
  await page.locator(".react-flow__node").first().focus();
  await page.keyboard.press("Enter");
  const unrelatedEdge = page.locator(".react-flow__edge").nth(3);
  await unrelatedEdge.focus();
  await page.keyboard.press("Enter");
  await expect(unrelatedEdge).toHaveClass(/selected/);
  await expect(unrelatedEdge.locator(".react-flow__edge-path")).toHaveAttribute(
    "style",
    /opacity: 1/,
  );
  const nearbyEdge = page.locator(".react-flow__edge").nth(1);
  await nearbyEdge.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".graph-edge-label.is-selected")).toBeVisible();
  await page.getByRole("button", { name: "Theme", exact: true }).click();
  await page.getByRole("menuitemradio", { name: "Light" }).click();
  await page.screenshot({
    path: testInfo.outputPath("graph-selected-light.png"),
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Theme", exact: true }).click();
  await page.getByRole("menuitemradio", { name: "Dark" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByText("Options", { exact: true }).click();
  await page
    .getByRole("button", { name: "Switch to horizontal layout" })
    .click();
  await expect(page.locator(".graph-edge-label.is-selected")).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("graph-selected-horizontal-dark.png"),
    animations: "disabled",
  });
});

test("selected edge endpoints stay in frame after layout changes despite a stale node selection", async ({
  page,
}, testInfo) => {
  await page.setViewportSize(
    testInfo.project.name === "mobile"
      ? { width: 390, height: 844 }
      : { width: 1280, height: 900 },
  );
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockService(page);
  await openAnalysis(page);

  const tokenIds = page
    .locator(".react-flow__node")
    .filter({ hasText: "Token IDs" });
  await tokenIds.focus();
  await page.keyboard.press("Enter");
  if (await page.getByRole("button", { name: "Back to graph" }).isVisible())
    await page.getByRole("button", { name: "Back to graph" }).click();
  await expect(page.locator('.model-node[data-node-id="input"]')).toHaveClass(
    /is-selected/,
  );

  const selectedEdge = page.locator(
    '.react-flow__edge[aria-label*="Token Embedding"][aria-label*="Decoder Block"][aria-label*="hidden_states"]',
  );
  await expect(selectedEdge).toBeAttached();
  await selectedEdge.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".edge-inspector")).toContainText("hidden_states");
  if (await page.getByRole("button", { name: "Back to graph" }).isVisible())
    await page.getByRole("button", { name: "Back to graph" }).click();

  await page.getByText("Options", { exact: true }).click();
  await page
    .getByRole("button", { name: "Switch to horizontal layout" })
    .click();
  await expect(page.locator(".graph-edge-label.is-selected")).toBeVisible();
  await expect(
    page.locator('.model-node[data-node-id="input"]'),
  ).not.toHaveClass(/is-selected/);

  const expectEdgeEndpointsInFrame = async () => {
    await expect
      .poll(() =>
        page.getByTestId("graph-canvas").evaluate((canvas) => {
          const viewport = canvas.querySelector<HTMLElement>(".react-flow");
          const source = canvas.querySelector<HTMLElement>(
            '.model-node[data-node-id="embedding"]',
          );
          const target = canvas.querySelector<HTMLElement>(
            '.model-node[data-node-id="block"]',
          );
          if (!viewport || !source || !target) return false;
          const bounds = viewport.getBoundingClientRect();
          return [source, target].every((node) => {
            const rect = node.getBoundingClientRect();
            return (
              rect.width > 0 &&
              rect.height > 0 &&
              rect.left >= bounds.left &&
              rect.right <= bounds.right &&
              rect.top >= bounds.top &&
              rect.bottom <= bounds.bottom
            );
          });
        }),
      )
      .toBe(true);
  };
  await expectEdgeEndpointsInFrame();
  await page
    .getByRole("button", { name: "Readable view", exact: true })
    .click();
  await expectEdgeEndpointsInFrame();

  await page.screenshot({
    path: testInfo.outputPath("selected-edge-horizontal-endpoints.png"),
    animations: "disabled",
  });
});

test("ELK routes named parallel ports to their exact handles in both directions", async ({
  page,
}) => {
  await mockService(page);
  await page.goto("/");
  const edge = rootGraph.edges[0];
  const graph = {
    ...rootGraph,
    id: "port-geometry",
    nodes: rootGraph.nodes.slice(0, 3),
    edges: [
      {
        ...edge,
        source: "input",
        target: "embedding",
        source_port: "first",
        target_port: "first",
      },
      {
        ...edge,
        source: "input",
        target: "embedding",
        source_port: "second",
        target_port: "second",
      },
      {
        ...edge,
        source: "embedding",
        target: "block",
        source_port: "",
        target_port: "default",
      },
      {
        ...edge,
        source: "input",
        target: "block",
        source_port: "branch",
        target_port: "",
      },
    ],
  };
  for (const direction of ["DOWN", "RIGHT"] as const) {
    const geometry = await page.evaluate(
      async ({ graph, direction }) => {
        const modulePath = "/src/graph/layout.ts";
        const { layoutGraph } = (await import(
          /* @vite-ignore */ modulePath
        )) as typeof import("../src/graph/layout");
        const result = await layoutGraph(graph, direction);
        return {
          nodes: result.nodes.map((node) => ({
            id: node.id,
            x: node.position.x,
            y: node.position.y,
            width: node.width!,
            height: node.height!,
            sourcePorts: node.data.sourcePorts,
            targetPorts: node.data.targetPorts,
          })),
          edges: result.edges.map((edge) => ({
            source: edge.source,
            target: edge.target,
            sourceHandle: edge.sourceHandle!,
            targetHandle: edge.targetHandle!,
            route: edge.data?.route,
            labelPosition: edge.data?.labelPosition,
          })),
        };
      },
      { graph, direction },
    );
    expect(geometry.edges).toHaveLength(4);
    const nodes = new Map(geometry.nodes.map((node) => [node.id, node]));
    const near = (actual: number, expected: number) =>
      expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1);
    for (const connection of geometry.edges) {
      const source = nodes.get(connection.source)!;
      const target = nodes.get(connection.target)!;
      const sourcePort = source.sourcePorts.find(
        (port) => port.id === connection.sourceHandle,
      )!;
      const targetPort = target.targetPorts.find(
        (port) => port.id === connection.targetHandle,
      )!;
      expect(sourcePort).toBeDefined();
      expect(targetPort).toBeDefined();
      expect(connection.route?.length).toBeGreaterThanOrEqual(2);
      expect(connection.labelPosition).toBeDefined();
      const label = connection.labelPosition!;
      for (const node of geometry.nodes) {
        const overlap =
          label.x + 110 > node.x &&
          label.x - 110 < node.x + node.width &&
          label.y + 12 > node.y &&
          label.y - 12 < node.y + node.height;
        expect(overlap, `label overlaps ${node.id} in ${direction}`).toBe(
          false,
        );
      }
      const start = connection.route![0];
      const end = connection.route!.at(-1)!;
      if (direction === "DOWN") {
        near(start.x, source.x + (source.width * sourcePort.position) / 100);
        near(start.y, source.y + source.height);
        near(end.x, target.x + (target.width * targetPort.position) / 100);
        near(end.y, target.y);
      } else {
        near(start.x, source.x + source.width);
        near(start.y, source.y + (source.height * sourcePort.position) / 100);
        near(end.x, target.x);
        near(end.y, target.y + (target.height * targetPort.position) / 100);
      }
      for (let index = 1; index < connection.route!.length; index += 1) {
        const previous = connection.route![index - 1];
        const current = connection.route![index];
        expect(previous.x === current.x || previous.y === current.y).toBe(true);
      }
    }
    const sourcePorts = geometry.nodes.find(
      (node) => node.sourcePorts.length >= 2,
    )!.sourcePorts;
    expect(new Set(sourcePorts.map((port) => port.id)).size).toBe(
      sourcePorts.length,
    );
  }
});
