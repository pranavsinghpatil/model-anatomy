import { expect, test } from "@playwright/test";
import type { AnalysisDetail, ArchitectureGraph } from "../src/api/contract";
import { expectGraphFramed, graphNode } from "./helpers";

test("live start does not restore a recent analysis without an explicit choice", async ({
  page,
}) => {
  test.skip(
    !process.env.E2E_LIVE_API,
    "Set E2E_LIVE_API=1 when the service runs on port 8001.",
  );
  const appRequests: string[] = [];
  page.on("request", (request) => appRequests.push(request.url()));
  expect((await page.request.get("/api/v1/examples")).status()).toBe(404);

  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /From open weights/ }),
  ).toBeVisible();
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  await page.goto("/app");
  await expect(
    page.getByRole("heading", { name: "Meet your model." }),
  ).toBeVisible();
  await expect(page.locator(".workspace-shell")).toHaveCount(0);
  await expect(page.getByTestId("graph-canvas")).toHaveCount(0);
  const continueRecent = page.getByRole("link", {
    name: "Continue recent analysis",
    exact: true,
  });
  if (await continueRecent.count()) {
    const href = await continueRecent.getAttribute("href");
    expect(href).toMatch(/^\/app\/analyses\//);
    await continueRecent.click();
    await expect(page).toHaveURL(/\/app\/analyses\/[^/?]+/);
  }
  expect(
    appRequests.some((url) => new URL(url).pathname === "/api/v1/examples"),
  ).toBe(false);
  expect(
    appRequests.some((url) => new URL(url).pathname.includes("/graphs/")),
  ).toBe(false);
});

for (const mode of ["Research", "Learn"] as const) {
  test(`live ${mode} mode submits, inspects a component, and exports`, async ({
    page,
  }, testInfo) => {
    test.setTimeout(240_000);
    test.skip(
      !process.env.E2E_LIVE_API,
      "Set E2E_LIVE_API=1 when the isolated live service is available.",
    );
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/app");
    await page
      .getByRole("group", { name: "Experience mode" })
      .getByRole("button", { name: mode, exact: true })
      .click();
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill("HuggingFaceTB/SmolLM2-360M");
    await expect(
      page.getByRole("button", { name: "Inspect model", exact: true }),
    ).toBeEnabled({ timeout: 60_000 });
    const createdResponse = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        response.url().endsWith("/api/v1/analyses"),
    );
    await page
      .getByRole("button", { name: "Inspect model", exact: true })
      .click();
    const response = await createdResponse;
    expect(response.status()).toBe(202);
    const created = (await response.json()) as AnalysisDetail;
    await expect(page).toHaveURL(new RegExp(`/app/analyses/${created.id}`));

    const defaultView = mode === "Research" ? "Architecture" : "Overview";
    await expect(
      page.getByRole("link", { name: defaultView, exact: true }),
    ).toHaveAttribute("aria-current", "page", { timeout: 130_000 });
    const analysisPath = `/api/v1/analyses/${created.id}`;
    await expect
      .poll(
        async () =>
          (
            (await page.request
              .get(analysisPath)
              .then((r) => r.json())) as AnalysisDetail
          ).status,
        { timeout: 180_000 },
      )
      .toBe("completed");
    const result = (await (
      await page.request.get(analysisPath)
    ).json()) as AnalysisDetail;
    expect(result.architecture).toMatch(/llama/i);
    expect(result.input_names).toEqual(["input_ids"]);
    expect(result.parameter_count).toBe(361_821_120);

    if (mode === "Learn") {
      await page
        .getByRole("link", { name: "Architecture", exact: true })
        .click();
    }
    await expect(page.getByTestId("graph-canvas")).toBeVisible({
      timeout: 130_000,
    });
    if (testInfo.project.name === "desktop") await expectGraphFramed(page);
    const root = (await (
      await page.request.get(
        `/api/v1/analyses/${created.id}/graphs/${result.root_graph_id}`,
      )
    ).json()) as ArchitectureGraph;
    const component =
      root.nodes.find((node) =>
        ["transformer_block", "attention", "mlp", "embedding"].includes(
          node.kind,
        ),
      ) ?? root.nodes[0];
    expect(component).toBeTruthy();
    await graphNode(page, component!.label);
    const inspector = page
      .getByRole("tabpanel", { name: "Details" })
      .getByTestId("inspector");
    await expect(inspector).toBeVisible();
    await expect(
      inspector.getByRole("heading", { name: component!.label }),
    ).toBeVisible();
    await inspector.getByRole("tab", { name: /^Tensors/ }).click();
    await expect(
      inspector.getByRole("heading", { name: "Owned tensors" }),
    ).toBeVisible();
    await inspector.getByRole("tab", { name: /^Evidence/ }).click();
    await expect(
      inspector.getByRole("heading", { name: "Provenance" }),
    ).toBeVisible();

    await page.getByLabel("Export analysis", { exact: true }).click();
    const download = page.waitForEvent("download");
    await page
      .getByRole("menuitem", { name: "Report JSON", exact: true })
      .click();
    expect((await download).suggestedFilename()).toMatch(/\.json$/);
    expect(errors).toEqual([]);
  });
}
