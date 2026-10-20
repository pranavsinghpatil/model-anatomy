import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { mockService } from "./service";

const source = resolve("public/benchmark-catalog.json");
const evidence = existsSync(source)
  ? JSON.parse(readFileSync(source, "utf8"))
  : null;
if (!evidence && process.env.REQUIRE_BENCHMARK_EVIDENCE === "1") {
  throw new Error(
    "REQUIRE_BENCHMARK_EVIDENCE=1 requires public/benchmark-catalog.json with all 250 measured cases.",
  );
}
test.skip(
  !evidence,
  "The measured 250-case evidence file has not been published yet.",
);

test("every published repository appears exactly once across catalog pages", async ({
  page,
}, info) => {
  test.setTimeout(90_000);
  expect(evidence.results).toHaveLength(250);
  await mockService(page);
  await page.goto("/catalog");
  await expect(
    page.getByRole("heading", { name: "Measured compatibility results" }),
  ).toBeVisible();
  const pages = Math.ceil(evidence.results.length / 24);
  const seen: string[] = [];
  for (let index = 0; index < pages; index++) {
    await expect(
      page.getByText(`Page ${index + 1} of ${pages}`, { exact: true }),
    ).toBeVisible();
    const expected = evidence.results
      .slice(index * 24, (index + 1) * 24)
      .map((row: { model_id: string }) => row.model_id);
    await expect(page.locator(".benchmark-card h3")).toHaveText(expected);
    seen.push(...(await page.locator(".benchmark-card h3").allTextContents()));
    if (index === 0)
      await page.screenshot({ path: info.outputPath("measured-catalog.png") });
    if (index + 1 < pages)
      await page.getByRole("button", { name: "Next", exact: true }).click();
  }
  expect(new Set(seen).size).toBe(250);
  expect(seen).toEqual(
    evidence.results.map((row: { model_id: string }) => row.model_id),
  );
});

test("actual measured cards fit a 320px viewport", async ({ page }) => {
  await mockService(page);
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/catalog");
  await expect(page.locator(".benchmark-card")).toHaveCount(24);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

test("published measurements render their recorded outcomes, counts, and diagnostics", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await mockService(page);
  await page.goto("/catalog");
  const pages = Math.ceil(evidence.results.length / 24);
  const statusLabels: Record<string, string> = {
    completed: "Complete static report",
    partial: "Partial graph",
    inspected: "Artifacts inspected",
    http_access: "HF request failed",
    network: "Network request failed",
    artifact_error: "Artifact processing failed",
    unsupported: "Reconstruction unavailable",
    invalid_report: "Report validation failed",
    failure: "Inspection failed",
    memorylimit: "Memory limit reached",
    timeout: "Time limit reached",
  };
  const locale = await page.evaluate(() => navigator.language);
  const formatted = (value: number | null | undefined) =>
    value == null ? "Unknown" : value.toLocaleString(locale);
  for (let pageIndex = 0; pageIndex < pages; pageIndex++) {
    await expect(
      page.getByText(`Page ${pageIndex + 1} of ${pages}`, { exact: true }),
    ).toBeVisible();
    const rows = evidence.results.slice(pageIndex * 24, (pageIndex + 1) * 24);
    const cards = page.locator(".benchmark-card");
    await expect(cards).toHaveCount(rows.length);
    const rendered = await cards.evaluateAll((elements) =>
      elements.map((card) => ({
        status: card.querySelector(".benchmark-status")?.textContent?.trim(),
        facts: Array.from(
          card.querySelectorAll(".benchmark-facts strong"),
          (node) => node.textContent?.trim(),
        ),
        notes: Array.from(
          card.querySelectorAll(".benchmark-diagnostic-notes li"),
          (node) => node.textContent?.trim(),
        ),
      })),
    );
    expect(rendered).toEqual(
      rows.map(
        (row: {
          status: string;
          architecture?: string;
          graph_count?: number;
          tensor_count?: number;
          parameter_counts?: { unique?: number };
          diagnostic_notes?: string[];
        }) => ({
          status: statusLabels[row.status],
          facts: [
            row.architecture ?? "Not reconstructed",
            formatted(row.graph_count),
            formatted(row.tensor_count),
            formatted(row.parameter_counts?.unique),
          ],
          notes: row.diagnostic_notes ?? [],
        }),
      ),
    );
    for (const row of rows) {
      if (!["completed", "partial", "inspected"].includes(row.status)) {
        expect([
          row.graph_count,
          row.tensor_count,
          row.parameter_counts?.unique,
        ]).toEqual([undefined, undefined, undefined]);
        expect(row.warning_count).toBeUndefined();
      }
    }
    if (pageIndex === 0) {
      const noteIndex = rows.findIndex(
        (row: { diagnostic_notes?: string[] }) => row.diagnostic_notes?.length,
      );
      expect(noteIndex).toBeGreaterThanOrEqual(0);
      const card = cards.nth(noteIndex);
      await card.locator(".benchmark-details summary").click();
      await expect(card.getByText("Diagnostic summary")).toBeVisible();
      await card.locator(".benchmark-details summary").click();
    }
    if (pageIndex + 1 < pages)
      await page.getByRole("button", { name: "Next", exact: true }).click();
  }
});

test("published selected artifact and revision-only actions keep the exact pinned input", async ({
  page,
}) => {
  await mockService(page);
  await page.goto("/catalog");
  const selected = evidence.results.find((row: { reference?: string | null }) =>
    /\/(?:blob|resolve)\/[0-9a-f]{40}\//.test(row.reference ?? ""),
  );
  expect(
    selected,
    "The curated manifest includes selected artifact cases",
  ).toBeTruthy();
  const selectedIndex = evidence.results.indexOf(selected);
  const pageIndex = Math.floor(selectedIndex / 24);
  for (let index = 0; index < pageIndex; index++)
    await page.getByRole("button", { name: "Next", exact: true }).click();
  const selectedCard = page.locator(".benchmark-card").nth(selectedIndex % 24);
  await selectedCard
    .getByRole("button", { name: "Inspect pinned model" })
    .click();
  await expect(page.locator("#model-input")).toHaveValue(selected.reference);

  const revisionOnly = evidence.results.find(
    (row: { reference?: string | null }) => !row.reference,
  );
  expect(
    revisionOnly,
    "The curated manifest includes revision-only cases",
  ).toBeTruthy();
  await page.goto("/catalog");
  const index = evidence.results.indexOf(revisionOnly);
  for (let offset = 0; offset < Math.floor(index / 24); offset++)
    await page.getByRole("button", { name: "Next", exact: true }).click();
  await page
    .locator(".benchmark-card")
    .nth(index % 24)
    .getByRole("button", { name: "Inspect pinned model" })
    .click();
  await expect(page.locator("#model-input")).toHaveValue(
    `https://huggingface.co/${encodeURIComponent(revisionOnly.model_id).replaceAll("%2F", "/")}/tree/${encodeURIComponent(revisionOnly.revision)}`,
  );
});
