import { expect, test } from "@playwright/test";
import type { AnalysisDetail } from "../src/api/contract";

const cases = [
  {
    model: "google-t5/t5-small",
    revision: "df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    tensors: 132,
    architecture: "T5ForConditionalGeneration",
  },
  {
    model: "microsoft/swin-tiny-patch4-window7-224",
    revision: "d00d478bfaf1f417d34a1186673ad4b4be264c51",
    tensors: 233,
    architecture: "SwinForImageClassification",
  },
  {
    model: "openai/whisper-tiny",
    revision: "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    tensors: 167,
    architecture: "WhisperForConditionalGeneration",
  },
  {
    model: "MCG-NJU/videomae-base-finetuned-kinetics",
    revision: "488eb9a0565f257b32866000305c8178965eb9f6",
    tensors: 186,
    architecture: "VideoMAEForVideoClassification",
  },
  {
    model: "google/siglip-base-patch16-224",
    revision: "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    tensors: 408,
    architecture: "SiglipModel",
  },
];

for (const item of cases) {
  test(`live category: ${item.model}`, async ({ page }, info) => {
    test.skip(!process.env.E2E_LIVE_API, "Requires the real API and worker.");
    test.setTimeout(180_000);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/app");
    await page
      .getByLabel("Hugging Face model", { exact: true })
      .fill(item.model);
    const submit = page.getByRole("button", {
      name: "Inspect model",
      exact: true,
    });
    await expect(submit).toBeEnabled({ timeout: 30_000 });
    const submitted = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        response.url().endsWith("/api/v1/analyses"),
    );
    await submit.click();
    const response = await submitted;
    expect(response.status()).toBe(202);
    const created = (await response.json()) as AnalysisDetail;
    await expect(page.getByTestId("graph-canvas")).toBeVisible({
      timeout: 130_000,
    });
    const path = `/api/v1/analyses/${created.id}`;
    const detail = (await (
      await page.request.get(path)
    ).json()) as AnalysisDetail;
    expect(["completed", "partial"]).toContain(detail.status);
    expect(detail.architecture).toBe(item.architecture);
    expect(detail.quality?.checkpoint_parameter_coverage).toBe(1);
    const report = await (
      await page.request.get(`${path}/exports/json`)
    ).json();
    expect(report.metadata.revision).toBe(item.revision);
    expect(Object.keys(report.tensors)).toHaveLength(item.tensors);
    expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
    if (item.architecture === "VideoMAEForVideoClassification")
      expect(report.metadata.inputs.input_specs.pixel_values.shape).toBeNull();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      ),
    ).toBe(false);
    await page.screenshot({
      path: info.outputPath("live-category.png"),
      animations: "disabled",
    });
    expect(errors).toEqual([]);
  });
}

test("live diffusion component selection reads its actual nested checkpoint", async ({
  page,
}, info) => {
  test.skip(!process.env.E2E_LIVE_API, "Requires the real API and worker.");
  test.setTimeout(180_000);
  const model = "stable-diffusion-v1-5/stable-diffusion-v1-5";
  const file = "unet/diffusion_pytorch_model.safetensors";
  await page.goto("/app");
  await page.getByLabel("Hugging Face model", { exact: true }).fill(model);
  const submit = page.getByRole("button", {
    name: "Inspect model",
    exact: true,
  });
  await expect(submit).toBeEnabled({ timeout: 30_000 });
  await submit.click();
  const view = page.getByTestId("artifact-inspection");
  await expect(view).toContainText("pipeline_components_require_selection", {
    timeout: 130_000,
  });
  await expect(page.getByTestId("model-profile")).toContainText(
    "UNet2DConditionModel",
  );
  const submitted = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.url().endsWith("/api/v1/analyses"),
  );
  await page
    .getByRole("button", {
      name: `Inspect ${file} (Safetensors checkpoint)`,
      exact: true,
    })
    .click();
  const response = await submitted;
  expect(response.status()).toBe(202);
  const created = (await response.json()) as AnalysisDetail;
  await expect(page.getByTestId("model-profile")).toContainText(
    "unet/config.json",
    { timeout: 130_000 },
  );
  await view.getByRole("tab", { name: /^Tensors/ }).click();
  await expect(view.locator("tbody tr")).toHaveCount(50, { timeout: 130_000 });
  const report = await (
    await page.request.get(`/api/v1/analyses/${created.id}/exports/json`)
  ).json();
  expect(report.root_graph.nodes).toEqual([]);
  expect(Object.keys(report.tensors).length).toBeGreaterThan(0);
  expect(report.metadata.artifacts.weight_payload_bytes_read).toBe(0);
  expect(
    Object.values(report.tensors).every((tensor: any) =>
      tensor.artifact.endsWith(`/${file}`),
    ),
  ).toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await page.screenshot({
    path: info.outputPath("live-diffusion-component.png"),
    animations: "disabled",
  });
});
