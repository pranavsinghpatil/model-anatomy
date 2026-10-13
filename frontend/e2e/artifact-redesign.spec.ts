import { expect, test } from "@playwright/test";
import type { ArtifactInspectionResponse } from "../src/api/contract";
import { analysisFor, mockService } from "./service";
import { openAnalysis } from "./helpers";

const revision = "0123456789abcdef0123456789abcdef01234567";

test("artifact inventory filter stays page-local and checkpoint action focuses files", async ({
  page: browser,
}) => {
  const files = Array.from({ length: 50 }, (_, index) =>
    index === 0 ? "model-00001-of-00002.safetensors" : `config-${index}.json`,
  );
  const artifactPage: ArtifactInspectionResponse = {
    kind: "artifact_inspection",
    reason_code: "checkpoint_selection_required",
    message: "Choose one checkpoint variant to inspect its tensor headers.",
    revision,
    files,
    file_count: 51,
    tensors: [],
    tensor_count: 0,
    checkpoint_tensor_element_count: null,
    checkpoint_available: false,
    source_config: null,
    evidence: [],
    offset: 0,
    limit: 50,
  };
  const job = {
    ...analysisFor(),
    model_id: "acme/vision-model",
    architecture: null,
    revision,
    status: "inspected" as const,
    stage: "inspected",
    parameter_count: null,
    root_graph_id: null,
    inspection: {
      kind: "artifact_inspection" as const,
      reason_code: artifactPage.reason_code!,
      message: artifactPage.message!,
      file_count: artifactPage.file_count,
      tensor_count: artifactPage.tensor_count,
      checkpoint_tensor_element_count: null,
      checkpoint_available: false,
    },
  };

  await mockService(browser, { job, artifactPages: [artifactPage] });
  await openAnalysis(browser, job.id, "artifacts");

  const inspection = browser.getByTestId("artifact-inspection");
  await expect(
    inspection.getByRole("heading", { name: "Artifacts" }),
  ).toBeVisible();
  await expect(
    inspection.getByRole("heading", { name: "Architecture report" }),
  ).toHaveCount(0);

  const filter = inspection.getByRole("searchbox", {
    name: "Filter files on this page",
  });
  await filter.fill("model-00001");
  await expect(
    inspection.getByRole("link", { name: /model-00001/ }),
  ).toBeVisible();
  await expect(
    inspection.getByRole("link", { name: /config-1\.json/ }),
  ).toHaveCount(0);
  await expect(
    inspection.getByText("1–50 of 51", { exact: true }),
  ).toBeVisible();

  const review = inspection.getByRole("button", {
    name: "Review checkpoint files",
  });
  await review.click();
  const heading = inspection.getByRole("heading", {
    name: "Repository file inventory",
  });
  await expect(heading).toBeFocused();
  await expect(heading).toBeInViewport();
});
