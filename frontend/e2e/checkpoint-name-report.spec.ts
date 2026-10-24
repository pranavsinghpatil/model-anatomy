import { expect, test } from "@playwright/test";
import type {
  AnalysisDetail,
  ArtifactInspectionResponse,
} from "../src/api/contract";
import { openAnalysis } from "./helpers";
import { analysisFor, mockService } from "./service";

const revision = "1bae0e1088993b9b82de08f59de5be5a34e42170";
const indexSha =
  "abc09b0491e66b9f1c3d9dd635896523f2eaef3ca3904a91cfaa216390de7ba4";
const names: NonNullable<AnalysisDetail["verified_checkpoint_names"]> = {
  basis: "previously_audited_publisher_state_keys",
  index_sha256: indexSha,
  name_audit_document_sha256:
    "8f0f95214ad0dbafcf5ac1031deaba3d062b943ecfe86cd83678a9d44ace6512",
  state_key_count: 96085,
  component_count: 632,
  unmatched_state_key_count: 0,
  converter_renamed_names: 0,
  converter_skipped_names: 0,
  converter_name_collisions: 0,
  weight_payload_bytes_read_by_audit: 0,
};
const headers: NonNullable<AnalysisDetail["verified_checkpoint_headers"]> = {
  basis: "previously_verified_pinned_header_audit",
  index_sha256: indexSha,
  audit_document_sha256: "7".repeat(64),
  shard_count: 48,
  tensor_count: 96085,
  tensor_element_count_including_scales: 508182659298,
  header_bytes_read_by_audit: 10685312,
  weight_payload_bytes_read_by_audit: 0,
};

test("separate checkpoint-name evidence is readable in report and artifacts", async ({
  page,
}) => {
  const job: AnalysisDetail = {
    ...analysisFor(),
    model_id: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    revision,
    status: "partial",
    stage: "partial",
    parameter_count: null,
    quality: { mode: "publisher_reference_topology" },
    verified_checkpoint_headers: headers,
    verified_checkpoint_names: names,
  };
  const artifactPage: ArtifactInspectionResponse = {
    kind: "architecture",
    reason_code: null,
    message: null,
    revision,
    files: [],
    file_count: 0,
    tensors: [],
    tensor_count: 0,
    checkpoint_tensor_element_count: null,
    checkpoint_available: true,
    verified_checkpoint_headers: headers,
    verified_checkpoint_names: names,
    source_config: null,
    evidence: [],
    offset: 0,
    limit: 50,
  };
  await mockService(page, { job, artifactPages: [artifactPage] });
  await openAnalysis(page, "job-test", "report");
  const report = page.getByRole("region", {
    name: "Publisher checkpoint-name audit",
  });
  await expect(report).toContainText("96,085");
  await expect(report).toContainText("632");
  await expect(report).toContainText("parameter totals");
  await page.getByRole("link", { name: "Artifacts", exact: true }).click();
  const artifacts = page.getByRole("region", {
    name: "Publisher checkpoint-name audit",
  });
  await expect(artifacts).toContainText("96,085");
  await expect(artifacts).toContainText("no individual tensor records");
  await expect(page.getByTestId("artifact-tensor-count")).toHaveAttribute(
    "title",
    "0",
  );
});
