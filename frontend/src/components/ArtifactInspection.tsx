import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, FileSearch, ScanSearch } from "lucide-react";
import { api } from "../api/client";
import type { InspectionSummary } from "../api/contract";
import { count, shape, valueText } from "../format";
import { ErrorNotice, IconButton, Loading } from "./Common";
import { EvidenceList } from "./Inspector";

const pageSize = 50;

function artifactUrl(modelId: string, revision: string, file: string) {
  const repo = modelId.split("/");
  const path = file.split("/");
  if (
    repo.length !== 2 ||
    repo.some((part) => !part || part === "." || part === "..") ||
    !path.length ||
    path.some(
      (part) => !part || part === "." || part === ".." || part.includes("\\"),
    )
  )
    return null;
  return `https://huggingface.co/${repo.map(encodeURIComponent).join("/")}/blob/${encodeURIComponent(revision)}/${path.map(encodeURIComponent).join("/")}`;
}

function tensorArtifactUrl(
  modelId: string,
  revision: string,
  artifact: string,
) {
  const repo = modelId.split("/");
  if (
    repo.length !== 2 ||
    repo.some((part) => !part || part === "." || part === "..")
  )
    return null;
  try {
    const url = new URL(artifact);
    const segments = url.pathname
      .split("/")
      .slice(1)
      .map((segment) => decodeURIComponent(segment));
    if (
      url.protocol !== "https:" ||
      url.hostname !== "huggingface.co" ||
      url.port ||
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      segments.length < 5 ||
      segments[0] !== repo[0] ||
      segments[1] !== repo[1] ||
      segments[2] !== "resolve" ||
      segments[3] !== revision ||
      segments
        .slice(4)
        .some(
          (part) =>
            !part || part === "." || part === ".." || /[\\/]/.test(part),
        )
    )
      return null;
    return url.href;
  } catch {
    return null;
  }
}

function range(offset: number, limit: number, total: number) {
  if (!total || offset >= total) return `0 of ${count(total)}`;
  return `${offset + 1}-${Math.min(offset + limit, total)} of ${count(total)}`;
}

export function ArtifactInspection({
  analysisId,
  modelId,
  revision,
  summary,
  onInspectFile,
  inspectionDisabled,
}: {
  analysisId: string;
  modelId: string;
  revision: string | null;
  summary: InspectionSummary | null | undefined;
  onInspectFile: (url: string) => void;
  inspectionDisabled: boolean;
}) {
  const [offset, setOffset] = useState(0);
  const inspection = useQuery({
    queryKey: ["artifacts", analysisId, offset],
    queryFn: ({ signal }) =>
      api.artifacts(analysisId, offset, pageSize, signal),
    placeholderData: (previous) => previous,
  });
  const page = inspection.data;
  const fileCount = page?.file_count ?? summary?.file_count ?? 0;
  const tensorCount = page?.tensor_count ?? summary?.tensor_count ?? 0;
  const maximumCount = Math.max(fileCount, tensorCount);
  const pageRevision = page?.revision ?? revision;
  const message = page?.message ?? summary?.message;
  const reasonCode = page?.reason_code ?? summary?.reason_code;

  return (
    <section className="artifact-inspection" data-testid="artifact-inspection">
      <header className="artifact-inspection-header">
        <div className="artifact-inspection-title">
          <FileSearch size={20} />
          <div>
            <span className="eyebrow">Artifacts inspected</span>
            <h2>Architecture unavailable</h2>
          </div>
        </div>
        <div className="artifact-inspection-stats">
          <div>
            <span>Files</span>
            <strong data-testid="artifact-file-count">
              {count(fileCount)}
            </strong>
          </div>
          <div>
            <span>Tensors</span>
            <strong data-testid="artifact-tensor-count">
              {count(tensorCount)}
            </strong>
          </div>
          <div>
            <span>
              {page?.checkpoint_format === "gguf"
                ? "Selected file elements"
                : "Checkpoint elements"}
            </span>
            <strong>
              {count(
                page?.checkpoint_format === "gguf"
                  ? page.selected_file_tensor_element_count
                  : (page?.checkpoint_tensor_element_count ??
                      summary?.checkpoint_tensor_element_count),
              )}
            </strong>
          </div>
        </div>
      </header>

      <div className="artifact-inspection-reason">
        {reasonCode && <code>{reasonCode}</code>}
        <p>{message ?? "Inspection details unavailable."}</p>
        {page?.selected_file_split_metadata && (
          <p>
            Split checkpoint: this report covers only the selected file.{" "}
            Reported parts:{" "}
            {page.selected_file_split_metadata["split.count"] ?? "Unknown"}.
          </p>
        )}
      </div>

      <div className="artifact-inspection-revision">
        <span>Verified revision</span>
        <code data-testid="artifact-revision">{pageRevision ?? "Unknown"}</code>
      </div>

      {inspection.error && (
        <ErrorNotice
          error={inspection.error}
          retry={() => void inspection.refetch()}
        />
      )}
      {inspection.isPending && <Loading text="Loading artifact metadata" />}

      {page && (
        <>
          <section
            className="artifact-section"
            aria-labelledby="artifact-files-title"
          >
            <div className="artifact-section-heading">
              <h3 id="artifact-files-title">Repository files</h3>
              <span>{range(offset, page.limit, page.file_count)}</span>
            </div>
            {page.files.length ? (
              <ul className="artifact-file-list">
                {page.files.map((file) => {
                  const href = artifactUrl(modelId, page.revision, file);
                  return (
                    <li key={file}>
                      {href ? (
                        <a href={href} target="_blank" rel="noreferrer">
                          <code>{file}</code>
                          <ExternalLink size={13} aria-hidden="true" />
                        </a>
                      ) : (
                        <code>{file}</code>
                      )}
                      {href && file.endsWith(".gguf") && (
                        <IconButton
                          label={`Inspect ${file}`}
                          disabled={inspectionDisabled}
                          onClick={() => onInspectFile(href)}
                        >
                          <ScanSearch size={16} />
                        </IconButton>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="artifact-empty">No files returned for this page.</p>
            )}
          </section>

          <section
            className="artifact-section"
            aria-labelledby="artifact-tensors-title"
          >
            <div className="artifact-section-heading">
              <h3 id="artifact-tensors-title">Checkpoint tensors</h3>
              <span>{range(offset, page.limit, page.tensor_count)}</span>
            </div>
            <div className="artifact-table-wrap">
              <table className="artifact-tensor-table">
                <thead>
                  <tr>
                    <th scope="col">Tensor</th>
                    <th scope="col">Shape</th>
                    <th scope="col">Dtype</th>
                    <th scope="col">Elements</th>
                    <th scope="col">Artifact</th>
                  </tr>
                </thead>
                <tbody>
                  {page.tensors.map((tensor) => {
                    const href = tensorArtifactUrl(
                      modelId,
                      page.revision,
                      tensor.artifact,
                    );
                    return (
                      <tr key={`${tensor.artifact}:${tensor.name}`}>
                        <th scope="row">
                          <code>{tensor.name}</code>
                        </th>
                        <td>
                          <code>{shape(tensor.shape)}</code>
                          {tensor.shape_convention === "ggml_dimensions" && (
                            <small>GGML axis order</small>
                          )}
                        </td>
                        <td>
                          <code>{tensor.dtype || "Unknown"}</code>
                        </td>
                        <td>{count(tensor.parameter_count)}</td>
                        <td>
                          {href ? (
                            <a href={href} target="_blank" rel="noreferrer">
                              <code>{tensor.artifact}</code>
                              <ExternalLink size={13} aria-hidden="true" />
                            </a>
                          ) : (
                            <code>{tensor.artifact}</code>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {!page.tensors.length && (
                <p className="artifact-empty">
                  No tensors returned for this page.
                </p>
              )}
            </div>
          </section>

          <nav className="artifact-pagination" aria-label="Artifact pages">
            <span>Page {Math.floor(offset / pageSize) + 1}</span>
            <div>
              <button
                type="button"
                aria-label="Previous artifact page"
                disabled={offset === 0 || inspection.isFetching}
                onClick={() =>
                  setOffset((value) => Math.max(0, value - pageSize))
                }
              >
                Previous
              </button>
              <button
                type="button"
                aria-label="Next artifact page"
                disabled={
                  offset + pageSize >= maximumCount || inspection.isFetching
                }
                onClick={() => setOffset((value) => value + pageSize)}
              >
                Next
              </button>
            </div>
          </nav>

          {page.source_config && (
            <details className="artifact-source-details">
              <summary>Source configuration</summary>
              <pre>{valueText(page.source_config)}</pre>
            </details>
          )}
          {page.evidence.length > 0 && (
            <details className="artifact-source-details">
              <summary>
                Artifact evidence ({count(page.evidence.length)})
              </summary>
              <EvidenceList items={page.evidence} />
            </details>
          )}
        </>
      )}
    </section>
  );
}
