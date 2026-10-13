import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import {
  ArrowRight,
  ExternalLink,
  FileSearch,
  ScanSearch,
  Search,
} from "lucide-react";
import { api } from "../api/client";
import type { InspectionSummary } from "../api/contract";
import { count, shape, valueText } from "../format";
import { ErrorNotice, IconButton, Loading } from "./Common";
import { EvidenceList } from "./Inspector";
import "./inspection.css";

const pageSize = 50;

function checkpointFileKind(file: string) {
  if (file.endsWith(".safetensors.index.json")) return "Safetensors index";
  if (file.endsWith(".safetensors")) return "Safetensors checkpoint";
  if (file.endsWith(".gguf")) return "GGUF checkpoint";
  return null;
}

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
  return `${offset + 1}–${Math.min(offset + limit, total)} of ${count(total)}`;
}

export function ArtifactInspection({
  analysisId,
  modelId,
  revision,
  summary,
  onInspectFile,
  inspectionDisabled,
  architecture = false,
}: {
  analysisId: string;
  modelId: string;
  revision: string | null;
  summary: InspectionSummary | null | undefined;
  onInspectFile: (url: string) => void;
  inspectionDisabled: boolean;
  architecture?: boolean;
}) {
  const [filesOffset, setFilesOffset] = useState(0);
  const [tensorsOffset, setTensorsOffset] = useState(0);
  const [tab, setTab] = useState("files");
  const [fileSearch, setFileSearch] = useState("");
  const [focusFileInventory, setFocusFileInventory] = useState(false);
  const filesHeadingRef = useRef<HTMLHeadingElement>(null);
  const filesInspection = useQuery({
    queryKey: ["artifacts", analysisId, "files", filesOffset],
    queryFn: ({ signal }) =>
      api.artifacts(analysisId, filesOffset, pageSize, signal),
    placeholderData: (previous) => previous,
  });
  const tensorsInspection = useQuery({
    queryKey: ["artifacts", analysisId, "tensors", tensorsOffset],
    queryFn: ({ signal }) =>
      api.artifacts(analysisId, tensorsOffset, pageSize, signal),
    enabled: tab === "tensors",
    placeholderData: (previous) => previous,
  });
  const page = filesInspection.data ?? tensorsInspection.data;
  const profile = page?.model_profile;
  const filesPage = filesInspection.data;
  const tensorsPage = tensorsInspection.data;
  const fileCount = page?.file_count ?? summary?.file_count ?? 0;
  const tensorCount = page?.tensor_count ?? summary?.tensor_count ?? 0;
  const pageRevision = page?.revision ?? revision;
  const message = page?.message ?? summary?.message;
  const reasonCode = page?.reason_code ?? summary?.reason_code;
  const pipelineSelectionRequired =
    reasonCode === "pipeline_components_require_selection" ||
    (profile?.architecture_blockers.includes(
      "pipeline_components_require_selection",
    ) ??
      false);
  const checkpointSelectionRequired =
    reasonCode === "checkpoint_selection_required";
  const selectionRequired =
    checkpointSelectionRequired || pipelineSelectionRequired;
  const inspectionError =
    filesInspection.error ?? tensorsInspection.error ?? undefined;
  const filteredFiles = (filesPage?.files ?? []).filter((file) =>
    file.toLocaleLowerCase().includes(fileSearch.trim().toLocaleLowerCase()),
  );

  useEffect(() => {
    if (!focusFileInventory || tab !== "files" || !filesPage) return;
    const frame = window.requestAnimationFrame(() => {
      const heading = filesHeadingRef.current;
      heading?.scrollIntoView({ behavior: "smooth", block: "start" });
      heading?.focus({ preventScroll: true });
      setFocusFileInventory(false);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [filesPage, focusFileInventory, tab]);

  return (
    <section className="artifact-inspection" data-testid="artifact-inspection">
      <header className="artifact-inspection-header">
        <div className="artifact-inspection-title">
          <FileSearch size={20} aria-hidden="true" />
          <div>
            <span className="eyebrow">
              {architecture ? "INSPECTION" : "ARTIFACT INSPECTION"}
            </span>
            <h1>Artifacts</h1>
            {!architecture && (
              <p>Repository inspected · no reconstructed graph</p>
            )}
          </div>
        </div>
        <details className="artifact-inspection-identity">
          <summary>
            <strong>{modelId}</strong>
            <code data-testid="artifact-revision">
              {pageRevision ?? "Unknown"}
            </code>
          </summary>
          <p>Verified revision used to inspect this repository.</p>
        </details>
      </header>

      {!architecture && (message || reasonCode) && (
        <section
          className="artifact-reason"
          aria-label="Architecture reconstruction status"
        >
          <strong>Architecture reconstruction unavailable</strong>
          {message && <p>{message}</p>}
          {reasonCode && (
            <details>
              <summary>Inspection reason</summary>
              <code>{reasonCode}</code>
            </details>
          )}
        </section>
      )}

      <dl
        className="artifact-inspection-stats"
        aria-label="Verified artifact counts"
        data-testid="artifact-inspection-stats"
      >
        <div>
          <dt>Files</dt>
          <dd data-testid="artifact-file-count" title={count(fileCount)}>
            {count(fileCount, true)}
          </dd>
        </div>
        <div>
          <dt>Tensor headers</dt>
          <dd data-testid="artifact-tensor-count" title={count(tensorCount)}>
            {count(tensorCount, true)}
          </dd>
        </div>
        <div>
          <dt>
            {page?.checkpoint_format === "gguf"
              ? "Selected file elements"
              : "Checkpoint elements"}
          </dt>
          <dd
            title={count(
              page?.checkpoint_format === "gguf"
                ? page.selected_file_tensor_element_count
                : (page?.checkpoint_tensor_element_count ??
                    summary?.checkpoint_tensor_element_count),
            )}
          >
            {count(
              page?.checkpoint_format === "gguf"
                ? page.selected_file_tensor_element_count
                : (page?.checkpoint_tensor_element_count ??
                    summary?.checkpoint_tensor_element_count),
              true,
            )}
          </dd>
        </div>
      </dl>

      {!architecture && (
        <details className="artifact-scope-disclosure">
          <summary>What this inspection establishes</summary>
          <p>
            It verifies the pinned file inventory and available bounded tensor
            headers. It does not establish graph structure, component ownership,
            execution order, or unique model parameter counts.
          </p>
        </details>
      )}

      {selectionRequired && (
        <section
          className="artifact-selection-action"
          aria-labelledby="artifact-selection-title"
          data-testid="checkpoint-selection-callout"
        >
          <div>
            <span className="artifact-outcome-label">Next step</span>
            <h3 id="artifact-selection-title">
              {pipelineSelectionRequired
                ? "Choose the checkpoint for the component you want to inspect."
                : "Choose a checkpoint file to read its tensor headers."}
            </h3>
            <p>
              {pipelineSelectionRequired
                ? "This repository declares multiple pipeline components. Nothing was selected automatically."
                : "Several checkpoint variants are present, so no variant was selected automatically."}
            </p>
            {pipelineSelectionRequired && (
              <code>pipeline_components_require_selection</code>
            )}
          </div>
          <button
            type="button"
            className="artifact-primary-action"
            onClick={() => {
              setTab("files");
              setFocusFileInventory(true);
            }}
          >
            Review checkpoint files <ArrowRight size={16} aria-hidden="true" />
          </button>
        </section>
      )}

      {inspectionError && (
        <ErrorNotice
          error={inspectionError}
          retry={() => {
            void filesInspection.refetch();
            void tensorsInspection.refetch();
          }}
        />
      )}
      {(filesInspection.isPending ||
        (tab === "tensors" && tensorsInspection.isPending)) && (
        <Loading text="Loading verified artifact metadata" />
      )}

      {page && (
        <>
          <Tabs.Root
            className="artifact-tabs"
            value={tab}
            onValueChange={setTab}
          >
            <Tabs.List
              className="artifact-tab-list"
              aria-label="Artifact inspection views"
            >
              <Tabs.Trigger value="files">
                Files <span>{count(fileCount, true)}</span>
              </Tabs.Trigger>
              <Tabs.Trigger value="tensors">
                Tensors <span>{count(tensorCount, true)}</span>
              </Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="files" aria-busy={filesInspection.isFetching}>
              {filesInspection.isPlaceholderData && (
                <Loading text="Loading next file page" />
              )}
              <section
                className="artifact-section"
                aria-labelledby="artifact-files-title"
              >
                <div className="artifact-section-heading">
                  <div>
                    <h3
                      id="artifact-files-title"
                      ref={filesHeadingRef}
                      tabIndex={-1}
                    >
                      Repository file inventory
                    </h3>
                    <p>Only files from the pinned revision are listed.</p>
                  </div>
                  <span>
                    {range(
                      filesPage?.offset ?? 0,
                      filesPage?.limit ?? page.limit,
                      page.file_count,
                    )}
                  </span>
                </div>
                <label className="artifact-file-search">
                  <Search size={16} aria-hidden="true" />
                  <span className="sr-only">Filter files on this page</span>
                  <input
                    type="search"
                    value={fileSearch}
                    onChange={(event) => setFileSearch(event.target.value)}
                    placeholder="Filter files on this page"
                  />
                </label>
                {filteredFiles.length ? (
                  <ul className="artifact-file-list">
                    {filteredFiles.map((file) => {
                      const href = pageRevision
                        ? artifactUrl(modelId, pageRevision, file)
                        : null;
                      const fileKind = checkpointFileKind(file);
                      return (
                        <li key={file}>
                          <div className="artifact-file-name">
                            {href ? (
                              <a
                                href={href}
                                target="_blank"
                                rel="noreferrer"
                                title={file}
                              >
                                <code>{file}</code>
                                <ExternalLink size={13} aria-hidden="true" />
                              </a>
                            ) : (
                              <code title={file}>{file}</code>
                            )}
                            {fileKind && <small>{fileKind}</small>}
                          </div>
                          {href && fileKind && (
                            <IconButton
                              label={`Inspect ${file} (${fileKind})`}
                              disabled={inspectionDisabled}
                              onClick={() => onInspectFile(href)}
                            >
                              <ScanSearch size={16} />
                              <span>Inspect</span>
                            </IconButton>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                ) : filesPage ? (
                  <p className="artifact-empty">
                    {fileSearch
                      ? "No files on this page match that filter."
                      : "No files returned for this page."}
                  </p>
                ) : null}
              </section>
              <nav
                className="artifact-pagination"
                aria-label="Repository file pages"
              >
                <span>
                  Showing{" "}
                  {range(
                    filesPage?.offset ?? 0,
                    filesPage?.limit ?? page.limit,
                    page.file_count,
                  )}{" "}
                  files
                </span>
                <div>
                  <button
                    type="button"
                    aria-label="Previous file page"
                    disabled={filesOffset === 0 || filesInspection.isFetching}
                    onClick={() => {
                      setFileSearch("");
                      setFilesOffset((value) => Math.max(0, value - pageSize));
                    }}
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    aria-label="Next file page"
                    disabled={
                      filesOffset + pageSize >= fileCount ||
                      filesInspection.isFetching
                    }
                    onClick={() => {
                      setFileSearch("");
                      setFilesOffset((value) => value + pageSize);
                    }}
                  >
                    Next
                  </button>
                </div>
              </nav>
            </Tabs.Content>
            <Tabs.Content
              value="tensors"
              aria-busy={tensorsInspection.isFetching}
            >
              {tensorsInspection.isPlaceholderData && (
                <Loading text="Loading next tensor page" />
              )}
              <section
                className="artifact-section"
                aria-labelledby="artifact-tensors-title"
              >
                <div className="artifact-section-heading">
                  <div>
                    <h3 id="artifact-tensors-title">Verified tensor headers</h3>
                    <p>
                      Header metadata only; payloads and computation are not
                      inspected.
                    </p>
                  </div>
                  <span>
                    {range(
                      tensorsPage?.offset ?? 0,
                      tensorsPage?.limit ?? page.limit,
                      page.tensor_count,
                    )}
                  </span>
                </div>
                <div className="artifact-table-wrap">
                  <table className="artifact-tensor-table">
                    <thead>
                      <tr>
                        <th scope="col">Tensor name</th>
                        <th scope="col">Shape</th>
                        <th scope="col">Type</th>
                        <th scope="col">Elements</th>
                        <th scope="col">Artifact</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(tensorsPage?.tensors ?? []).map((tensor) => {
                        const href = tensorArtifactUrl(
                          modelId,
                          tensorsPage?.revision ?? page.revision,
                          tensor.artifact,
                        );
                        return (
                          <tr key={`${tensor.artifact}:${tensor.name}`}>
                            <th scope="row" title={tensor.name}>
                              <code>{tensor.name}</code>
                            </th>
                            <td>
                              <code title={shape(tensor.shape)}>
                                {shape(tensor.shape)}
                              </code>
                              {tensor.shape_convention ===
                                "ggml_dimensions" && (
                                <small>GGML native axis order</small>
                              )}
                            </td>
                            <td>
                              <code>{tensor.dtype || "Unknown"}</code>
                            </td>
                            <td>{count(tensor.parameter_count)}</td>
                            <td title={tensor.artifact}>
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
                  {tensorsPage && !tensorsPage.tensors.length && (
                    <p className="artifact-empty">
                      No tensors returned for this page.
                    </p>
                  )}
                </div>
              </section>

              <nav
                className="artifact-pagination"
                aria-label="Checkpoint tensor pages"
              >
                <span>
                  Showing{" "}
                  {range(
                    tensorsPage?.offset ?? 0,
                    tensorsPage?.limit ?? page.limit,
                    page.tensor_count,
                  )}{" "}
                  tensors
                </span>
                <div>
                  <button
                    type="button"
                    aria-label="Previous tensor page"
                    disabled={
                      tensorsOffset === 0 || tensorsInspection.isFetching
                    }
                    onClick={() =>
                      setTensorsOffset((value) => Math.max(0, value - pageSize))
                    }
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    aria-label="Next tensor page"
                    disabled={
                      tensorsOffset + pageSize >= tensorCount ||
                      tensorsInspection.isFetching
                    }
                    onClick={() =>
                      setTensorsOffset((value) => value + pageSize)
                    }
                  >
                    Next
                  </button>
                </div>
              </nav>
            </Tabs.Content>
          </Tabs.Root>

          {profile && (
            <details className="artifact-source-details artifact-profile-details">
              <summary>Repository profile and declared components</summary>
              <section
                className="artifact-model-profile"
                aria-label="Declared model profile"
                data-testid="model-profile"
              >
                <dl>
                  <dt>Repository kind</dt>
                  <dd>{profile.repository_kind}</dd>
                  {profile.declared_task && (
                    <>
                      <dt>Declared task</dt>
                      <dd>{profile.declared_task}</dd>
                    </>
                  )}
                  {profile.declared_library && (
                    <>
                      <dt>Declared library</dt>
                      <dd>{profile.declared_library}</dd>
                    </>
                  )}
                  {profile.config_model_type && (
                    <>
                      <dt>Config model type</dt>
                      <dd>{profile.config_model_type}</dd>
                    </>
                  )}
                  {profile.declared_architectures.length > 0 && (
                    <>
                      <dt>Declared architectures</dt>
                      <dd>{profile.declared_architectures.join(", ")}</dd>
                    </>
                  )}
                  {profile.formats.length > 0 && (
                    <>
                      <dt>Formats</dt>
                      <dd>{profile.formats.join(", ")}</dd>
                    </>
                  )}
                  {profile.selected_file && (
                    <>
                      <dt>Selected file</dt>
                      <dd>
                        <code>{profile.selected_file}</code>
                      </dd>
                    </>
                  )}
                  {profile.selected_config && (
                    <>
                      <dt>Selected config</dt>
                      <dd>
                        <code>{profile.selected_config}</code>
                      </dd>
                    </>
                  )}
                </dl>
                {profile.components.length > 0 && (
                  <div>
                    <h4>Declared components</h4>
                    <ul>
                      {profile.components.map((component) => (
                        <li key={component.path}>
                          <strong>{component.path}</strong>
                          {(component.library || component.declared_class) && (
                            <span>
                              {[component.library, component.declared_class]
                                .filter(Boolean)
                                .join(" · ")}
                            </span>
                          )}
                          {component.checkpoint_files.length > 0 && (
                            <ul>
                              {component.checkpoint_files.map((file) => {
                                const href = pageRevision
                                  ? artifactUrl(modelId, pageRevision, file)
                                  : null;
                                return (
                                  <li key={file}>
                                    <code>{file}</code>
                                    {href && checkpointFileKind(file) && (
                                      <button
                                        type="button"
                                        disabled={inspectionDisabled}
                                        onClick={() => onInspectFile(href)}
                                      >
                                        Select this component checkpoint
                                      </button>
                                    )}
                                  </li>
                                );
                              })}
                            </ul>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {profile.architecture_blockers.length > 0 && (
                  <div>
                    <h4>Architecture blockers</h4>
                    <ul>
                      {profile.architecture_blockers.map((blocker) => (
                        <li key={blocker}>
                          <code>{blocker}</code>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </details>
          )}
          {page.selected_file_split_metadata && (
            <details className="artifact-source-details">
              <summary>Split checkpoint metadata</summary>
              <p>
                This report covers only the explicitly selected file in a split
                checkpoint.
              </p>
              <dl className="artifact-metadata-list">
                {Object.entries(page.selected_file_split_metadata).map(
                  ([key, value]) => (
                    <div key={key}>
                      <dt>
                        <code>{key}</code>
                      </dt>
                      <dd>{count(value)}</dd>
                    </div>
                  ),
                )}
              </dl>
            </details>
          )}
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
