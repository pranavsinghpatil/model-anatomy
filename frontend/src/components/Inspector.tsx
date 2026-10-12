import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import {
  ArrowLeft,
  ArrowRight,
  Boxes,
  ChevronRight,
  FileSearch,
  Layers,
} from "lucide-react";
import { api } from "../api/client";
import type {
  AnalysisDetail,
  Citation,
  Evidence,
  GraphNode,
  TensorMetadata,
} from "../api/contract";
import {
  count,
  dimensionCount,
  reportDimensionKeys,
  shape,
  valueText,
} from "../format";
import { ErrorNotice, IconButton, Loading, Warnings } from "./Common";
import { componentPurpose } from "./LearnGuide";
import "./inspection.css";

export const matchesCitation = (
  evidence: Pick<Evidence, "source" | "kind" | "locator">,
  citation: Citation,
) =>
  evidence.source === citation.source &&
  evidence.locator === citation.locator &&
  evidence.kind === citation.kind;

export function EvidenceList({
  items,
  citation,
}: {
  items: Evidence[];
  citation?: Citation | null;
}) {
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (citation)
      container.current
        ?.querySelector(".evidence-highlight")
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [citation, items]);
  return (
    <div className="evidence-list" ref={container}>
      {items.length ? (
        items.map((evidence, index) => (
          <details
            key={`${evidence.source}:${evidence.locator}:${index}`}
            className={`evidence-item ${citation && matchesCitation(evidence, citation) ? "evidence-highlight" : ""}`}
            open={
              citation && matchesCitation(evidence, citation) ? true : undefined
            }
          >
            <summary>
              <span className={`evidence-kind evidence-${evidence.kind}`}>
                {evidence.kind}
              </span>
              <span>{evidence.source}</span>
            </summary>
            <code className="locator">
              {evidence.locator || "Unknown locator"}
            </code>
            {Object.keys(evidence.details ?? {}).length > 0 && (
              <pre>{valueText(evidence.details)}</pre>
            )}
          </details>
        ))
      ) : (
        <p className="empty-copy">No evidence returned.</p>
      )}
    </div>
  );
}

function TensorRow({ tensor }: { tensor: TensorMetadata }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="tensor-row">
      <button
        className="tensor-summary"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        <code>{tensor.name}</code>
        <div>
          <span>{shape(tensor.shape)}</span>
          <span className="dtype">{tensor.dtype ?? "Unknown dtype"}</span>
        </div>
        <div className="tensor-count">
          {count(tensor.parameter_count)} params
          <ChevronRight size={13} className={expanded ? "rotate" : ""} />
        </div>
      </button>
      {expanded && <pre className="tensor-extra">{valueText(tensor)}</pre>}
    </div>
  );
}

export type InspectorTab = "module" | "tensors" | "evidence";

export function Inspector({
  analysis,
  analysisId,
  graphId,
  node,
  openGraph,
  citation,
  tab,
  onTabChange,
}: {
  analysis: AnalysisDetail;
  analysisId: string;
  graphId: string;
  node: GraphNode | null;
  openGraph: (node: GraphNode) => void;
  citation: Citation | null;
  tab?: InspectorTab;
  onTabChange?: (tab: InspectorTab) => void;
}) {
  const [internalTab, setInternalTab] = useState<InspectorTab>("module");
  const selectedTab = tab ?? internalTab;
  const changeTab = (value: string) => {
    const next = value as InspectorTab;
    setInternalTab(next);
    onTabChange?.(next);
  };
  const [offset, setOffset] = useState(0);
  const detail = useQuery({
    queryKey: ["node", analysisId, graphId, node?.id, offset],
    queryFn: ({ signal }) =>
      api.node(analysisId, graphId, node!.id, offset, signal),
    enabled: Boolean(node),
  });
  useEffect(() => {
    setOffset(0);
    if (tab === undefined) setInternalTab(citation ? "evidence" : "module");
  }, [analysisId, graphId, node?.id, citation, tab]);
  if (
    !node &&
    citation?.source === "stored_ir" &&
    citation.locator === "metadata"
  )
    return (
      <div className="inspector-content" data-testid="model-evidence">
        <h2>Model evidence</h2>
        <div className="citation-record">
          <span>
            {citation.source} / {citation.kind}
          </span>
          <code>{citation.locator}</code>
        </div>
        <h3>Stored analysis</h3>
        <dl className="properties">
          <dt>Model ID</dt>
          <dd data-testid="metadata-model_id">{analysis.model_id}</dd>
          <dt>Revision</dt>
          <dd data-testid="metadata-revision">
            {analysis.revision ?? "Unknown"}
          </dd>
          <dt>Architecture</dt>
          <dd data-testid="metadata-architecture">
            {analysis.architecture ?? "Unknown"}
          </dd>
          <dt>Parameters</dt>
          <dd data-testid="metadata-parameter_count">
            {count(analysis.parameter_count)}
          </dd>
        </dl>
        <h3>Dimensions</h3>
        <dl className="properties">
          {reportDimensionKeys(analysis).map((key) => (
            <div className="property-row" key={key}>
              <dt>
                {key === "heads" && Array.isArray(analysis.dimensions.heads)
                  ? "heads per stage"
                  : key}
              </dt>
              <dd data-testid={`metadata-${key}`}>
                {dimensionCount(analysis.dimensions[key])}
              </dd>
            </div>
          ))}
        </dl>
        {(analysis.engine_version ||
          (analysis.limitations?.length ?? 0) > 0) && (
          <section className="inspector-scope" data-testid="metadata-scope">
            <h3>Analysis scope</h3>
            {analysis.engine_version && (
              <p>
                Engine version: <code>{analysis.engine_version}</code>
              </p>
            )}
            {(analysis.limitations?.length ?? 0) > 0 && (
              <ul>
                {analysis.limitations?.map((limitation, index) => (
                  <li key={index}>{limitation}</li>
                ))}
              </ul>
            )}
          </section>
        )}
        <h3>Warnings</h3>
        {analysis.warnings.length ? (
          <div className="warnings" data-testid="metadata-warnings">
            <ul>
              {analysis.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="empty-copy" data-testid="metadata-warnings">
            No warnings returned.
          </p>
        )}
      </div>
    );
  if (!node)
    return (
      <div className="panel-empty">
        <FileSearch size={28} />
        <strong>Component details</strong>
        <p>No component selected</p>
      </div>
    );
  const resolved = detail.data?.node ?? node;
  const evidence = detail.data?.evidence ?? resolved.evidence;
  return (
    <div className="inspector" data-testid="inspector">
      <div className="inspector-heading">
        <span className="eyebrow">{resolved.kind.replaceAll("_", " ")}</span>
        <h2>{resolved.label}</h2>
        <p className="inspector-purpose">{componentPurpose(resolved)}</p>
        <details className="metadata">
          <summary>Module path</summary>
          <code className="inspector-long-path">
            {resolved.module_path ?? "Unknown module path"}
          </code>
        </details>
      </div>
      <Tabs.Root
        className="panel-tabs"
        value={selectedTab}
        onValueChange={changeTab}
      >
        <Tabs.List aria-label="Component detail views">
          <Tabs.Trigger value="module">Component</Tabs.Trigger>
          <Tabs.Trigger value="tensors">
            Tensors
            {detail.data && (
              <span>{count(detail.data.tensor_count, true)}</span>
            )}
          </Tabs.Trigger>
          <Tabs.Trigger value="evidence">
            Evidence<span>{evidence.length}</span>
          </Tabs.Trigger>
        </Tabs.List>
        {detail.isPending && <Loading text="Loading component evidence" />}
        {detail.error && (
          <ErrorNotice
            error={detail.error}
            retry={() => void detail.refetch()}
          />
        )}
        <div className="inspector-content">
          <Tabs.Content value="module">
            <>
              <section className="inspector-section">
                <h3>Inputs</h3>
                {resolved.inputs.length ? (
                  resolved.inputs.map((tensor, index) => (
                    <div className="port" key={index}>
                      <span>{tensor.semantic_name}</span>
                      <code>{shape(tensor.shape)}</code>
                      <small>{tensor.dtype ?? "Unknown dtype"}</small>
                    </div>
                  ))
                ) : (
                  <p className="empty-copy">
                    No input specifications returned.
                  </p>
                )}
              </section>
              <section className="inspector-section">
                <h3>Outputs</h3>
                {resolved.outputs.length ? (
                  resolved.outputs.map((tensor, index) => (
                    <div className="port" key={index}>
                      <span>{tensor.semantic_name}</span>
                      <code>{shape(tensor.shape)}</code>
                      <small>{tensor.dtype ?? "Unknown dtype"}</small>
                    </div>
                  ))
                ) : (
                  <p className="empty-copy">
                    No output specifications returned.
                  </p>
                )}
              </section>
              <div className="inspector-stat">
                <span>Parameters</span>
                <strong title={count(resolved.parameter_count)}>
                  {count(resolved.parameter_count)}
                </strong>
              </div>
              <section className="inspector-section">
                <h3>Source modules</h3>
                {resolved.source_modules.length ? (
                  <details className="metadata">
                    <summary>
                      {count(resolved.source_modules.length)} source module
                      paths
                    </summary>
                    <div>
                      {resolved.source_modules.map((path) => (
                        <code className="module-source" key={path}>
                          {path}
                        </code>
                      ))}
                    </div>
                  </details>
                ) : (
                  <p className="empty-copy">Unknown</p>
                )}
              </section>
              <details className="metadata">
                <summary>Component metadata</summary>
                <dl className="properties">
                  <dt>Type</dt>
                  <dd>{resolved.module_type ?? "Unknown"}</dd>
                  <dt>Confidence</dt>
                  <dd>
                    {resolved.confidence == null
                      ? "Unknown"
                      : `${(resolved.confidence * 100).toFixed(1)}%`}
                  </dd>
                  <dt>Repetition</dt>
                  <dd>
                    {resolved.repeat > 1
                      ? `${count(resolved.repeat)} instances / ${resolved.repetition?.mode ?? "unknown"}`
                      : "Single instance"}
                  </dd>
                  <dt>Node ID</dt>
                  <dd>
                    <code>{resolved.id}</code>
                  </dd>
                  <dt>Graph ID</dt>
                  <dd>
                    <code>{graphId}</code>
                  </dd>
                </dl>
              </details>
              {resolved.child_graph_id && (
                <button
                  className="expand-button"
                  onClick={() => openGraph(resolved)}
                >
                  <Layers size={15} /> Open nested graph{" "}
                  <ChevronRight size={15} />
                </button>
              )}
              {resolved.repetition && (
                <details className="metadata">
                  <summary>Repeated instance paths</summary>
                  <pre>{valueText(resolved.repetition.instance_paths)}</pre>
                </details>
              )}
              <details className="metadata">
                <summary>Module record</summary>
                <pre>
                  {detail.isPending
                    ? "Loading"
                    : valueText(detail.data?.module)}
                </pre>
              </details>
              {Object.keys(resolved.metadata).length > 0 && (
                <details className="metadata">
                  <summary>Semantic metadata</summary>
                  <pre>{valueText(resolved.metadata)}</pre>
                </details>
              )}
            </>
          </Tabs.Content>
          <Tabs.Content value="tensors">
            <>
              <div className="section-heading">
                <Boxes size={15} />
                <h3>Owned tensors</h3>
              </div>
              {detail.data && (
                <>
                  <div className="pagination">
                    <span>
                      {detail.data.tensor_count
                        ? `${count(offset + 1)}-${count(offset + detail.data.tensors.length)} of ${count(detail.data.tensor_count)}`
                        : "0 tensors"}
                    </span>
                    <IconButton
                      label="Previous tensor page"
                      disabled={offset === 0}
                      onClick={() =>
                        setOffset((value) => Math.max(0, value - 50))
                      }
                    >
                      <ArrowLeft size={14} />
                    </IconButton>
                    <IconButton
                      label="Next tensor page"
                      disabled={offset + 50 >= detail.data.tensor_count}
                      onClick={() => setOffset((value) => value + 50)}
                    >
                      <ArrowRight size={14} />
                    </IconButton>
                  </div>
                  {detail.data.tensors.length ? (
                    detail.data.tensors.map((tensor) => (
                      <TensorRow tensor={tensor} key={tensor.name} />
                    ))
                  ) : (
                    <p className="empty-copy">No owned tensors returned.</p>
                  )}
                </>
              )}
            </>
          </Tabs.Content>
          <Tabs.Content value="evidence">
            <>
              <h3>Provenance</h3>
              <EvidenceList items={evidence} citation={citation} />
              {citation &&
                !evidence.some((item) => matchesCitation(item, citation)) && (
                  <div className="citation-record">
                    <strong>Cited evidence</strong>
                    <span>
                      {citation.source} / {citation.kind}
                    </span>
                    <code>{citation.locator}</code>
                  </div>
                )}
            </>
          </Tabs.Content>
          {detail.data && <Warnings warnings={detail.data.warnings} />}
        </div>
      </Tabs.Root>
    </div>
  );
}
