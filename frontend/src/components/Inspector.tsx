import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Boxes,
  ChevronRight,
  Cpu,
  FileSearch,
  GitCommit,
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
import { componentPurpose } from "./component-purpose";
import { ModuleProperties } from "./ModuleProperties";
import "./inspection.css";
import { nodeTitle } from "../graph/presentation";

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

function ParameterSplit({
  total,
  nodes,
}: {
  total: number;
  nodes: GraphNode[];
}) {
  const parts = nodes
    .filter((node) => node.parameter_count > 0)
    .map((node) => ({
      id: node.id,
      label: nodeTitle(node),
      count: node.parameter_count,
    }))
    .sort((a, b) => b.count - a.count);
  if (!parts.length) return null;
  const summed = parts.reduce((sum, part) => sum + part.count, 0);
  const overlaps = summed > total * 1.02;
  const partitions = summed >= total * 0.95 && summed <= total * 1.02;
  const shown = parts.slice(0, 8);
  return (
    <div className="overview-distribution-section">
      <div className="section-label-row">
        <span className="distribution-label">Parameters in this graph</span>
      </div>
      {partitions ? (
        <div
          className="distribution-bar"
          role="img"
          aria-label="Parameter allocation"
        >
          {shown.map((part, index) => (
            <div
              key={part.id}
              className="dist-seg"
              style={{
                width: `${(part.count / summed) * 100}%`,
                background: [
                  "#6366f1",
                  "#38bdf8",
                  "#a855f7",
                  "#f59e0b",
                  "#10b981",
                ][index % 5],
              }}
              title={`${part.label}: ${count(part.count)}`}
            />
          ))}
        </div>
      ) : null}
      <ul className="allocation-list">
        {shown.map((part) => (
          <li key={part.id}>
            <span>{part.label}</span>
            <strong>
              {count(part.count)} · {((part.count / total) * 100).toFixed(1)}%
            </strong>
          </li>
        ))}
      </ul>
      {parts.length > shown.length ? (
        <p className="allocation-note">
          {parts.length - shown.length} more components have parameters.
        </p>
      ) : null}
      {overlaps ? (
        <p className="allocation-note">
          Component counts overlap when weights are tied. Percents use the
          unique model total of {count(total)}.
        </p>
      ) : null}
    </div>
  );
}

export function Inspector({
  analysis,
  analysisId,
  graphId,
  nodes = [],
  node,
  openGraph,
  citation,
  tab,
  onTabChange,
}: {
  analysis: AnalysisDetail;
  analysisId: string;
  graphId: string;
  nodes?: GraphNode[];
  node: GraphNode | null;
  openGraph: (node: GraphNode) => void;
  citation: Citation | null;
  tab?: InspectorTab;
  onTabChange?: (tab: InspectorTab) => void;
}) {
  const [guideOpen, setGuideOpen] = useState(false);
  useEffect(() => setGuideOpen(false), [analysisId, graphId, node?.id]);
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
      <div className="panel-empty model-overview-panel" data-testid="inspector">
        <div className="overview-header-badge">
          <Layers size={14} />
          <span>Model Architecture Overview</span>
        </div>
        <h3 className="overview-model-title">{analysis.model_id}</h3>
        {analysis.architecture && (
          <p className="overview-model-arch">
            <code>{analysis.architecture}</code>
          </p>
        )}
        <div className="overview-metrics-grid">
          {analysis.parameter_count !== null &&
          analysis.parameter_count !== undefined ? (
            <div className="overview-metric-card" data-metric="parameters">
              <div className="metric-header-row">
                <span className="overview-metric-label">Parameters</span>
                <Cpu size={14} className="metric-icon" />
              </div>
              <strong className="overview-metric-value">
                {count(analysis.parameter_count)}
              </strong>
            </div>
          ) : null}
          {analysis.status ? (
            <div className="overview-metric-card" data-metric="status">
              <div className="metric-header-row">
                <span className="overview-metric-label">Status</span>
                <Activity size={14} className="metric-icon" />
              </div>
              <strong className="overview-metric-value status-value">
                <span className="status-indicator-dot" />
                {analysis.status}
              </strong>
            </div>
          ) : null}
          {analysis.inspection?.tensor_count ? (
            <div className="overview-metric-card" data-metric="tensors">
              <div className="metric-header-row">
                <span className="overview-metric-label">Tensors</span>
                <Layers size={14} className="metric-icon" />
              </div>
              <strong className="overview-metric-value">
                {analysis.inspection.tensor_count}
              </strong>
            </div>
          ) : null}
          {analysis.revision ? (
            <div className="overview-metric-card" data-metric="revision">
              <div className="metric-header-row">
                <span className="overview-metric-label">Revision</span>
                <GitCommit size={14} className="metric-icon" />
              </div>
              <strong className="overview-metric-value">
                {analysis.revision.slice(0, 8)}
              </strong>
            </div>
          ) : null}
        </div>
        {analysis.parameter_count ? (
          <ParameterSplit total={analysis.parameter_count} nodes={nodes} />
        ) : null}
        <div className="overview-hint-box">
          <FileSearch size={15} />
          <span>
            Select any component in the graph or component tree to inspect its
            shapes, weights, and tensor byte offsets.
          </span>
        </div>
      </div>
    );
  const resolved = detail.data?.node ?? node;
  const sourceTopology =
    analysis.quality?.mode === "publisher_reference_topology";
  const moduleArchitecture =
    analysis.quality?.mode === "static_module_architecture";
  const checkpointNameCount = resolved.metadata?.checkpoint_state_key_count;
  const evidence = detail.data?.evidence ?? resolved.evidence;
  return (
    <div className="inspector" data-testid="inspector">
      <div className="inspector-heading">
        <div className="inspector-kind-row">
          <span className="eyebrow">{resolved.kind.replaceAll("_", " ")}</span>
        </div>
        <h2>{nodeTitle(resolved)}</h2>
        {nodeTitle(resolved) !== resolved.label && (
          <code className="inspector-native-type">{resolved.label}</code>
        )}
        {resolved.parameter_count > 0 && analysis.parameter_count ? (
          <div className="node-share-banner">
            <span className="node-share-number">
              {count(resolved.parameter_count)} parameters
            </span>
            <span className="node-share-pct">
              (
              {(
                (resolved.parameter_count / analysis.parameter_count) *
                100
              ).toFixed(1)}
              % of model)
            </span>
          </div>
        ) : null}
        <details
          className="inspector-definition"
          open={guideOpen}
          onToggle={(event) => setGuideOpen(event.currentTarget.open)}
        >
          <summary>About this component type</summary>
          <p className="inspector-purpose">{componentPurpose(resolved)}</p>
        </details>
      </div>
      <Tabs.Root
        className="component-detail-tabs"
        value={selectedTab}
        onValueChange={changeTab}
      >
        <Tabs.List className="panel-tabs" aria-label="Component detail views">
          <Tabs.Trigger value="module">Component</Tabs.Trigger>
          <Tabs.Trigger value="tensors">
            {sourceTopology ? "Tensors unverified" : "Tensors"}{" "}
            {detail.data && !sourceTopology && (
              <span>{count(detail.data.tensor_count, true)}</span>
            )}
          </Tabs.Trigger>
          <Tabs.Trigger value="evidence">
            Evidence <span>{evidence.length}</span>
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
              <ModuleProperties node={resolved} />
              <details
                className="metadata"
                open={moduleArchitecture ? undefined : true}
              >
                <summary>Optional input and output specifications</summary>
                {moduleArchitecture && (
                  <p className="empty-copy">
                    Activation shapes are not collected by module architecture
                    inspection. Their absence does not make the architecture
                    partial.
                  </p>
                )}
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
                      {sourceTopology
                        ? "Input shape not verified."
                        : "No input specifications returned."}
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
                      {sourceTopology
                        ? "Output shape not verified."
                        : "No output specifications returned."}
                    </p>
                  )}
                </section>
              </details>
              <div className="inspector-stat">
                <span>Parameters</span>
                <strong
                  title={
                    sourceTopology
                      ? "Unverified"
                      : count(resolved.parameter_count)
                  }
                >
                  {sourceTopology ? "Unknown" : count(resolved.parameter_count)}
                </strong>
              </div>
              {typeof checkpointNameCount === "number" && (
                <div className="inspector-stat">
                  <span>Matched checkpoint names</span>
                  <strong>{count(checkpointNameCount)}</strong>
                </div>
              )}
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
                  <p className="empty-copy">
                    {sourceTopology
                      ? "Checkpoint ownership not verified."
                      : "Unknown"}
                  </p>
                )}
              </section>
              <details className="metadata">
                <summary>
                  {sourceTopology ? "Publisher reference path" : "Module path"}
                </summary>
                <code className="inspector-long-path">
                  {resolved.module_path ?? "Unknown module path"}
                </code>
              </details>
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
