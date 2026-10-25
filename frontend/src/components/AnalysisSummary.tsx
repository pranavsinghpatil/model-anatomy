import { Link } from "react-router-dom";
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Cpu,
  FileCheck2,
  FileSearch,
  Network,
  ShieldCheck,
  Zap,
} from "lucide-react";
import type { AnalysisDetail } from "../api/contract";
import { count, dimensionCount, reportDimensionKeys } from "../format";
import { CopyButton } from "./ui";
import { Status } from "./Common";
import "./analysis-summary.css";

export function Overview({ detail }: { detail: AnalysisDetail }) {
  const artifact = detail.status === "inspected";
  const sourceTopology =
    detail.quality?.mode === "publisher_reference_topology";
  const inspection = detail.inspection;
  const modelName = detail.model_id.split("/").at(-1) ?? detail.model_id;

  return (
    <div
      className={`content-page overview-page summary-page${artifact ? " is-artifact-only" : ""}`}
    >
      <header className="summary-heading">
        <div>
          <span className="eyebrow">MODEL ORIENTATION</span>
          <h1>{modelName}</h1>
          <p>{detail.model_id}</p>
        </div>
        <Status status={detail.status} />
      </header>

      <section className="summary-identity" aria-label="Analysis identity">
        <div>
          <span className="eyebrow">
            {detail.model_id.split("/")[0] || "Model repository"}
          </span>
          <h2>
            {artifact
              ? "Artifact inspection"
              : (detail.architecture ?? "Architecture unknown")}
          </h2>
        </div>
        <div className="summary-revision">
          <span>Verified revision</span>
          <code data-testid="model-revision">
            {detail.revision ?? "Unknown"}
          </code>
          {detail.revision && (
            <CopyButton value={detail.revision} label="Copy revision" />
          )}
        </div>
      </section>

      {artifact ? (
        <section
          className="summary-artifact-outcome"
          aria-labelledby="summary-artifact-title"
        >
          <div className="summary-artifact-mark">
            <FileSearch size={20} aria-hidden="true" />
          </div>
          <div className="summary-artifact-copy">
            <span className="eyebrow">NO ARCHITECTURE GRAPH</span>
            <h2 id="summary-artifact-title">
              The repository was verified at this revision.
            </h2>
            <p>
              {inspection?.message ??
                "Architecture reconstruction was unavailable for this inspection."}
            </p>
            <p>
              File inventory and bounded tensor headers are evidence about the
              stored artifacts. They do not establish component ownership,
              computation order, or a unique model parameter count.
            </p>
            {inspection?.reason_code && (
              <code className="summary-reason-code">
                {inspection.reason_code}
              </code>
            )}
          </div>
          <dl className="summary-artifact-facts">
            <div>
              <dt>Repository files</dt>
              <dd>{count(inspection?.file_count)}</dd>
            </div>
            <div>
              <dt>Tensor headers</dt>
              <dd>{count(inspection?.tensor_count)}</dd>
            </div>
            <div>
              <dt>Checkpoint selected</dt>
              <dd>{inspection?.checkpoint_available ? "Yes" : "No"}</dd>
            </div>
          </dl>
        </section>
      ) : (
        <dl className="model-metrics summary-metrics">
          <div>
            <dt>Parameters</dt>
            <dd title={count(detail.parameter_count)}>
              {count(detail.parameter_count, true)}
            </dd>
          </div>
          {reportDimensionKeys(detail).map((key) => (
            <div key={key} data-testid={`model-metric-${key}`}>
              <dt>
                {key === "hidden"
                  ? "Hidden size"
                  : key === "heads" && Array.isArray(detail.dimensions.heads)
                    ? "Heads per stage"
                    : key === "vocabulary"
                      ? "Vocabulary"
                      : key[0].toUpperCase() + key.slice(1)}
              </dt>
              <dd>{dimensionCount(detail.dimensions[key])}</dd>
            </div>
          ))}
        </dl>
      )}

      {!artifact && (
        <div className="summary-specs-banner" aria-label="Model specifications">
          <div className="spec-cell">
            <span className="spec-label">Architecture Class</span>
            <span className="spec-val font-mono">
              {detail.architecture ?? "Unknown"}
            </span>
          </div>
          <div className="spec-cell">
            <span className="spec-label">Extraction Pipeline</span>
            <span className="spec-val">
              {sourceTopology
                ? "Reviewed publisher source · partial"
                : "Native PyTorch modules + checkpoint metadata"}
            </span>
          </div>
          <div className="spec-cell">
            <span className="spec-label">Input Tensors</span>
            <span className="spec-val font-mono">
              {detail.input_names?.length
                ? detail.input_names.join(", ")
                : "Not verified"}
            </span>
          </div>
          <div className="spec-cell">
            <span className="spec-label">Evidence Provenance</span>
            <span className="spec-val">
              {sourceTopology
                ? "Pinned config and publisher source"
                : "Safetensors Range-Stream"}
            </span>
          </div>
        </div>
      )}

      <section className="explore-next summary-next">
        <div className="section-title">
          <div>
            <BookOpen size={18} aria-hidden="true" />
            <h2>Continue this inspection</h2>
          </div>
          <p>
            {artifact
              ? "Start with the verified files, then inspect available tensor headers."
              : "Move from the model summary to its structure, artifacts, or audit report."}
          </p>
        </div>
        <div className="next-actions">
          <Link
            className={artifact ? "summary-primary-link" : undefined}
            to={`?view=${artifact ? "artifacts" : "architecture"}`}
          >
            {artifact ? (
              <FileSearch size={21} aria-hidden="true" />
            ) : (
              <Network size={21} aria-hidden="true" />
            )}
            <div>
              <h3>
                {artifact
                  ? "Inspect files and choose a checkpoint"
                  : "Explore the architecture"}
              </h3>
              <p>
                {artifact
                  ? "Selection is explicit when a repository has multiple checkpoints or components."
                  : "Select a component to review its shape, tensors, and supporting evidence."}
              </p>
            </div>
            <ArrowRight size={18} aria-hidden="true" />
          </Link>
          {!artifact && (
            <Link to="?view=artifacts">
              <FileSearch size={21} aria-hidden="true" />
              <div>
                <h3>Inspect Safetensors & files</h3>
                <p>
                  Tensor headers, shapes, and shard offsets without weight
                  downloads.
                </p>
              </div>
              <ArrowRight size={18} aria-hidden="true" />
            </Link>
          )}
          <Link to="?view=report">
            <FileCheck2 size={21} aria-hidden="true" />
            <div>
              <h3>Review report scope</h3>
              <p>
                See what the recorded checks establish and which limits apply.
              </p>
            </div>
            <ArrowRight size={18} aria-hidden="true" />
          </Link>
        </div>
      </section>

      {!artifact && (
        <div className="scope-note">
          <strong>Static reconstruction</strong>
          <p>
            Recorded structure and coverage describe this inspection. They do
            not establish runtime behavior for every possible execution path.
          </p>
        </div>
      )}
    </div>
  );
}

const coverage = (value: number | null | undefined) =>
  value == null ? "Unknown" : `${(value * 100).toFixed(1)}%`;
const bool = (value: boolean | null | undefined) =>
  value == null ? "Unknown" : value ? "Yes" : "No";

export function Report({ detail }: { detail: AnalysisDetail }) {
  const q = detail.quality;
  const artifact = detail.status === "inspected";
  const sourceTopology = q?.mode === "publisher_reference_topology";
  const headerAudit = detail.verified_checkpoint_headers;
  const nameAudit = detail.verified_checkpoint_names;
  const inspection = detail.inspection;
  const architectureChecks = q?.architecture_checks;
  const checkLabels: Record<string, string> = {
    module_inventory: "Native module inventory",
    parameter_mapping: "Checkpoint parameter mapping",
    native_architecture: "Native architecture compatibility",
    module_topology: "Bounded module-call dependencies",
  };
  const hasCompleteArchitectureChecks = Object.keys(checkLabels).every(
    (key) => architectureChecks?.[key] === true,
  );
  const hasIncompleteArchitectureChecks = Object.keys(checkLabels).some(
    (key) => architectureChecks?.[key] === false,
  );
  const checks: [string, string | number][] = [
    ...Object.entries(checkLabels).map(([key, label]): [string, string] => [
      label,
      architectureChecks?.[key] == null
        ? "Not reported"
        : architectureChecks[key]
          ? "Complete"
          : "Incomplete",
    ]),
    [
      "Checkpoint parameter coverage",
      coverage(q?.checkpoint_parameter_coverage),
    ],
    ["Unknown graph components", q?.unknown_nodes ?? "Unknown"],
    [
      "Unmapped checkpoint tensors",
      q?.unmapped_checkpoint_tensors ?? "Unknown",
    ],
    ["Forward executed", bool(q?.forward_executed)],
    ["Runtime trace available", bool(q?.runtime_trace_available)],
    ["Mode", q?.mode ?? "Unknown"],
  ];

  return (
    <div
      className={`content-page report-page summary-page${artifact ? " is-artifact-only" : ""}`}
    >
      <header className="summary-heading">
        <div>
          <span className="eyebrow">TRUST & PROVENANCE</span>
          <h1>Inspection report</h1>
          <p>
            {detail.model_id} · revision {detail.revision ?? "unknown"}
          </p>
        </div>
        <Status status={detail.status} />
      </header>

      <div
        className="report-audit-banner"
        aria-label="Inspection audit highlights"
      >
        <div className="audit-score-card">
          <div className="audit-icon-wrapper" aria-hidden="true">
            <ShieldCheck size={24} />
          </div>
          <div>
            <span className="eyebrow">AUDIT STATUS</span>
            <h3>
              {artifact
                ? "Artifact Inspection"
                : detail.status === "completed"
                  ? hasCompleteArchitectureChecks
                    ? "Complete Architecture"
                    : hasIncompleteArchitectureChecks
                      ? "Inspection completed with gaps"
                      : "Inspection completed"
                  : detail.status === "partial"
                    ? "Partial Reconstruction"
                    : "Inspection not completed"}
            </h3>
            <p>
              {artifact
                ? "Pinned repository artifacts were inspected. No native architecture graph was reconstructed."
                : sourceTopology
                  ? headerAudit
                    ? nameAudit
                      ? "This partial graph comes from reviewed publisher source. Separate pinned audits verified shard headers and checkpoint-name correspondence; shapes, parameter totals, and runtime flow remain unverified."
                      : "This partial graph comes from reviewed publisher source. A separate pinned audit verified shard headers; tensor ownership and runtime flow remain unverified."
                    : "This partial graph comes from reviewed publisher source. Checkpoint shard headers and tensor ownership are unverified."
                  : hasCompleteArchitectureChecks
                    ? "Architecture facts come from inspected checkpoint metadata and native PyTorch module definitions."
                    : hasIncompleteArchitectureChecks
                      ? "One or more required architecture checks are incomplete. Review the recorded results and limits below."
                      : "The inspection was marked completed, but required architecture-check results are not reported, so complete architecture coverage cannot be confirmed."}
            </p>
          </div>
        </div>
        <div className="audit-stats-grid">
          <div className="audit-mini-stat">
            <span className="stat-label">Checkpoint parameter coverage</span>
            <strong className="stat-value">
              {q?.checkpoint_parameter_coverage != null
                ? `${(q.checkpoint_parameter_coverage * 100).toFixed(1)}%`
                : "Not measured"}
            </strong>
          </div>
          <div className="audit-mini-stat">
            <span className="stat-label">Engine Version</span>
            <strong className="stat-value font-mono">
              {detail.engine_version ?? "Not reported"}
            </strong>
          </div>
          <div className="audit-mini-stat">
            <span className="stat-label">Execution Mode</span>
            <strong className="stat-value font-mono">
              {q?.mode ?? "Unknown"}
            </strong>
          </div>
        </div>
      </div>

      {headerAudit && (
        <section
          className="report-section"
          aria-label="Separate checkpoint header audit"
        >
          <h2>Verified checkpoint headers</h2>
          <p>
            A separate audit of this exact revision matched{" "}
            {count(headerAudit.tensor_count)} tensor names across{" "}
            {count(headerAudit.shard_count)} shard headers to the pinned index.
            It read {count(headerAudit.header_bytes_read_by_audit)} header bytes
            and zero weight payload bytes. The{" "}
            {count(headerAudit.tensor_element_count_including_scales)} tensor
            elements include scales; they are not a model parameter count.
          </p>
          <p>
            Index SHA-256: <code>{headerAudit.index_sha256}</code>
          </p>
        </section>
      )}

      {nameAudit && (
        <section
          className="report-section"
          aria-label="Publisher checkpoint-name audit"
        >
          <h2>Checkpoint names matched to reference components</h2>
          <p>
            A separate source-reviewed audit matched all{" "}
            {count(nameAudit.state_key_count)} checkpoint state-key names to{" "}
            {count(nameAudit.component_count)} closest reference components at
            this exact index. There were zero unmatched, renamed, skipped, or
            colliding names. Individual tensor shapes, parameter totals, and
            converter output values remain unverified.
          </p>
          <p>
            Name audit SHA-256:{" "}
            <code>{nameAudit.name_audit_document_sha256}</code>
          </p>
        </section>
      )}

      {artifact ? (
        <section
          className="report-section report-outcome"
          data-testid="report-checks"
        >
          <div className="report-outcome-heading">
            <FileSearch size={19} aria-hidden="true" />
            <div>
              <span className="eyebrow">ARTIFACTS INSPECTED</span>
              <h2>Verified facts and boundaries</h2>
            </div>
          </div>
          <p>
            {inspection?.message ??
              "Architecture reconstruction was unavailable for this inspection."}
          </p>
          {inspection?.reason_code && (
            <code className="summary-reason-code">
              {inspection.reason_code}
            </code>
          )}
          <dl className="report-checks report-artifact-facts">
            <div>
              <dt>Repository files in pinned revision</dt>
              <dd>{count(inspection?.file_count)}</dd>
            </div>
            <div>
              <dt>Tensor headers inspected</dt>
              <dd>{count(inspection?.tensor_count)}</dd>
            </div>
            <div>
              <dt>Checkpoint artifacts present</dt>
              <dd>{inspection?.checkpoint_available ? "Yes" : "No"}</dd>
            </div>
            {inspection?.declared_checkpoint_index && (
              <>
                <div>
                  <dt>Index-declared tensor names</dt>
                  <dd>
                    {count(
                      inspection.declared_checkpoint_index
                        .declared_tensor_names,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Index-declared shard files</dt>
                  <dd>
                    {count(
                      inspection.declared_checkpoint_index.declared_shards,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Tensor inspection limit</dt>
                  <dd>
                    {count(
                      inspection.declared_checkpoint_index.tensor_record_limit,
                    )}
                  </dd>
                </div>
              </>
            )}
          </dl>
          {inspection?.declared_checkpoint_index && (
            <p className="report-boundary">
              Index declarations come from the pinned index JSON. Shard headers
              were not verified for this result, so these are not inspected
              tensor or parameter counts.
            </p>
          )}
          <p className="report-boundary">
            No architecture graph or semantic coverage is available for this
            result. Tensor element totals do not establish unique model
            parameters, module ownership, or dataflow.
          </p>
        </section>
      ) : (
        <section className="report-section" data-testid="report-checks">
          <h2>Coverage and checks</h2>
          {q?.architecture_scope && <p>{q.architecture_scope}</p>}
          <dl className="report-checks">
            {checks.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      <div className="report-columns">
        <section className="report-section" data-testid="analysis-scope">
          <span className="eyebrow">METHOD</span>
          <h2>Analysis scope</h2>
          <p>
            {artifact
              ? "The result records a pinned repository inventory and any bounded checkpoint headers that were safe to inspect. It does not reconstruct an architecture."
              : sourceTopology
                ? "The pinned configuration and reviewed publisher source describe reference modules and their order. This result does not verify runtime flow or checkpoint ownership."
                : "Completeness covers native modules, checkpoint bindings, and bounded module-call dependencies. Intermediate math expressions, activation shapes, and numerical equivalence are outside this architecture check. Forward execution is disabled."}
          </p>
          {detail.limitations?.length ? (
            <ul>
              {detail.limitations.map((limitation, i) => (
                <li key={i}>{limitation}</li>
              ))}
            </ul>
          ) : (
            <p>No additional method limitations returned.</p>
          )}
        </section>
        <section className="report-section report-warning-section">
          <span className="eyebrow">ATTENTION</span>
          <h2>Warnings and incomplete checks</h2>
          {detail.warnings.length ? (
            <ul>
              {detail.warnings.map((warning, i) => (
                <li key={i}>{warning}</li>
              ))}
            </ul>
          ) : artifact ? (
            <>
              <p>
                No extraction warnings were recorded because architecture
                extraction did not run.
              </p>
              <ul>
                {(inspection?.architecture_blockers ?? []).map((blocker, i) => (
                  <li key={`blocker-${i}`}>{blocker}</li>
                ))}
                {!inspection?.architecture_blockers?.length && (
                  <li>
                    {inspection?.message ??
                      "Architecture reconstruction unavailable."}
                  </li>
                )}
              </ul>
            </>
          ) : (
            <p className="clean-status-notice">
              <CheckCircle2 size={15} aria-hidden="true" />
              <span>No warnings returned.</span>
            </p>
          )}
          {detail.partial_reasons?.length ? (
            <ul>
              {detail.partial_reasons.map((reason, i) => (
                <li key={`reason-${i}`}>{reason}</li>
              ))}
            </ul>
          ) : detail.status === "partial" ? (
            <p>
              This report is partial; no specific incomplete-check details were
              returned.
            </p>
          ) : artifact ? null : (
            <p className="clean-status-notice">
              <CheckCircle2 size={15} aria-hidden="true" />
              <span>No incomplete-check details returned.</span>
            </p>
          )}
        </section>
      </div>

      <details className="report-section report-identity-details">
        <summary>Artifact identity and report metadata</summary>
        <dl className="report-checks">
          <div>
            <dt>Model ID</dt>
            <dd>
              <code>{detail.model_id}</code>
            </dd>
          </div>
          <div>
            <dt>Model revision</dt>
            <dd>
              <code data-testid="model-revision">
                {detail.revision ?? "Unknown"}
              </code>
              {detail.revision && (
                <CopyButton value={detail.revision} label="Copy revision" />
              )}
            </dd>
          </div>
          <div>
            <dt>Engine version</dt>
            <dd>
              <code>{detail.engine_version ?? "Unknown"}</code>
            </dd>
          </div>
          <div>
            <dt>Expires</dt>
            <dd>
              {detail.expires_at
                ? new Date(detail.expires_at).toLocaleString()
                : "Unknown"}
            </dd>
          </div>
        </dl>
      </details>
    </div>
  );
}
