import { Link } from "react-router-dom";
import {
  ArrowRight,
  BookOpen,
  FileCheck2,
  FileSearch,
  Network,
} from "lucide-react";
import type { AnalysisDetail } from "../api/contract";
import { count, dimensionCount, reportDimensionKeys } from "../format";
import { CopyButton } from "./ui";
import { Status } from "./Common";
import { usePreferences } from "../state/preferences";
import "./analysis-summary.css";

export function Overview({ detail }: { detail: AnalysisDetail }) {
  const { mode } = usePreferences();
  const artifact = detail.status === "inspected";
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

      <section className="explore-next summary-next">
        <div className="section-title">
          <div>
            <BookOpen size={18} aria-hidden="true" />
            <h2>
              {mode === "learn"
                ? "Where to go next"
                : "Continue this inspection"}
            </h2>
          </div>
          <p>
            {artifact
              ? "Start with the verified files, then inspect available tensor headers."
              : "Move from the model summary to its structure or recorded report."}
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
  const inspection = detail.inspection;
  const checks: [string, string | number][] = [
    ["Semantic coverage", coverage(q?.semantic_coverage)],
    ["Module semantic coverage", coverage(q?.module_semantic_coverage)],
    [
      "Checkpoint parameter coverage",
      coverage(q?.checkpoint_parameter_coverage),
    ],
    ["Tensor shape coverage", coverage(q?.tensor_shape_coverage)],
    ["Unknown nodes", q?.unknown_nodes ?? "Unknown"],
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
              <dt>Checkpoint selected</dt>
              <dd>{inspection?.checkpoint_available ? "Yes" : "No"}</dd>
            </div>
          </dl>
          <p className="report-boundary">
            No architecture graph or semantic coverage is available for this
            result. Tensor element totals do not establish unique model
            parameters, module ownership, or dataflow.
          </p>
        </section>
      ) : (
        <section className="report-section" data-testid="report-checks">
          <h2>Coverage and checks</h2>
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
              : "Static inspection describes artifact-backed structure. Coverage does not establish runtime correctness or every possible execution path."}
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
          ) : (
            <p>No warnings returned.</p>
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
          ) : (
            <p>No incomplete-check details returned.</p>
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
