import { Link } from "react-router-dom";
import { ArrowRight, BookOpen, FileCheck2, Network } from "lucide-react";
import type { AnalysisDetail } from "../api/contract";
import { count, dimensionCount, reportDimensionKeys } from "../format";
import { CopyButton } from "./ui";
import { Status } from "./Common";
import { usePreferences } from "../state/preferences";
export function Overview({ detail }: { detail: AnalysisDetail }) {
  const { mode } = usePreferences();
  const artifact = detail.status === "inspected";
  return (
    <div className="content-page overview-page">
      <div className="page-heading">
        <span className="eyebrow">MODEL ORIENTATION</span>
        <h1>{detail.model_id.split("/").at(-1)}</h1>
        <p>
          {artifact
            ? "Verified artifacts are available. Architecture reconstruction is unavailable for this inspection."
            : "A model-specific reconstruction, grounded in its recorded artifacts."}
        </p>
      </div>
      <section className="overview-identity">
        <div>
          <span className="eyebrow">{detail.model_id.split("/")[0]}</span>
          <h2>{detail.architecture ?? "Architecture unknown"}</h2>
        </div>
        <Status status={detail.status} />
      </section>
      {!artifact && (
        <dl className="model-metrics">
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
      <div className="revision-block">
        <span>Model revision</span>
        <code data-testid="model-revision">{detail.revision ?? "Unknown"}</code>
        {detail.revision && (
          <CopyButton value={detail.revision} label="Copy revision" />
        )}
      </div>
      <section className="explore-next">
        <div className="section-title">
          <div>
            <BookOpen size={18} />
            <h2>
              {mode === "learn"
                ? "A path to understanding"
                : "Continue your inspection"}
            </h2>
          </div>
        </div>
        <div className="next-actions">
          <Link to={`?view=${artifact ? "artifacts" : "architecture"}`}>
            <Network size={22} />
            <div>
              <h3>
                {artifact
                  ? "Explore verified artifacts"
                  : "Explore the architecture"}
              </h3>
              <p>
                {artifact
                  ? "Read the file inventory and available tensor metadata."
                  : "Select a component to see its purpose, shapes and evidence."}
              </p>
            </div>
            <ArrowRight size={18} />
          </Link>
          <Link to="?view=report">
            <FileCheck2 size={22} />
            <div>
              <h3>Understand the report’s scope</h3>
              <p>
                Review coverage, unresolved checks and the inspection method.
              </p>
            </div>
            <ArrowRight size={18} />
          </Link>
        </div>
      </section>
      <div className="scope-note">
        <strong>Static inspection</strong>
        <p>
          Shapes and structure describe the recorded inspection. Coverage is not
          runtime verification; unknown facts remain explicit.
        </p>
      </div>
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
  const checks: [[string, string | number], ...[string, string | number][]] = [
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
    <div className="content-page report-page">
      <div className="page-heading">
        <span className="eyebrow">TRUST & PROVENANCE</span>
        <h1>Inspection report</h1>
        <p>
          What this inspection established, and where its boundaries remain.
        </p>
      </div>
      <section className="report-section" data-testid="report-checks">
        <h2>Report checks</h2>
        {artifact ? (
          <p>
            Architecture coverage is unavailable for an artifact-only
            inspection. Tensor element counts do not establish unique model
            parameter counts.
          </p>
        ) : (
          <dl className="report-checks">
            {checks.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        )}
        <h3>Incomplete checks</h3>
        {detail.partial_reasons?.length ? (
          <ul>
            {detail.partial_reasons.map((reason, i) => (
              <li key={i}>{reason}</li>
            ))}
          </ul>
        ) : (
          <p>
            {detail.status === "partial"
              ? "This report is partial; no specific incomplete-check details were returned."
              : "No incomplete-check details returned."}
          </p>
        )}
      </section>
      <section className="report-section">
        <h2>Warnings</h2>
        {detail.warnings.length ? (
          <ul>
            {detail.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        ) : (
          <p>No warnings returned.</p>
        )}
      </section>
      <section className="report-section" data-testid="analysis-scope">
        <h2>Analysis scope</h2>
        <p>
          Static inspection describes artifact-backed structure. Coverage does
          not establish runtime correctness or every possible execution path.
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
      <section className="report-section">
        <h2>Artifact identity</h2>
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
      </section>
    </div>
  );
}
