import { Link } from "react-router-dom";
import {
  LoaderCircle,
  Square,
  GitBranch,
  FileSearch,
  Network,
  FileCheck2,
} from "lucide-react";
import type { AnalysisDetail } from "../api/contract";
import { Button } from "./ui";
import { Status } from "./Common";
import "./progress.css";
import { retryReference } from "../state/submissions";
const stages: Record<string, [string, string]> = {
  queued: [
    "In the queue",
    "Your inspection will start when a worker is available.",
  ],
  resolving_revision: [
    "Pinning the model revision",
    "Pinning this repository revision so the findings can be reproduced.",
  ],
  checking_cache: [
    "Checking inspected artifacts",
    "Checking for an inspection of this same revision.",
  ],
  resolving: [
    "Resolving model artifacts",
    "Finding the configuration and checkpoint metadata.",
  ],
  reading_metadata: [
    "Reading artifact metadata",
    "Reading configuration and tensor headers without loading weights.",
  ],
  building_structure: [
    "Reconstructing the structure",
    "Building the supported native structure and reading its modules.",
  ],
  analyzing_components: [
    "Mapping model components",
    "Linking model components to recorded modules and tensor facts.",
  ],
  storing_metadata: ["Saving the report", "Saving the inspection report."],
  validating: [
    "Validating the report",
    "Checking the report and recording unresolved facts.",
  ],
};
export function Progress({
  detail,
  cancel,
  cancelling,
}: {
  detail: AnalysisDetail;
  cancel: () => void;
  cancelling: boolean;
}) {
  const stage = stages[detail.stage ?? ""] ?? [
    "Inspecting model",
    "Preparing the model metadata for inspection.",
  ];
  return (
    <div className="progress-page">
      <div className="progress-model">
        <Network size={16} />
        <span>{detail.model_id}</span>
      </div>
      <span className="progress-symbol">
        <LoaderCircle size={30} className="spin" />
      </span>
      <span className="eyebrow">INSPECTION IN PROGRESS</span>
      <h1>{stage[0]}</h1>
      <p>{stage[1]}</p>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Inspection progress"
        aria-valuetext={`${stage[0]}. ${stage[1]}`}
      />
      <div className="progress-stage">
        <Status status={detail.status} />
        <span role="status" aria-live="polite">
          {stage[0]}
        </span>
      </div>
      <details className="progress-technical-details">
        <summary>Technical details</summary>
        <code>{detail.stage ?? "Stage not reported"}</code>
      </details>
      <div className="inspection-steps" aria-label="Inspection workflow">
        {[
          {
            Icon: GitBranch,
            label: "Resolve revision",
            active: ["queued", "resolving_revision", "checking_cache"],
          },
          {
            Icon: FileSearch,
            label: "Read artifacts",
            active: ["resolving", "reading_metadata"],
          },
          {
            Icon: Network,
            label: "Map structure",
            active: ["building_structure", "analyzing_components"],
          },
          { Icon: FileCheck2, label: "Check findings", active: ["validating"] },
        ].map(({ Icon, label, active }) => (
          <div
            key={label}
            className={active.includes(detail.stage) ? "is-current" : undefined}
          >
            <Icon size={18} />
            <span>{label}</span>
            {active.includes(detail.stage) && <small>Current stage</small>}
          </div>
        ))}
      </div>
      <Button disabled={cancelling} onClick={cancel}>
        <Square size={14} />
        {cancelling ? "Cancelling…" : "Cancel inspection"}
      </Button>
      <p className="progress-leave-note">
        You can leave this page; the inspection continues in the background.{" "}
        <Link to="/">View your inspections</Link>
      </p>
    </div>
  );
}
export function Recovery({ detail }: { detail: AnalysisDetail }) {
  const messages: Record<string, [string, string]> = {
    failed: [
      "Inspection failed",
      "The service could not finish this inspection. Review the reason, then retry or choose another model.",
    ],
    unsupported: [
      "Architecture unsupported",
      "This model cannot be reconstructed by the current inspection method.",
    ],
    timed_out: [
      "Inspection timed out",
      "The inspection exceeded its allowed runtime. Try a smaller model or an explicit supported checkpoint.",
    ],
    cancelled: [
      "Inspection cancelled",
      "This inspection has stopped. You can start another when ready.",
    ],
  };
  const [title, copy] = messages[detail.status] ?? [
    "Architecture unavailable",
    "No architecture graph was returned for this inspection.",
  ];
  return (
    <div className="page-state">
      <h1>{title}</h1>
      <p>{copy}</p>
      {detail.error && (
        <div className="analysis-error" role="alert">
          <p>{detail.error.message}</p>
          <details>
            <summary>Technical details</summary>
            <code>{detail.error.code}</code>
          </details>
        </div>
      )}
      <Link
        className="button button-primary"
        to={`/?model=${encodeURIComponent(retryReference(detail))}`}
      >
        Edit model and retry
      </Link>
      <p className="recovery-inspections">
        <Link to="/">Your inspections</Link>
      </p>
    </div>
  );
}
