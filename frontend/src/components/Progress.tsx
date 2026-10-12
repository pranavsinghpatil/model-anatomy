import { Link } from "react-router-dom";
import { LoaderCircle, ShieldCheck, Square } from "lucide-react";
import type { AnalysisDetail } from "../api/contract";
import { Button } from "./ui";
import { Status } from "./Common";
const stages: Record<string, [string, string]> = {
  queued: [
    "Waiting for the worker",
    "Your inspection is queued. It will begin when worker capacity is available.",
  ],
  resolving_revision: [
    "Pinning the model revision",
    "Resolving a specific repository revision so the evidence can be reproduced.",
  ],
  checking_cache: [
    "Checking inspected artifacts",
    "Looking for a previous inspection of this exact revision and engine version.",
  ],
  resolving: [
    "Resolving model artifacts",
    "Finding the repository configuration and checkpoint metadata.",
  ],
  reading_metadata: [
    "Reading artifact metadata",
    "Reading bounded headers and configuration without loading model weights.",
  ],
  building_structure: [
    "Reconstructing the structure",
    "Instantiating supported native structure and inspecting its modules.",
  ],
  analyzing_components: [
    "Mapping model components",
    "Associating semantic components with recorded modules and tensor facts.",
  ],
  validating: [
    "Validating the report",
    "Checking the reconstructed representation and recording unresolved facts.",
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
    "The worker is processing the inspection. Progress below comes from the server.",
  ];
  return (
    <div className="progress-page">
      <span className="progress-symbol">
        <LoaderCircle size={30} className="spin" />
      </span>
      <span className="eyebrow">INSPECTION IN PROGRESS</span>
      <h1>{stage[0]}</h1>
      <p>{stage[1]}</p>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Inspection in progress"
      />
      <div className="progress-stage" role="status">
        <Status status={detail.status} />
        <code>{detail.stage ?? detail.status}</code>
      </div>
      <Button disabled={cancelling} onClick={cancel}>
        <Square size={14} />
        {cancelling ? "Cancelling…" : "Cancel inspection"}
      </Button>
      <div className="progress-note">
        <ShieldCheck size={16} />
        <span>
          Only completed stages are facts. Duration varies by repository.
        </span>
      </div>
    </div>
  );
}
export function Recovery({ detail }: { detail: AnalysisDetail }) {
  const messages: Record<string, [string, string]> = {
    failed: [
      "Inspection failed",
      "The service could not complete this inspection. Review the recorded error before trying again.",
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
      "The worker was asked to stop this inspection. You can start another when ready.",
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
          <strong>{detail.error.code}</strong>
          <p>{detail.error.message}</p>
        </div>
      )}
      <Link className="button button-primary" to="/app">
        Start a new inspection
      </Link>
    </div>
  );
}
