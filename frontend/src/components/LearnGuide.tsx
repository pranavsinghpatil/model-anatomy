import { useState } from "react";
import type { AnalysisDetail, GraphNode } from "../api/contract";

export type LearnStep =
  "orient" | "ports" | "component" | "tensors" | "evidence";

const componentDefinitions: Record<string, string> = {
  embedding: "Maps discrete token or category IDs to learned vectors.",
  position_embedding:
    "Adds learned position information to token representations.",
  rotary_position_embedding:
    "Applies rotary position information to query and key representations.",
  self_attention:
    "Relates positions within the same sequence to form context-aware representations.",
  attention: "Relates input representations to form context-aware outputs.",
  attention_core:
    "Computes attention from query, key, and value representations.",
  cross_attention: "Relates one sequence to a separate context sequence.",
  feed_forward:
    "Applies learned transformations to each position’s representation.",
  mlp: "Applies learned transformations to its input representation.",
  gated_mlp: "Uses a learned gate to modulate a feed-forward transformation.",
  normalization: "Rescales activations using a normalization rule.",
  layer_norm:
    "Normalizes activations across their feature dimension for each example and position.",
  rms_norm: "Normalizes activation magnitude using a root-mean-square scale.",
  batch_norm:
    "Normalizes activations using statistics across a batch or running estimates.",
  linear: "Applies a learned linear projection to the input features.",
  projection:
    "Projects features into a different learned representation space.",
  residual: "Combines a main-path value with a skip-path value.",
  residual_block:
    "Groups operations with a skip path that carries an earlier value forward.",
  transformer_block:
    "Groups the attention and feed-forward operations of a Transformer layer.",
  decoder_block:
    "Groups operations used by a decoder layer; exact composition depends on the model.",
  encoder_block:
    "Groups operations used by an encoder layer; exact composition depends on the model.",
  input: "Marks a value entering the inspected graph.",
  output: "Marks a value leaving the inspected graph.",
};

const guideSteps: { id: LearnStep; label: string; copy: string }[] = [
  {
    id: "orient",
    label: "Orient",
    copy: "The graph shows reported components and their connections. Select a node to inspect its details.",
  },
  {
    id: "ports",
    label: "Inputs and outputs",
    copy: "Shapes list dimensions in order. B means batch size, S sequence length, and T decoder sequence length. H and W indicate image height and width when shown together; other symbols are report-defined dimensions.",
  },
  {
    id: "component",
    label: "Component",
    copy: "The report supplies the component kind, module information, and associated metadata. The general definition is authored guidance; the label and metadata are the report’s facts.",
  },
  {
    id: "tensors",
    label: "Tensors",
    copy: "This view lists checkpoint tensor metadata attributed to the selected component, including its reported shape and dtype.",
  },
  {
    id: "evidence",
    label: "Evidence",
    copy: "Evidence records identify the source and locator used by the report. A locator points to the supporting artifact or record.",
  },
];

export function componentPurpose(node: GraphNode | null) {
  if (!node)
    return "Select a component to inspect its reported inputs, outputs, tensors, and evidence.";
  return componentMeaning(node.kind);
}

export function componentMeaning(kind: string) {
  return (
    componentDefinitions[kind.toLowerCase()] ??
    "Purpose unknown for this component kind."
  );
}

export function LearnGuide({
  analysis,
  node,
  onStep,
  artifactOnly,
}: {
  analysis: AnalysisDetail;
  node: GraphNode | null;
  onStep?: (step: LearnStep) => void;
  artifactOnly?: boolean;
}) {
  const isArtifactOnly = artifactOnly ?? analysis.status === "inspected";
  const [selectedStep, setSelectedStep] = useState<LearnStep>("orient");
  const selected =
    guideSteps.find((step) => step.id === selectedStep) ?? guideSteps[0];
  const chooseStep = (step: LearnStep) => {
    setSelectedStep(step);
    onStep?.(step);
  };

  return (
    <details className="learn-guide" aria-label="How to read this report">
      <summary>
        <span>Graph guide</span>
        <small>
          {isArtifactOnly
            ? "Repository metadata"
            : node
              ? `${node.label} · ${node.kind.replaceAll("_", " ")}`
              : "Select a node, edge, or connection to inspect reported facts"}
        </small>
      </summary>
      <div className="learn-guide-content">
        <header>
          <span className="eyebrow">Reading guide</span>
          <p>
            {isArtifactOnly
              ? `Repository and checkpoint metadata for ${analysis.model_id} at revision ${analysis.revision ?? "Unknown"}.`
              : node
                ? componentPurpose(node)
                : `Architecture report for ${analysis.model_id} at revision ${analysis.revision ?? "Unknown"}.`}
          </p>
        </header>
        {isArtifactOnly ? (
          <p className="learn-guide-copy">
            Artifact metadata comes from the repository inspection response.
          </p>
        ) : (
          <>
            <nav
              className="learn-guide-steps"
              aria-label="Reading guide topics"
            >
              {guideSteps.map((step) => (
                <button
                  key={step.id}
                  type="button"
                  aria-pressed={selected.id === step.id}
                  onClick={() => chooseStep(step.id)}
                >
                  {step.label}
                </button>
              ))}
            </nav>
            <p className="learn-guide-copy" aria-live="polite">
              {selected.copy}
            </p>
          </>
        )}
      </div>
    </details>
  );
}
