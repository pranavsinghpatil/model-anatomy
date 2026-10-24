import type { GraphNode } from "../api/contract";

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
