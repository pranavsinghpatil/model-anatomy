# EmbeddingGemma2 Graph Audit

## Observed Report

Analysis `9e4b8981d4df4febaad4e7e68e5757a0` reused the real pinned
EmbeddingGemma2 report from engine `2026.10.13.1`. Its cache-hit reconstruction
step took 891 ms; it did not redo extraction or resolve the existing warnings.

- Graph classification: 90.4%; native module classification: 75.4%.
- Checkpoint parameter mapping: 100%; graph connection shapes: 42.7%.
- 44 unclassified graph components across the nested hierarchy; 38 warnings.
- Root graph: nine components, six dataflow relationships.
- `audio_tower` and `embed_audio` have no recorded root relationship.

These are extraction gaps, not an excuse to rename Partial or hide unknowns.
Some graph problems are presentation defects, but layout cannot repair wrong or
missing source dependencies.

## Priority 1: Track Written Output Fields

In native Transformers 5.19.0 `modeling_embedding_gemma2.py`,
`get_image_features` assigns the projected/split representation to
`vision_outputs.pooler_output` before returning the output object (around lines
686-694). `get_audio_features` similarly writes the projected audio field
(around lines 877-880); video performs analogous field writes.

The current `StaticFlowAnalyzer.assign` handles names and tuple/list unpacking,
but not `ast.Attribute`. `attribute` also does not resolve fields written onto
returned objects. Therefore the image concatenation's source falls back to the
vision tower instead of its embedding projection. This is a correctness bug,
not merely an unknown dimension.

Acceptance gate: source-only fixtures must retain written named fields through
helper returns, aliases, and conditional branch merges, without executing model
code or leaking field writes between alternatives. A fresh pinned report must
prove the image/video path includes the projection before merging. Do not draw
replacement edges manually or relabel the old cached report.

## Priority 2: Declare Modality Input Profiles

Native forward requires both `input_features` and `input_features_mask` for the
audio branch (around line 821). The current static initializer creates a tensor
dependency for the former but defaults the unrecognized mask to None. Likewise,
video-specific arguments are not all represented as symbolic inputs. Audio/video
dataflow can therefore be absent from the canonical profile even though their
modules exist in raw containment.

Acceptance gate: explicit source-backed modality profiles or conditional paths
must preserve feature/mask/frame dependencies, optional-input guards, merge
operations, and the final language-model input. Required masks cannot silently
be omitted. Unknown shapes remain unknown. Do not turn structural inventory
into unconditional execution relationships.

## Priority 3: Explain the Graph Visually

The inspected browser view truncates long class names, uses identical class
titles for different embedding projections, repeats Unknown shape on edges,
and mixes isolated structural components with reconstructed dataflow. The UI
chat received these findings with explicit user authorization and owns that
work. Module ownership labels, readable framing and distinct structural grouping
must retain unknown states and source evidence. Verify desktop/mobile behavior;
do not claim visual acceptance solely from successful graph JSON responses.

## This Increment

The local read-only debug viewer exposes these facts and the actual event history.
The graph extraction bugs above are **not fixed by the viewer**. No engine version,
cache identity, graph facts, visitor ownership or API behavior changes in this
increment. Source-field/conditional-modality regression work is the next engine
gate before a stronger multimodal support claim.

Verification for the operator increment: backend/infra **363 passed, 19 skipped**,
including 11 new viewer cases for read-only inspection, blob integrity, privacy,
Unicode controls, partial log records, bounded reads and rotation/truncation.
Ruff lint/format and diff checks passed. The real saved analysis was inspected
through the new command, and a visible PowerShell terminal follows API/worker
events. The viewer import does not load Torch. Read-only review found rotation
and Unicode-control defects; both were fixed and regression-tested, with no
further actionable reader findings. Runtime graph extraction was not changed.
