# Evidence-preserving reconstruction by model family

The 120-repository baseline in `benchmarks/model-matrix-100.json` and the
250-repository expansion in `benchmarks/model-matrix-250.json` are compatibility
probes, not universal-support certifications. A repository/task label is discovery
metadata. It does not establish a model's computation or training history.

## Shared acceptance gates

Every family starts with an immutable Hugging Face revision and actual file
inventory. Inspect bounded configuration and checkpoint headers before considering
architecture construction. Keep tensor payload reads at zero, forward execution
disabled, and publisher/custom Python unexecuted. Failed validation stays a failure;
missing native support may return an explicitly graph-free artifact inspection.

A trustworthy architecture additionally requires native component identities,
parameter ownership with shape checks, input/output contracts, source-backed
relationships, and explicit provenance. An IR round trip and valid public API
schemas establish structural validity, not scientific completeness. Coverage ratios
of one are necessary for the matrix's `completed` label, but still describe only
the engine's recorded static profile and supported analysis subset.

## Separate reconstruction routes

| Family | Current boundary | Next implementation and deterministic acceptance |
| --- | --- | --- |
| Native Transformers text and vision | Native Auto mappings, meta construction, checkpoint mapping, bounded source analysis. Partial semantics remain explicit. | Pin representative checkpoints per native family; validate heads, ties, buffers, input axes, shape ownership and graph relationships. Extend aliases/converters only from native source evidence. |
| Native classes absent from Auto mappings | A declared unmapped class cannot silently become a backbone. The matrix records this for CLIPSegForImageSegmentation; native SpeechT5ForSpeechToSpeech also needs a separate construction contract. | Introduce narrowly reviewed native class routes only after exact config/class compatibility, meta-only construction and negative routing tests. SpeechT5 construction must also satisfy the existing no-forward/no-real-tensor guard. Never import publisher-provided class names generically. |
| Optional native dependencies | A known class may require an unavailable optional package during construction. The measured Mask2Former case requires SciPy; several detection/backbone paths require separate review. | Define explicit optional dependency groups and pinned native fixtures before promoting support. Recheck that constructors neither fetch pretrained backbones nor allocate real tensors. Dependency installation alone does not prove graph semantics. |
| Multimodal understanding | Composite module ownership may be available while packed images, bridges and cross-modal shapes remain unknown. | Add per-family source-backed contracts for image axes/patch packing, text/image bridges and cross-attention. Test modality edges separately from containment and tensor inventory. |
| ASR, TTS, codecs and speech-to-speech | Generic signatures alone cannot establish waveform, mel, codec-token or speaker representations. | Separate waveform encoders, acoustic decoders, codecs, vocoders and speaker conditioning. Pin sample-rate/representation evidence where available; verify channel/time axes and each native component's checkpoint ownership. Streaming/duplex state needs its own contract. |
| Diffusers text-to-image | Pipeline manifests establish declared component inventory. Selected components can expose verified headers; a manifest does not prove a pipeline DAG. | Add a pinned native Diffusers dependency and support exact component classes on meta first. Derive pipeline links from reviewed native calls, including conditioning, scheduler and denoiser roles. Verify component configuration, shapes and ownership before presenting any pipeline graph. |
| Text/image-to-video | Image generation support does not establish temporal semantics. | Add exact temporal latent layouts, frame axes, temporal attention/convolution and conditioning contracts per native pipeline. Test temporal relationships and VAE boundaries against source/config evidence. Keep unverified links absent. |
| Decision Transformer and RL policies | Native sequence-policy modules differ from archived executable policies. RLHF is a training method, not an inference architecture. | Validate state/action/return/time dimensions and action heads for native decision models. Inspect policy ZIP metadata only with bounded non-executing readers; never deserialize pickle/SB3 checkpoints. Require an explicit supported native policy implementation before graph construction. |
| Reward models and RLHF/SFT LMs | Publisher task tags do not prove how weights were trained. | Reconstruct verified inference heads/backbones. Keep training declarations as attributed metadata; never derive RLHF provenance from tensor names. |
| PEFT adapters | Adapter tensors are deltas, not a full base model. | Pin the adapter and its declared base independently; validate target module compatibility, rank and shapes. Show ownership separately until an explicit, evidence-preserving composition route exists. |
| GGUF and other frameworks | A bounded GGUF header describes tensors and quantization, not a computation graph. Other formats may support inventory only. | Retain exact file/variant selection, split-file metadata and GGML axis order. Add native family reconstruction separately; do not infer execution edges or unique model parameters from header element totals. |

## Choosing the next increment

Group matrix outcomes by actual blocker: access/transport, invalid artifact,
resource/storage budget, native construction, checkpoint mapping, API serialization,
and incomplete semantics. Fix reproducible general correctness defects first.
Then promote one family at a time using tiny native fixtures plus pinned real
checkpoints. Rerun affected cases with a new engine identity and preserve the
baseline. A safe artifact-only outcome must never be counted as complete anatomy.

## Acceptance plans for specialist tasks in the 250-case matrix

These are implementation plans, not claims that task-specific anatomy already
exists. The catalog's measured outcome remains the authority for each pinned case.

| Task family | Specific route and acceptance gate |
| --- | --- |
| Time series (Chronos, TimesFM, Granite) | Separate forecast horizon, patch/time/channel axes, context windows and scaling from language-token semantics. Use exact registered native classes where available; preserve component and tensor ownership. Test each configuration's real input/output contract without treating successful T5 construction as complete forecasting semantics. |
| Protein sequence models (ESM, ProtBERT, Ankh) | Reconstruct the native sequence backbone, then attribute residue alphabet, special tokens, pooling and prediction heads to actual config/source. Sequence representations are distinct from protein NER and folding predictions. Add residue-axis and head-shape fixtures before promoting biological semantics. |
| Tabular models (TabPFN, EXAONE, Kumo) | Begin with file/config inventory and safe tensor headers where supported. Do not unpickle `.ckpt`, `.pt` or executable serialized estimators. A reviewed native adapter must establish row/column/query axes, feature encoding and prediction heads before graph support is claimed. |
| Robotics policies (LeRobot Pi0/Pi05) | Preserve vision, language, proprioceptive state, timestep and action components separately. Derive normalization and action dimensions from bounded artifact evidence. Review native constructors for downloads and allocations, then require component ownership and modality-edge fixtures. Policy success or safety is not established by static anatomy. |

Selection itself needs evidence: a publisher name containing “tabular” or a
biomedical NER model containing “protein” cannot establish the corresponding
specialist task. The discovery regression tests reject these false matches; the
curated manifest records replacements and exact-revision verification.

## Contracts confirmed by the 250-model repair rechecks

- **Hierarchical CLAP audio:** attention heads are a per-stage schedule, not a scalar.
  Validate the effective native defaults, depth sum and stage/head alignment before construction.
  The pinned repaired CLAP report is partial: shifted windows, padding, branch conditions,
  normalization and modality similarity operations still need source-backed semantic acceptance.
- **CLIP/XCLIP composites:** preserve fetched config JSON independently from Transformers'
  mutable native config objects, and require IR JSON round trips for architecture and
  artifact-only reports. Native modality ownership does not prove all similarity operations,
  normalization or temporal pooling semantics. A repository without a selected Safetensors
  checkpoint remains artifact-only even when native configuration loading succeeds.

The 250-model audit on engine `2026.10.20.2` found 87 partial graphs among 250
pinned repositories. Full-report warnings identified unresolved source branches
in 67 models and opaque calls in 64; those warning cohorts overlap. Semantic and
module-semantic coverage were below one in 54 partial reports each, checkpoint
parameter coverage in 4, and tensor-shape coverage in 49. Preserve each report's
partial status until source-backed behavior and tensor ownership are established.

The 142 artifact-only outcomes were primarily caused by unavailable
Safetensors (49), unavailable native architecture support (49), required
checkpoint selection (20), and required pipeline component selection (12).
Configuration absence (6), GGUF header-only inspection (4), a metadata budget
limit (1), and a standalone adapter (1) account for the remainder. These are
honest bounded outcomes; adding a family route requires native construction and
semantic acceptance, while selection ambiguity requires an explicit choice.

Four recorded HTTP failures were confirmed by bounded public probes as gated
401 responses. XTTS-v2 exposed non-finite JSON constants and no Safetensors file
at the pinned path. Keep those access and artifact outcomes separate from
unsupported native architecture and partial graph semantics.

Historical `.2` reports with the supported IR schema remain available through
explicit report selection and retain their `.2` provenance. Automatic analysis
cache reuse stays bound to the current engine and schema. Benchmark links match
the exact model, pin, artifact selection and schema, preferring a current-engine
report and then the newest compatible report. See
[`MODEL_COVERAGE_REPAIRS.md`](MODEL_COVERAGE_REPAIRS.md) for the audit and the
final 22-case `.3` recheck record. Seven reports have fewer warnings; all original
outcome classifications remain, with unresolved semantics still marked partial.

Pinned rechecks and unchanged baseline observations remain separate evidence. Future family
promotions must add deterministic semantic fixtures rather than suppressing warnings or
relabeling partial reports as complete.
