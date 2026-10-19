# Evidence-preserving reconstruction by model family

The 120-repository matrix in `benchmarks/model-matrix-100.json` is a compatibility
probe, not a universal-support certification. A repository/task label is discovery
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
