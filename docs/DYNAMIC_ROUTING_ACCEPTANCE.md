# Dynamic Model Routing

## Actual Failures
The submitted Bonsai job `a1d53bbc62394cf78235fdf78d7715d5` stopped at checkpoint
selection, not a crashed inspection. Its model variants and vision projectors
must not be combined or selected arbitrarily.

Live testing exposed a real Swin API failure: its staged heads `[3,6,12,24]`
failed the scalar-only response schema after successful extraction. The API and
both displays now preserve stages. Swin layers count 12 blocks, not four stages.
Detail/list regressions cover this 503 failure.

## Routing Contract
- Resolve an immutable revision and inspect actual repository files.
- Retain task/library/architecture/format declarations as hints, not graph facts.
- Select native wrappers through upstream AutoModel mapping names, without
  importing unrelated implementations. Missing architecture declarations may
  use native encoder-decoder/speech hints with a warning. Unavailable declared
  heads must never silently become another backbone.
- Pipeline manifests require component selection. Use sibling configs and
  directory-relative Safetensors indexes; validate paths and actual headers.
- Non-PyTorch Safetensors remain verified artifacts, not native PyTorch graphs.
- VideoMAE has source-established B,T,C,H,W axes and configured canonical frame
  count. Other unverified video/composite layouts stay unknown. CLIP uses its
  nested vision config. Native blocker reasons are retained in the profile.

No endpoint, remote-code execution, weight-loading fallback, prepared report or
LLM graph inference was added. Cache engine is `2026.10.12.8`; old reports remain
immutable. These changes do not establish universal support.

## Pinned Evidence
These cases return **partial static reports**, not complete modality support.
Tests check exact revision, native architecture, tensor count, input contract,
full checkpoint parameter mapping, graph/IR validation and zero payload reads.

| Model | Revision | Tensors | Remaining Gaps |
| --- | --- | ---: | --- |
| `google-t5/t5-small` | `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` | 132 | Mask and numerical-guard source paths. |
| `microsoft/swin-tiny-patch4-window7-224` | `d00d478bfaf1f417d34a1186673ad4b4be264c51` | 233 | Shifted windows, padding, classification and shapes. |
| `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | 167 | Augmentation, functional operations and some shapes. |
| `MCG-NJU/videomae-base-finetuned-kinetics` | `488eb9a0565f257b32866000305c8178965eb9f6` | 186 | Input `[B,16,3,224,224]` established; pooling/position semantics incomplete. |
| `google/siglip-base-patch16-224` | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | 408 | Image-text semantics and composite pixel layout. |

Diffusion acceptance pins `stable-diffusion-v1-5/stable-diffusion-v1-5` to
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. The manifest and selected
`unet/diffusion_pytorch_model.safetensors` headers are inspected using
`unet/config.json`. No native Diffusers pipeline graph is claimed.

Tests: `test_category_integration.py`, `test_native_routing.py`,
`test_repository_routing.py`, `test_modality_inputs.py` and
`frontend/e2e/live-categories.spec.ts`. Network/browser checks are opt-in; existing
canonical acceptance retains its complete scoped requirements.

## Manual Check
Open `http://127.0.0.1:5173/` and submit the IDs above. Inspect graphs, component
facts, tensor evidence, report checks, warnings and JSON downloads. Partial means
extraction gaps remain, not a different name for revision.

For Bonsai, select `Ternary-Bonsai-2-27B-PTQ1_0.gguf`: expect 851 tensor records,
not a fabricated graph. For diffusion, select the UNet checkpoint above and check
the sibling config and native blocker. Selections pin the exact inspected commit.

Arbitrary sizes, universal semantic completeness, more formats/custom code,
native Diffusers and hosted cloud operation remain open. Metadata stays bounded.
Next semantic changes must close the explicit matrix gaps without hiding warnings.

## Verification - October 12
- Final backend/infra suite: **314 passed, 17 skipped**, one existing
  Starlette/httpx deprecation warning. Skips cover opt-in network/platform checks.
- Final pinned network run: **10 passed** across this matrix and the existing
  canonical GPT-2, SmolLM2, Qwen3 and ViT acceptance checks.
- Real browser: **10 native-category checks passed** (five models on desktop and
  mobile), followed by **2 diffusion component checks passed**. The native browser
  run reused real cached reports from earlier cold API/worker inspections; the
  pinned network run independently performs fresh engine extraction. No HTTP
  responses were mocked in these live checks.
- Type checking, production build, Ruff lint/format and Prettier passed. The
  build retains a roughly 500 kB main-chunk warning; this is not a bundle-size
  acceptance claim. Read-only routing review found no actionable issues.
- Final frontend regression run recorded a passing result with no failed tests:
  72 deterministic checks and 24 opt-in live skips. The 12 new live category
  checks above were verified separately against the running API.
- Local `/health` reports database/storage/worker healthy. AI is unavailable;
  Nebius/Supabase deployment and resource/load acceptance remain unverified.

Failed attempts are not omitted: an earlier concurrent live run hit a real Swin
worker timeout and a T5 browser queue wait under observed low-memory contention;
remaining cases were cancelled. A later diffusion browser test used an incorrect
accessible button label and timed out before selection. The selector was fixed
and both viewport reruns passed. A restricted runner could not reach the existing
Vite server and exited on port conflict; the local-network-enabled rerun passed.
These reruns do not prove resource isolation or queue latency under contention.

Application data is real submitted inspection data. Deterministic browser tests
use test-only fixtures to check errors and interface states; they are not served
by the application. The UI redesign belongs to a separate chat; this increment
only changes compatibility rendering and staged dimensions.
