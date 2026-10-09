# System Gaps and Acceptance Gates

This is an implementation audit, not a support promise. Status describes evidence
in this repository as of 2026-10-09. "Tested" means there is a pinned real-model
engine or live HTTP acceptance path; a generic code path or fixture is not proof
that a model family works end to end.

## Current Evidence

| Family / scale | Status | Evidence and boundary |
| --- | --- | --- |
| Decoder-only text, GPT-2 (~124M) and SmolLM2-135M | **Tested** | Pinned real-model engine checks assert revision, parameter count, dimensions, full checkpoint parameter mapping, graph validity, no forward pass, and zero weight payload reads. See [test_engine_integration.py](../backend/tests/test_engine_integration.py) and [ENGINE_ACCEPTANCE.md](ENGINE_ACCEPTANCE.md). Reports may be partial: symbolic dataflow shapes and static classifications are not runtime observations. |
| Decoder-only text, SmolLM2-360M | **Tested** | A real non-prepared model traverses local HTTP, worker, graph/tensor inspection, JSON/ZIP export, exact-revision cache, and honest missing-repository failure. The report is partial and reads zero weight payload bytes. This is local Windows evidence, not cloud capacity evidence. See [test_live_http.py](../backend/tests/test_live_http.py) and [LIVE_BACKEND_ACCEPTANCE.md](LIVE_BACKEND_ACCEPTANCE.md). |
| Qwen3-4B-Thinking-2507 (~4.02B) | **Tested engine; locally observed HTTP result** | Pinned revision `768f209d9ea81521153ed38c47d515654e938aea` passes the committed real-engine checks: 4,022,468,096 unique parameters, 36 layers, hidden size 2560, complete checkpoint parameter mapping, valid graph/IR round-trip and zero weight payload reads. Its non-prepared local HTTP job also finished as partial. This is not proof of larger text or other modalities. |
| Text models larger than the tested Qwen 4B checkpoint | **Unverified** | There is no parameter-count ceiling. Meta construction still creates the complete module hierarchy and IR; header-only weight access does not make those steps constant-memory. Service jobs are bounded by a 4 GiB memory setting and 120 seconds, and recognized top-level native layer fields reject more than 512 layers. See [registry.py](../backend/model_anatomy/src/registry.py), [settings.py](../backend/model_anatomy/api/settings.py), and [ENGINE_ACCEPTANCE.md](ENGINE_ACCEPTANCE.md). |
| Encoder-decoder text | **Unverified** | Input and graph code has encoder/decoder sequence conventions, but there is no pinned end-to-end encoder-decoder acceptance model. Cross-attention and decoder wiring therefore have no family-level proof. |
| Google ViT image classification (~86.57M) | **Tested canonical static architecture; completed report** | `google/vit-base-patch16-224` at `10064cc17b69d623a025c9e90e11230cc8ba6967`: 86,567,656 parameters, 200 tensors, `[B,3,224,224]` pixels, 196 patches, CLS/position addition to `[B,197,768]`, encoder output projections/residuals, CLS selection `[B,768]` and classifier `[B,1000]`. Engine `2026.10.09.2` preserves known Q/K/V named-output shapes; all scoped edges now have shapes, all native modules are classified and all checkpoint parameters map. No extraction warnings or payload reads. Fresh HTTP-worker and real desktop/mobile tests require completed status. Shapes describe fixed-resolution, unmasked inputs without interpolation, not runtime observations. Preprocessing and other vision architectures remain unverified. |
| Image-text / multimodal | **Unverified** | No pinned multimodal model or modality-bridge acceptance test exists. Generic module classification and input names do not establish the relationship between image encoder, projector/resampler, text decoder, and modality-specific evidence. |
| Audio | **Unverified** | `input_features` and `input_values` receive generic symbolic specs, but no real audio model or audio-specific task/output semantics are tested. |
| Video | **Unsupported by the current explicit input description; otherwise unverified** | There is no video/frame/temporal input name or video-specific shape convention in `InputResolver`; the public `TensorSpec` can carry symbolic dimensions but does not identify a modality or temporal semantics. Native construction might succeed for some architectures, but that would not demonstrate a trustworthy video graph. |
| Custom / remote-code architecture | **Unsupported by policy** | `auto_map`, third-party config/model classes, external backbone fetches, and remote code are rejected. Unsupported Safetensors layouts or PyTorch-incompatible metadata also fail instead of falling back to loading weights. See [registry.py](../backend/model_anatomy/src/registry.py) and [artifacts.py](../backend/model_anatomy/src/artifacts.py). |

"Small", "medium", and "large" are not currently product-defined parameter bands.
The tested checkpoints are a narrow sample, not evidence of general size scaling.
Family support and size support are separate
claims: a successful larger text model says nothing about image/video semantics,
and a small vision model would not establish large-model capacity.

## Actual Bounds

- Model input resolves to an immutable Hugging Face commit and requires `config.json`
  plus PyTorch-format Safetensors. It does not accept arbitrary model files or
  load checkpoint tensor payloads. The report explicitly records zero payload
  bytes read. See [artifacts.py](../backend/model_anatomy/src/artifacts.py) and
  [pipeline.py](../backend/model_anatomy/src/pipeline.py).
- Configuration JSON is limited to 1 MiB; index JSON to 16 MiB. A shard header
  is limited to 16 MiB, total Safetensors headers to 64 MiB, and checkpoints to
  64 shards. These bound metadata transfer, not total tensor count, module count,
  or IR size.
- Recognized top-level layer fields over 512 are rejected; nested composite
  config dimensions do not receive the same validation. There is no
  explicit parameter-count, tensor-count, repository-file-count, or graph-node
  cap in the cited reconstruction path.
- Worker jobs have a 120-second deadline and 4 GiB memory setting. On Linux the
  child applies an address-space limit; deployment separately configures a 4 GiB
  container memory cap. Windows local runs enforce timeout but not that memory
  limit. Header-only extraction and meta-device construction avoid weight
  allocation and forward execution, but do not establish that arbitrarily large
  configs fit these bounds.
- Cloud readiness is not model-family or scale evidence. The recorded backend
  acceptance is local Windows with SQLite/local storage. Nebius CPU VM deployment,
  Linux container memory enforcement, PostgreSQL/Supabase operations, and live
  Nebius explanation inference remain deployment gates, not verified facts. The
  explanation provider is separate from reconstruction and cannot supply graph
  facts. See [LIVE_BACKEND_ACCEPTANCE.md](LIVE_BACKEND_ACCEPTANCE.md).
  Deployment configuration and benchmark work remain a separate delivery tranche.

## Prioritized Acceptance Gaps

**Current closure gate:** the user has paused expansion. Close existing extraction
warnings, report accuracy and summary applicability before starting any of the
family, scale or deployment additions below. See [READING_REPORTS.md](READING_REPORTS.md)
for report meanings and the manual verification sequence. The queue below records
future acceptance requirements; it does not authorize moving past this gate.

1. **Close modality correctness before claiming family breadth.** Add pinned,
   small real-model engine acceptance for (a) encoder-decoder text, (b) additional image
   encoder/classifier, (c) image-text generation, and (d) audio. Each should
   assert exact revision and config dimensions; expected forward argument and
   input-spec names/shapes; expected component ownership and graph relationships;
   checkpoint-to-module coverage; evidence provenance; successful IR round-trip
   and graph validation; and zero tensor-payload reads. Compare any advertised
   modality edge or shape against native source/config evidence. Expected failures
   should be explicit unsupported/partial status with a reason, never a plausible
   generic graph. The canonical ViT patch/CLS/output path is now established;
   masked/interpolated variants and other image families still need separate proof.
   Do not mark a whole modality tested from
   input specs or one checkpoint alone. Video models may also call their input
   `pixel_values`; the current four-dimensional convention must not be assumed
   correct for frame sequences or composite multimodal configurations.
2. **Define video behavior explicitly.** Decide which video artifact families
   are in scope and how frame, temporal, spatial, and any audio streams appear
   in the IR. Then add a pinned real video model acceptance case that checks those
   shapes and cross-component edges against config/source and Safetensors headers.
   Until this exists, retain the unsupported status for explicit video input
   semantics; native model construction alone is insufficient.
3. **Measure a size ladder separately from family coverage.** Define parameter
   bands and representative decoder-only text models, then run real pinned engine
   and HTTP-worker acceptance at increasing sizes, including the largest model
   intended for the CPU deployment. Record parameter/tensor/layer counts, shard
   and header bytes, extraction wall time and peak RSS, worker outcome, semantic
   and checkpoint coverage, and exact artifact revision. Include a case near each
   enforced boundary (512 layers, 64 shards, 16 MiB single header / 64 MiB total)
   to verify rejection or supported behavior. A band is supported only when the
   full worker path finishes within its declared resource budget with an
   inspectable report; do not infer from parameter count alone.
4. **Verify production runtime independently.** On the target Nebius Linux VM,
   run the same cold, non-prepared model submissions under the actual container
   memory/CPU limits, verify measured RSS and timeout behavior, and repeat with
   PostgreSQL/private Supabase storage configured. Keep real Nebius explanation
   inference as a separate optional-provider check. No cloud result should be
   inferred from Windows, SQLite, or mocked-provider checks.

## System-by-System Queue

| System | Current evidence | Next gate |
| --- | --- | --- |
| Artifact resolution | Pinned Safetensors and index/header safety checks, no payload fallback | Additional formats need their own safe metadata readers. Common revision-tree URLs are not accepted yet. Never enable pickle loading to improve coverage. |
| Semantic engine | Real text and canonical ViT patch/CLS/output semantics; explicit unknowns | First resolve GPT-2's native padding-diagnostic static warning with evidence. Family additions remain paused; masked/interpolated vision paths need separate acceptance. |
| Scale/resource control | Separate worker, timeout, bounded headers/shards/report, tested 4B text engine | Nested dimension validation, module/tensor/graph budgets, measured larger checkpoints, and actual Linux memory-limit evidence. |
| Jobs/API | Real text and image cold/cache jobs, isolation, limits, exports; worker lifecycle unit tests | Repeat crash/restart/concurrent submission/retention tests against hosted PostgreSQL, not only local SQLite. |
| Storage/cache | Local storage tested; private Supabase adapter/configuration exists | Real private-bucket integration, expired artifacts, provider outage and integrity recovery on the deployed service. |
| AI explanations | Evidence-bound provider integration and adversarial unit tests | Actual Nebius/NVIDIA credentials/model call, outage checks and evaluated citation correctness. Currently unavailable in the running app. |
| Frontend | Readable focused canvas, explicit overview, reversible panels, nesting/tensors/exports checked desktop/mobile | Modality-appropriate summaries, dense graph/long label accessibility and further visual polish using real reports. This is a first layout correction, not a finished redesign. |
| Deployment | Local live API/worker evidence; deployment configuration remains a separate tranche | Nebius access/credits, HTTPS, Linux containers, PostgreSQL/private storage and live inference before claiming production readiness. |

## Initial Gap Pass Verification

- Full backend suite with `RUN_HF_INTEGRATION=1` and `RUN_HTTP_INTEGRATION=1`:
  110 passed, one Linux memory-enforcement test skipped on Windows. One existing
  Starlette TestClient deprecation warning remains.
- `RUN_HTTP_INTEGRATION=1 pytest backend/tests/test_live_http.py -q -s`:
  two passed, each with fresh API/worker/data and no prepared reports or mocks.
  Text and image both verify cold/cache results, ownership, export, visitor
  isolation, rate limits and honest missing-repository failure.
- Frontend build/type checking, formatting, and 46 desktop/mobile checks cover
  readable component framing as well as explicit whole-graph framing. Test
  scenarios use fixtures only for deterministic failures; separate browser
  acceptance connects to the real local service.
- Focused review found a non-tab close control inside the panel tablist. It was
  moved outside the tablist and covered by a regression assertion.

The successful paths remain pinned native inspection with explicit static-analysis
limitations. Broader family, size and production claims must carry their own
acceptance evidence.

## Report Clarity Pass Verification

- Analysis status is separate from model revision and engine version. The API/UI
  expose recorded coverage, unknown/unmapped counts, execution flags and specific
  incomplete checks. Missing or invalid coverage/count values cannot pass.
- General method limitations are shown as analysis scope, not extraction warnings.
  Actual warnings remain visible and can make an otherwise fully covered report
  partial. Legacy summaries and terminal badges are reassessed from immutable
  stored bytes without modifying reports, ownership or persisted job records.
- Full backend suite with both real integration flags: **152 passed**, one
  Linux memory-enforcement check skipped on Windows, one existing Starlette
  TestClient deprecation warning. Frontend build/type checking, formatting, Ruff
  and scoped diff checks pass.
- **50 desktop/mobile regression checks passed**; six opt-in live cases were not
  enabled in that run. A separate live ViT run passed on both desktop and mobile,
  asserting the new quality fields, scope/warning separation, nested patch graph,
  classifier tensor ownership and ZIP export against the real API and worker.
- Current-engine audits of pinned SmolLM2-135M and Qwen3-4B-Thinking-2507 finish
  completed with four coverage ratios of 1, zero unknown/unmapped counts and no
  extraction warnings. GPT-2 retains the specific `is_torch_fx_proxy` opaque-call
  warning in its native padding diagnostic and remains partial. Earlier rows
  describing partial HTTP observations are historical, not current-engine audits.
- The user's existing ViT report at revision
  `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` was checked directly: completed,
  four coverage ratios of 1, no unknown/unmapped records or extraction warnings.
  Eight general scope notes remain visible. No model forward pass was executed.
- This closes report-clarity issues only. GPT-2's warning, modality-inapplicable
  summary fields, broader static-path evidence and hosted runtime gates remain
  open. No claim of a gap-free or universally supported application is made.

## ViT Semantic Pass Verification

- Full backend suite with both real integration flags: 128 passed, one Linux
  memory-enforcement test skipped on Windows, one existing Starlette TestClient
  deprecation warning. This includes explicit residual operand/skip wiring and
  standalone CLS/position checkpoint ownership assertions.
- 48 desktop/mobile browser checks passed: the existing 46 workspace/prepared
  checks plus two real pinned ViT submissions with no request interception.
- Frontend build/type checking and formatting, backend Ruff, and scoped Git diff
  checks pass. See [VIT_ACCEPTANCE.md](VIT_ACCEPTANCE.md) for manual testing.
