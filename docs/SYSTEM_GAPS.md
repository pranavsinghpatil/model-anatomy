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
| Google ViT image classification (~86.57M) | **Tested metadata/ownership path; partial semantics** | `google/vit-base-patch16-224` at `10064cc17b69d623a025c9e90e11230cc8ba6967` passes pinned engine and fresh real HTTP-worker checks: 86,567,656 parameters, 200 tensors, config-derived `[B,3,224,224]` image input, classifier tensor ownership, root dataflow, nested graph, export and cold/cache reuse. All checkpoint parameters map and zero payload bytes are read. Unknown native output modules and unresolved shapes remain; patch sequence/CLS-token transformations and image preprocessing are not fully established. Other vision architectures remain unverified. |
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

1. **Close modality correctness before claiming family breadth.** Add pinned,
   small real-model engine acceptance for (a) encoder-decoder text, (b) additional image
   encoder/classifier, (c) image-text generation, and (d) audio. Each should
   assert exact revision and config dimensions; expected forward argument and
   input-spec names/shapes; expected component ownership and graph relationships;
   checkpoint-to-module coverage; evidence provenance; successful IR round-trip
   and graph validation; and zero tensor-payload reads. Compare any advertised
   modality edge or shape against native source/config evidence. Expected failures
   should be explicit unsupported/partial status with a reason, never a plausible
   generic graph. First close ViT's unresolved semantic modules and prove patch
   sequence/CLS-token transformations. Do not mark a whole modality tested from
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
| Semantic engine | Real text and ViT metadata/ownership checks; explicit unknowns | ViT patch/output semantics first, then encoder-decoder and modality bridges. Add pinned acceptance for every advertised family. |
| Scale/resource control | Separate worker, timeout, bounded headers/shards/report, tested 4B text engine | Nested dimension validation, module/tensor/graph budgets, measured larger checkpoints, and actual Linux memory-limit evidence. |
| Jobs/API | Real text and image cold/cache jobs, isolation, limits, exports; worker lifecycle unit tests | Repeat crash/restart/concurrent submission/retention tests against hosted PostgreSQL, not only local SQLite. |
| Storage/cache | Local storage tested; private Supabase adapter/configuration exists | Real private-bucket integration, expired artifacts, provider outage and integrity recovery on the deployed service. |
| AI explanations | Evidence-bound provider integration and adversarial unit tests | Actual Nebius/NVIDIA credentials/model call, outage checks and evaluated citation correctness. Currently unavailable in the running app. |
| Frontend | Readable focused canvas, explicit overview, reversible panels, nesting/tensors/exports checked desktop/mobile | Modality-appropriate summaries, dense graph/long label accessibility and further visual polish using real reports. This is a first layout correction, not a finished redesign. |
| Deployment | Local live API/worker evidence; deployment configuration remains a separate tranche | Nebius access/credits, HTTPS, Linux containers, PostgreSQL/private storage and live inference before claiming production readiness. |

## This Pass's Verification

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
