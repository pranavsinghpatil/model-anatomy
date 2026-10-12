# System Gaps and Acceptance Gates

This is an implementation audit, not a support promise. Status describes evidence
in this repository as of 2026-10-12. "Tested" means there is a pinned real-model
engine or live HTTP acceptance path; a generic code path or fixture is not proof
that a model family works end to end.

## Current Evidence

The runtime now serves only submitted visitor-owned jobs. Prepared examples and
their endpoint have been removed. References to prepared checks in earlier
verification sections below describe historical test runs, not current features.

| Family / scale | Status | Evidence and boundary |
| --- | --- | --- |
| Decoder-only text, GPT-2 (~124M) and SmolLM2-135M | **Tested canonical static architecture** | October 12 pinned engine checks require all four coverage ratios of 1, no extraction warnings, exact revision/dimensions, full checkpoint parameter mapping, graph validation, no forward pass and zero weight payload reads. See [test_engine_integration.py](../backend/tests/test_engine_integration.py) and [ENGINE_ACCEPTANCE.md](ENGINE_ACCEPTANCE.md). Static shapes and classifications are not runtime observations or proof of every input path. |
| Decoder-only text, SmolLM2-360M | **Tested local application path** | Fresh live desktop/mobile browser tests require completed status, no extraction warnings, nested graph/tensor inspection and exports. The local HTTP acceptance also checks exact-revision cache, visitor isolation, limits and honest missing-repository failure. This is local Windows evidence, not cloud capacity evidence. See [test_live_http.py](../backend/tests/test_live_http.py) and [LIVE_BACKEND_ACCEPTANCE.md](LIVE_BACKEND_ACCEPTANCE.md); earlier partial results there are historical. |
| Qwen3-4B-Thinking-2507 (~4.02B) | **Tested canonical static engine** | Pinned revision `768f209d9ea81521153ed38c47d515654e938aea` passes the October 12 engine checks: 4,022,468,096 unique parameters, 36 layers, hidden size 2560, all four coverage ratios of 1, no extraction warnings, valid graph/IR round-trip and zero weight payload reads. Earlier local HTTP partial results use an older engine. This is not proof of larger text or other modalities. |
| Text models larger than the tested Qwen 4B checkpoint | **Unverified** | There is no parameter-count ceiling. Meta construction still creates the complete module hierarchy and IR; header-only weight access does not make those steps constant-memory. Service jobs are bounded by a 4 GiB memory setting and 120 seconds, and recognized nested scalar/per-stage layer fields reject more than 512 layers. See [registry.py](../backend/model_anatomy/src/registry.py), [settings.py](../backend/model_anatomy/api/settings.py), and [ENGINE_ACCEPTANCE.md](ENGINE_ACCEPTANCE.md). |
| Encoder-decoder text | **Pinned partial T5 report** | T5-small has 132 tensors and full native parameter mapping; unresolved mask/numerical-guard source paths remain. This is not full cross-attention/control-flow proof. See [DYNAMIC_ROUTING_ACCEPTANCE.md](DYNAMIC_ROUTING_ACCEPTANCE.md). |
| Google ViT image classification (~86.57M) | **Tested canonical static architecture; completed report** | `google/vit-base-patch16-224` at `10064cc17b69d623a025c9e90e11230cc8ba6967`: 86,567,656 parameters, 200 tensors, `[B,3,224,224]` pixels, 196 patches, CLS/position addition to `[B,197,768]`, encoder output projections/residuals, CLS selection `[B,768]` and classifier `[B,1000]`. Engine `2026.10.12.2` retains known Q/K/V named-output shapes; all scoped edges have shapes, all native modules are classified and all checkpoint parameters map. No extraction warnings or payload reads. Fresh HTTP-worker and real desktop/mobile tests require completed status. Shapes describe fixed-resolution, unmasked inputs without interpolation, not runtime observations. Preprocessing and other vision architectures remain unverified. |
| Additional image architectures | **Pinned partial Swin report** | 233 tensors, full native parameter mapping and staged heads; API serialization is fixed. Shifted windows, padding, shapes and classification still need semantic acceptance. |
| Image-text / multimodal | **Pinned partial SigLIP report** | 408 tensors and full native parameter mapping. Composite pixel layout and cross-modality semantics remain unknown; this does not prove image-text generation or arbitrary modality bridges. |
| Audio | **Pinned partial Whisper report** | 167 tensors and full native parameter mapping. Augmentation/control flow and some connection shapes remain unresolved. |
| Video | **Pinned partial VideoMAE report** | 186 tensors and full native parameter mapping. Native source/config establishes canonical `[B,16,3,224,224]`; pooling/position semantics and classifications remain incomplete. Other video layouts remain unverified and explicitly unknown. |
| Diffusion pipelines | **Manifest/component artifact inspection** | Stable Diffusion v1.5 is pinned and its selected UNet Safetensors headers are verified against sibling config. Native Diffusers graphs and pipeline execution edges are not implemented. |
| Custom / remote-code architecture | **Unsupported by policy** | `auto_map`, third-party classes and external backbone fetches remain disabled. Valid non-PyTorch Safetensors may produce graph-free artifact inspection; malformed headers still fail. No weights/custom code are loaded. |

"Small", "medium", and "large" are not currently product-defined parameter bands.
The tested checkpoints are a narrow sample, not evidence of general size scaling.
Family support and size support are separate
claims: a successful larger text model says nothing about image/video semantics,
and a small vision model would not establish large-model capacity.

## Artifact Inspection Is Not Architecture Support

The web worker can return a separate `inspected` result for repositories without
a supported native architecture, adapter-only repositories, missing Safetensors,
or checkpoint selection/metadata budget constraints. This preserves the pinned
file inventory and unambiguous validated tensor headers when available. It does
not provide a graph, unique model parameter total, module ownership or semantic
coverage. Malformed metadata and provider failures still fail honestly.

The reported Kolibri-1 revision has 116,303 indexed tensor records, above the
20,000-record inspection budget; its native architecture is also unavailable in
the installed Transformers version. Do not promise its full graph or increase
resource limits without separate validation. The Qwen Image multiple-angles
repository is a LoRA adapter, not its own complete diffusion pipeline.
See `test_artifact_inspection_integration.py` for pinned acceptance cases.

### Compatibility Expansion Order

Artifact format support and architecture semantics are separate gates. The
reported `prism-ml/Ternary-Bonsai-2-27B-gguf` repository contains multiple GGUF
language-model variants and separate vision projectors, not a canonical
Safetensors checkpoint. Do not combine these files or infer a hybrid-attention
graph from their tensor names. Explicit file selection and bounded GGUF metadata
inspection are the immediate compatibility increment.

After the format increment, pinned T5, Swin, SigLIP, Whisper and VideoMAE checks
now establish native construction, input boundaries and tensor mapping, with
explicitly partial graphs. Full semantic acceptance and image-text generation
remain open. Each family still needs
model-specific input/output semantics, component ownership, modality or
cross-attention relationships, tensor mapping and provenance checks. Merely
constructing a native module or returning file inventory is not a passed gate.
Video additionally needs explicit frame/temporal representation before a support
claim. These are remaining work, not capabilities delivered by the GGUF change.

## Actual Bounds

- Architecture reconstruction resolves to an immutable Hugging Face commit and requires `config.json`
  plus PyTorch-format Safetensors. Artifact inspection may show file inventory
  without either, but never infers a graph. It does not accept arbitrary model files or
  load checkpoint tensor payloads. The report explicitly records zero payload
  bytes read. See [artifacts.py](../backend/model_anatomy/src/artifacts.py) and
  [pipeline.py](../backend/model_anatomy/src/pipeline.py).
- Configuration JSON is limited to 1 MiB; index JSON to 16 MiB. A shard header
  is limited to 16 MiB, total Safetensors headers to 64 MiB, and checkpoints to
  64 shards. These bound metadata transfer, not weight size.
- Recognized scalar and per-stage layer counts are limited to 512, including
  nested dictionaries/lists. Configuration containers are limited to depth 64
  and 50,000 container records. Repository file records are limited to 8,192;
  checkpoint tensor records to 20,000 in total; native module, parameter and
  buffer records to 20,000 each. This limits records, not parameter elements.
  Native record checks run after meta construction, so they are not a complete
  pre-construction memory guarantee for arbitrary native families.
- Graph traversal is iterative and rejects cycles/shared graph objects. Limits
  are 1,024 graphs, depth 64, 20,000 total nodes and 50,000 total edges. Stored
  reports default to 32 MiB. These budgets do not establish larger-model support.
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

**Current closure gate:** close the existing documented product and system gaps
before adding extra features or model families. The October 12 hardening closes
GPT-2's padding guard and vocabulary applicability for vision reports. Hosted
operation still needs credentials and acceptance evidence. See [LOCAL_TESTING.md](LOCAL_TESTING.md)
and [READING_REPORTS.md](READING_REPORTS.md)
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
| Semantic engine | Four pinned canonical models now require complete static coverage with no warnings; native tracing predicate identities/source hashes are recorded | Broader Python control-flow proof and masked/interpolated variants need separate acceptance; unknown predicates remain warnings. |
| Scale/resource control | Separate worker, timeout, bounded headers/shards/report, nested layer checks and explicit native/tensor/graph record budgets | Measured larger checkpoints, additional native config forms and actual Linux/container memory evidence. |
| Jobs/API | Real text and image cold/cache jobs, isolation, limits, exports; worker lifecycle unit tests | Repeat crash/restart/concurrent submission/retention tests against hosted PostgreSQL, not only local SQLite. |
| Storage/cache | Local storage tested; private Supabase adapter/configuration exists | Real private-bucket integration, expired artifacts, provider outage and integrity recovery on the deployed service. |
| AI explanations | Evidence-bound provider integration and adversarial unit tests | Actual Nebius/NVIDIA credentials/model call, outage checks and evaluated citation correctness. Currently unavailable in the running app. |
| Frontend | Readable canvas, reversible panels, nested inspection/exports, input-fact-based vocabulary applicability checked desktop/mobile | Dense graph/long label accessibility and further visual polish using real reports; no universal modality claim. |
| Deployment | Local live API/worker evidence; deployment configuration remains a separate tranche | Nebius access/credits, HTTPS, Linux containers, PostgreSQL/private storage and live inference before claiming production readiness. |

## October 12 Hardening Verification

- Final engine `2026.10.12.2`: 222 backend/infrastructure tests passed with both
  real Hugging Face and HTTP-worker acceptance enabled. One Linux-only memory
  test skipped on Windows; one existing Starlette TestClient deprecation warning
  remains. Infrastructure helpers are a separate delivery increment.
- Four pinned engine cases require no extraction warnings, all four static
  coverage ratios of 1, exact dimensions/parameters and zero weight payload reads.
  The final rerun includes the Qwen named-mask condition and decorated methods.
- Source regression tests retain uncertain assignment/return dependencies, mark
  conditional continuation calls and exclude environments from terminated paths.
- Browser checks: 62 deterministic desktop/mobile checks, plus six real local
  browser checks using the final engine. Nested text/image graphs, tensor evidence,
  downloads, empty visitor state and the removed examples route are covered.
- Frontend build/types, formatting, generated OpenAPI types, Ruff, Python
  compilation and scoped whitespace checks pass. Native construction still
  precedes native-record limits; general Python control-flow completeness and
  universal modality support are not claimed.
- Cloud credentials, actual Supabase/NVIDIA use, public HTTPS and deployed
  capacity/recovery remain blocked on setup and real acceptance. The running
  local application needs no cloud account. See [LOCAL_TESTING.md](LOCAL_TESTING.md).

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

## Submitted-Analyses-Only Verification

- Removed `/api/v1/examples`, prepared snapshot loading/configuration, public
  sample-ID shortcuts, example response types/flags and the prepared sidebar.
  Frontend types were regenerated from the running FastAPI schema. Current API
  and browser runtime source has no prepared/example data path.
- New visitors start empty. Legacy example URLs return an explicit error, not a
  graph. All report, graph, node, export, cancel and question routes use persisted
  visitor-owned jobs. Real exact-revision caching remains labelled Cached.
- Full backend suite with both real integration flags: **157 passed**, one Linux
  memory-enforcement check skipped on Windows, one existing Starlette TestClient
  deprecation warning. Includes removal/ownership regressions and cold/cache
  HTTP-worker checks against real text and image artifacts.
- **52 desktop/mobile regression checks passed**, with six opt-in live cases
  disabled in that run. A separate **six-case live desktop/mobile run passed**:
  real SmolLM2-360M and pinned ViT submissions plus removed-example/empty-workspace
  checks. Live tests use no intercepted responses or substitute engine outputs.
- The first live run passed four cases, then Vite crashed watching a locked test
  download in the separate live-results directory. Excluding generated live
  artifacts from its watcher fixed the dev-server issue; the full live rerun
  passed. Build/type checking, Prettier, Ruff and diff checks pass.
- Historical CLI outputs and test-only fixtures/mocks remain regression evidence,
  not runtime sources. Existing uncommitted deployment drafts still need their
  retired snapshot-packaging/preflight assumptions removed before deployment;
  they are not used by the running application or included in this app-only PR.
  Hosted runtime/AI verification and the earlier extraction gaps remain open.

## ViT Semantic Pass Verification

- Full backend suite with both real integration flags: 128 passed, one Linux
  memory-enforcement test skipped on Windows, one existing Starlette TestClient
  deprecation warning. This includes explicit residual operand/skip wiring and
  standalone CLS/position checkpoint ownership assertions.
- 48 desktop/mobile browser checks passed: the existing 46 workspace/prepared
  checks plus two real pinned ViT submissions with no request interception.
- Frontend build/type checking and formatting, backend Ruff, and scoped Git diff
  checks pass. See [VIT_ACCEPTANCE.md](VIT_ACCEPTANCE.md) for manual testing.
