# Native Runtime Compatibility

## Reported Failure

The new frontend on port 5187 showed job `f62dd3641b5145ffb6ae55c5158eb3ad`
for `google/embeddinggemma-2` as artifact inspection. It used engine
`2026.10.12.6` and Transformers 4.57.6, which has no native `embedding_gemma2`
configuration mapping. The tensor inventory was real, but it was not architecture
reconstruction. Saved reports are immutable; submitting again with a new engine
creates a separate result rather than relabelling the old inventory.

## Implementation

- Pin official Transformers 5.19.0 and regenerate `uv.lock`.
- Preserve shared CLI/web reconstruction, native task routing, meta construction,
  remote-code denial, weight-loading guards and bounded metadata reads.
- Wrapped Q/K/V projections use their real source graphs instead of assuming
  direct `out_features` or skipping their clipping operations.
- Invalid explicit head counts, head dimensions and projection widths do not
  enter structural QKV refinement. Zero is invalid, not a missing value.
- Resolve the upstream combined tracing predicate by callable identity, retain
  its source hash and static-profile assumption, and never execute it.
- Keep unknown audio feature layouts unknown; only native Whisper establishes
  the existing B,mel,frames contract.
- Engine identity is `2026.10.13.1`; existing report/cache bytes are unchanged.
- Embedding-task/model names cannot classify whole attention/feed-forward
  backbones as embedding-only subtrees.
- Correlated JSON diagnostic events connect API request IDs to worker analysis
  IDs, revisions, stages and failures. DEBUG enables progress/timing; exception
  text, locals, request bodies and secrets are excluded. See `LOCAL_TESTING.md`.

## Pinned Model

`google/embeddinggemma-2` at
`914f7f89142e33e77833254d9c9b90c3cef7303b` constructs native
`EmbeddingGemma2Model` with text, vision and audio towers plus embedding
projections. The real checkpoint contains 1,376 tensor records; native unique
parameter count is 744,371,488. Checkpoint element count is a different measure
and must not replace the native parameter total.

Acceptance requires full checkpoint parameter mapping, preserved native towers,
unknown unverified image/video/audio layouts, valid graph/IR round-trip, no
forward pass and zero weight payload bytes. This is a **partial static architecture
report**, not a complete semantic understanding of every modality or input path.

The initial extraction retained 38 warnings: source-level functional calls,
conditions, modality transformations and dimensions need further work. Do not
remove warnings or claim complete coverage to make the result appear successful.
GGUF hybrid models, native Diffusers graphs, arbitrary custom code and large-model
capacity remain separate unresolved gates.

## Verification

`backend/tests/test_category_integration.py` includes the pinned multimodal case.
`test_static_predicates.py`, `test_attention_projection_boundaries.py`,
`test_modality_inputs.py` and `test_vit_semantics.py` cover the integration fixes.

An initial upgraded suite found an obsolete ViT layout assertion; focused checks
after adapting it to the actual native residual/projection structure passed 41
tests. An initial pinned run passed 10 cases but failed GPT-2's warning-free gate
on a new upstream tracing predicate. That failure motivated the audited identity
rule above.

Final locked environment: Python 3.11.4, CPU Torch 2.14.1, Transformers 5.19.0,
Hugging Face Hub 2.2.0 and Safetensors 0.8.0. `uv sync --frozen` installed the
API/engine/dev extras separately from the previously running shared virtualenv;
`uv lock --check --offline` passed.

- Backend/infra: **352 passed, 19 skipped**. Skips are opt-in network/platform
  checks; the real network/HTTP checks below run separately.
- Pinned model matrix and canonical engine checks: **11 passed**. Canonical
  GPT-2, SmolLM2, Qwen3 and ViT retain their strict coverage/warning requirements.
- Fresh HTTP -> durable worker -> real Hugging Face metadata: **3 passed**, for
  SmolLM2-360M, ViT and EmbeddingGemma2. Graph/nested graph, component/tensors,
  JSON/ZIP exports, exact-revision cache, visitor isolation, quotas and honest
  missing-repository failures are checked, with no mocked HTTP responses.
- Those HTTP checks verify request-ID/analysis-ID correlation and real failure
  events with stage, HTTPError type and frame locations. Six focused diagnostic
  tests check privacy, frame bounds, JSON escaping, levels and configuration.
- Ruff lint/format passed; the local launcher parses as valid PowerShell.
- Read-only review found the staged-head positional-input bug, which was fixed
  with regression cases. Subsequent scoped extraction/privacy reviews found no
  actionable findings. This is not a proof of universal or hosted correctness.

Earlier failed attempts remain relevant: the first live multimodal test assumed
flat config dimensions; another log assertion used `resolving` instead of the
actual persisted `resolving_revision`. Both test assumptions were corrected and
all three fresh live cases rerun successfully. A restricted full-suite runner
stalled and was stopped; the local-network-enabled runs above completed.

## Activated Local Backend

With user approval, the existing idle API and worker were replaced with the
verified isolated environment. The current SQLite database was backed up before
restart; no job, cache, report or visitor identity was reset. API port 8001 and
the integrated frontend on `http://127.0.0.1:5173/app` remain in use. DEBUG events
are written beneath `.local/app-v1/native-runtime/`; production quotas were not
changed. Existing bounded manual-testing allowances remain active.

A fresh submission through the frontend's real `/api` proxy returned native
`EmbeddingGemma2Model`, engine `2026.10.13.1`, nine root nodes, a nonempty nested
graph, 1,376 tensors, 744,371,488 parameters and 38 explicit warnings. Its JSON
export verified the pinned revision, no forward pass and zero weight payload
reads. Worker diagnostics recorded a roughly 36-second cold extraction and
41-second child lifecycle on this busy local machine; these are not load or
memory benchmarks. `/health` reports API/database/storage/worker ready and AI
unavailable without cloud credentials.

Browser-control calls timed out after restart, so a fresh visual/interactive UI
check is **not claimed**. Earlier browser inspection confirmed the new workspace
on 5173, and fresh HTTP validation exercised its API proxy, but neither replaces
desktop/mobile visual acceptance. The separate UI chat continues to own frontend
edits. Hosted resource enforcement, capacity and universal semantics remain open.
