# Artifact Inspection Acceptance

## October 12, 2026

Engine `2026.10.12.5` adds a distinct `inspected` outcome. This is not a
completed architecture, a mock graph, or a renamed partial report.

| Submitted repository | Pinned revision | Verified outcome |
| --- | --- | --- |
| `Aleph-Alpha/Kolibri-1` | `35bc4d3be745502227a67247de77d70e691614ee` | File inventory and source configuration; no graph or tensor claims. The index lists 116,303 tensor records across 32 shards, above the 20,000-record inspection budget. The installed Transformers version also lacks the native architecture. |
| `akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA` | `0d91c4dfa22b3f5aa89c020eadc30d8f42cb51c0` | 344 repository files and 384 verified tensor records from the sole root checkpoint, `multiple_angles_step1000.safetensors`. Nested training checkpoints are listed, not silently combined. This adapter is not an inspected base-model architecture. |

Both paths read zero weight payload bytes and execute no forward pass or remote
model code. Reports expose neither a unique model parameter count nor graph
coverage. The internal serialization has an empty graph placeholder; public
graph/node requests, Mermaid exports and architecture questions return 409.
Non-diagram exports and paginated verified records retain visitor ownership.

## Reproduce

Install the locked environment with the API, engine and development extras.
Run from the repository root:

```powershell
$env:RUN_HF_INTEGRATION = '1'
uv run --no-sync pytest backend/tests/test_engine_integration.py backend/tests/test_artifact_inspection_integration.py -q
```

For manual application testing, stop the identified existing API, worker and frontend,
then run `powershell -ExecutionPolicy Bypass -File scripts/start-local.ps1`.
The launcher preserves `.local/app-v1`, requires API and engine dependencies,
binds to loopback, and applies 100 visitor analyses/hour and 500 shared analyses/day.
Normal deployment defaults remain 2/hour and 20/day. No quota rows are erased.

Open `http://127.0.0.1:5173`, submit either model ID again, and expect
**Artifacts inspected / Architecture unavailable**. Old failed jobs remain
unchanged. Check source links, tensor pagination for the adapter, and JSON/ZIP
downloads. Submit SmolLM2-135M to verify the existing architecture path separately.

Real desktop/mobile browser acceptance, from `frontend` with the app running:

```powershell
$env:E2E_BASE_URL = 'http://127.0.0.1:5173'
$env:E2E_LIVE_API = '1'
npm run test:e2e -- e2e/live-artifacts.spec.ts e2e/live.spec.ts e2e/live-vit.spec.ts --workers=2
```

Deterministic tests cover missing explicitly requested files, malformed headers,
invalid native dimensions, bounded checkpoint selection, escaped Markdown file
names, cache reuse, ownership and prohibited graph/question access. Test fixtures
are test-only; production results always come from submitted repositories.

## Remaining Boundaries

Verification for this increment: 250 backend/infrastructure tests passed, with
nine opt-in/platform skips and one existing Starlette TestClient deprecation
warning. All six pinned real-repository engine checks passed separately, and all
66 deterministic desktop/mobile browser checks passed. All ten real desktop/mobile
browser checks passed separately, including ViT, SmolLM2 and the two reported repositories.
The production frontend
build, generated OpenAPI types, Python compilation, Ruff and frontend formatting
checks also passed. Separate real browser runs exercise both artifact repositories;
these checks are not replaced by the deterministic fixtures.

This does not establish universal architecture support. Diffusion, multimodal,
audio, video and very large/custom model architectures still need family-specific
implementation and acceptance. Ambiguous checkpoints may return files only;
malformed data and transport failures still fail. Supporting Kolibri's full
inventory requires a separately designed bounded streaming/storage path, not
simply raising record and memory limits. Hosted Supabase/Nebius/NVIDIA verification
still requires cloud setup and credentials.
