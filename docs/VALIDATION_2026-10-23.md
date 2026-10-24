# Backend validation — 2026-10-23

## Scope and identity

Validated engine `2026.10.23.1`, schema `1.1`, at code commit
`438da7e3b6ccec9b144aa2e8f4c87c87c04625d6` on
`codex/value-dtype-evidence` ([PR #47](https://github.com/pranavsinghpatil/model-anatomy/pull/47)).
The isolated tree was extracted from `git archive HEAD`, without the existing
Cursor drafts. The combined tree includes those drafts. No application source
or test expectation was changed during this validation.

Environment: Windows, Python 3.11.4, CPU PyTorch 2.14.1, Transformers 4.57.6,
Safetensors 0.8.0, pytest 9.1.1, FastAPI 0.142.4, httpx 0.28.1, Ruff 0.16.10.
Both trees used the same installed dependencies. This does not test another
dependency version, GPU execution or a production Linux worker.

## Fresh results

| Check | Committed snapshot | Combined working tree |
| --- | --- | --- |
| Default Python suite | 404 passed, 17 skipped; 38.21 s | 408 passed, 17 skipped; 46.09 s |
| Pinned engine/category acceptance | 10 passed; 42.96 s | 10 passed; 44.61 s |
| Pinned artifact/GGUF acceptance | 4 passed; 74.06 s | 4 passed; 82.63 s |
| Real HTTP API + worker | 2 passed; 39.32 s | 2 passed; 55.40 s |
| Ruff lint, scoped format, Python compilation | Passed | Passed |

The default suite skips 16 opt-in network cases and the Linux-only worker-memory
gate. The default runs each emit one existing Starlette/httpx deprecation
warning. All 16 opt-in network cases passed separately on each tree; only the
Linux worker-memory gate remains unrun. Both working-tree and PR-range
whitespace checks passed. There were no failed tests in this validation.

The combined frontend passed `npm run build`, including TypeScript checking.
Vite reports a minified chunk above 500 kB. Browser interaction and live browser
acceptance were not run in this validation.

## Real model and service evidence

The strict engine cases are pinned GPT-2, SmolLM2-135M, Qwen3-4B and ViT. They
require zero extraction warnings, full scoped semantic/shape/checkpoint coverage,
IR round trips, zero weight-payload reads and no executed model forward method.
Six category cases cover T5, Swin, Whisper, VideoMAE, SigLIP and diffusion artifact
selection. Their acceptance permits documented partial or artifact-only results.

Four additional pinned tests verify Kolibri's explicit metadata-budget boundary,
an image LoRA's base-model requirement, and header-only inspection from two GGUF
publishers. They require zero payload reads and no fabricated architecture graph.

The live tests start separate API and durable worker processes on free loopback
ports, with new temporary databases/object stores. They scrub inherited
`MODEL_ANATOMY_*`, `NEBIUS_*` and `DATABASE_URL` settings. They do not replace the
existing local app or use prepared data.

| Live model | Pinned revision | Parameters | Checkpoint tensors | Observed status |
| --- | --- | ---: | ---: | --- |
| HuggingFaceTB/SmolLM2-360M | f8027fd0eaeea54caa13c31d31b9fdc459c38b49 | 361,821,120 | 290 | completed, zero warnings |
| google/vit-base-patch16-224 | 10064cc17b69d623a025c9e90e11230cc8ba6967 | 86,567,656 | 200 | completed, zero warnings |

Both trees verify real cold submissions and exact-revision cache reuse, root and
child graphs, component tensor ownership/evidence, upstream config dimensions,
JSON/ZIP exports, full checkpoint coverage and zero weight-payload reads. Separate
visitors cannot read another visitor's analysis; the submission quota returns
429. Missing repositories terminate as `failed` with `artifact_http`, without a
graph or parameter count, and their JSON export returns 503. Health reports
analysis available and AI unavailable without credentials.

Cold and cached `completed` statuses and empty warning lists were also read from
the test-owned databases after shutdown; the generic SmolLM HTTP assertion alone
would permit `partial`.

## Remaining correctness boundary

A fresh pinned T5 diagnostic on the committed snapshot still reports **26
warnings**: 41 graphs, 305 nodes, 132 checkpoint tensors, 60,506,624 unique
parameters, full checkpoint ownership, 15,536 header bytes and zero weight-payload
reads. Model forward was not executed. This confirms the existing partial result;
it does not close normalization/module-output dtype, cast guards, cache-type or
symbolic sequence-length proofs. See [DTYPE_EVIDENCE.md](DTYPE_EVIDENCE.md).

Next implementation gate: prove dtype transfer for one concrete native
normalization/module-output path using source evidence and native meta-tensor
oracles, then remeasure T5 and retain the canonical completed-model gates. Keep
opaque/custom outputs unknown. The Linux worker-memory gate, live browser
acceptance, hosted Nebius/Supabase deployment and complete semantics for other
categories remain unverified by this run.

## Reproduction and retained evidence

Run each suite sequentially, from the tree being validated, using the installed
virtual environment's absolute Python path for an isolated archive:

```powershell
D:/neb-hf/.venv/Scripts/python.exe -m pytest -q -o faulthandler_timeout=60
$env:RUN_HF_INTEGRATION = '1'
D:/neb-hf/.venv/Scripts/python.exe -m pytest -q backend/tests/test_engine_integration.py backend/tests/test_category_integration.py
D:/neb-hf/.venv/Scripts/python.exe -m pytest -q backend/tests/test_artifact_inspection_integration.py backend/tests/test_gguf_model_integration.py backend/tests/test_gguf_inspection.py::test_real_pinned_bonsai_inventory_and_explicit_ptq1_header
$env:RUN_HTTP_INTEGRATION = '1'
D:/neb-hf/.venv/Scripts/python.exe -m pytest -q -s backend/tests/test_live_http.py
D:/neb-hf/.venv/Scripts/python.exe -m ruff check backend scripts infra
D:/neb-hf/.venv/Scripts/python.exe -m ruff format --check backend/model_anatomy/src/static_flow.py backend/model_anatomy/src/static_dtypes.py backend/model_anatomy/src/version.py backend/tests/test_static_dtypes.py
D:/neb-hf/.venv/Scripts/python.exe -m compileall -q backend scripts infra
git diff --check
git diff f6a1ffc88803e3e199400cee714f460b5faf7260...438da7e3b6ccec9b144aa2e8f4c87c87c04625d6 --check
Push-Location D:/neb-hf/frontend
npm run build
Pop-Location
```

For the T5 diagnostic, set `PYTHONPATH` to the inspected tree's `backend` directory
so an editable install cannot select the combined tree. The archived source is
the exact code commit above:

```powershell
$env:PYTHONPATH = 'D:/neb-hf/.local/validation-438da7e-head/backend'
@'
import json
from model_anatomy.src.pipeline import reconstruct
ir = reconstruct('google-t5/t5-small', revision='df1b051c49625cf57a3d0d8d3863ed4d13564fe4', allow_artifact_inspection=True)
report = ir.metadata['analysis_report']
print(json.dumps({'warnings': len(report['warnings']), 'graphs': len(list(ir.root_graph.walk())), 'nodes': sum(len(g.nodes) for g in ir.root_graph.walk()), 'tensors': len(ir.tensors), 'parameters': ir.metadata['unique_parameter_count'], 'checkpoint_coverage': report['checkpoint_parameter_coverage'], 'forward_executed': report['forward_executed'], 'artifacts': ir.metadata['artifacts']}, indent=2))
'@ | D:/neb-hf/.venv/Scripts/python.exe -
Remove-Item Env:PYTHONPATH
```

Local ignored evidence lives in `.local/verification/`: `validation-438da7e-*.xml`
JUnit reports, `validation-438da7e-http-*-jobs.json` observed job summaries,
`http-438da7e-head/` and `http-438da7e-working/` API/worker logs, databases and
immutable reports, plus `t5-validation-438da7e-head*.json`. The archived source is
`.local/validation-438da7e-head/`. These are verification data, never application
fallbacks. Existing draft file hashes were recorded separately to verify that
validation preserved them.
