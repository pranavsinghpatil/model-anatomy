# Model coverage repair evidence

Status: `.2` measured audit and final `.3` targeted recheck complete.

This record separates observed reconstruction gaps from accepted artifact-only
outcomes and upstream access failures. A native model can load successfully and
still remain partial. Coverage and warning counts below describe the recorded
static profile, not inference quality or runtime numerical equivalence.

## 250-model audit on engine 2026.10.20.2

The latest row per repository was selected from 500 capture records, yielding
250 pinned repositories: 16 completed, 87 partial, 142 artifacts inspected, and
5 failed attempts. The 87 partial reports overlap across these findings:

| Partial cohort | Models |
| --- | ---: |
| Unresolved source branches in verified full reports | 67 |
| Opaque calls in verified full reports | 64 |
| Semantic coverage below 1 | 54 |
| Module semantic coverage below 1 | 54 |
| Checkpoint parameter coverage below 1 | 4 |
| Tensor shape coverage below 1 | 49 |

The warning cohorts overlap each other and the coverage cohorts. Do not add the
counts to estimate unique models. These warnings preserve uncertainty in source
branches and unmodeled operations; they are not grounds for changing a partial
result to complete.

The 142 artifact-only results have one primary reason each:

| Primary reason | Models |
| --- | ---: |
| No supported Safetensors checkpoint | 49 |
| Native architecture unavailable | 49 |
| Checkpoint selection required | 20 |
| Pipeline component selection required | 12 |
| Configuration unavailable | 6 |
| GGUF header only | 4 |
| Metadata budget exceeded | 1 |
| Adapter requires its base model | 1 |

Artifact inspection reports an inventory and any verified tensor headers. It
does not report a graph, unique parameter count, execution order, or semantic
coverage. Checkpoint and pipeline-component ambiguity still requires explicit
selection.

The capture records four HTTP access failures and one artifact-processing
failure. Follow-up bounded Hugging Face probes established that the four HTTP
failures returned 401 `GatedRepo` responses for the pinned metadata/checkpoint
files. The XTTS-v2 config contains non-finite `Infinity` JSON constants, and the
requested Safetensors file returned 404 `EntryNotFound`. These are external
access/artifact blockers; they were not reclassified as successful inspection.

The audit summary is [model-coverage-audit-250.json](benchmarks/model-coverage-audit-250.json).
The bounded upstream checks are [model-coverage-upstream-failures.json](benchmarks/model-coverage-upstream-failures.json).
The summary retains only model IDs, pins, exception classes, counts, and safe
reason codes; it omits raw exception messages, URLs, and local paths.

## Historical reports and benchmark links

Reports from engine `.2` remain openable by explicit report ID while their
schema remains supported. The visitor-owned job records the report's original
engine version. Automatic submission/cache lookup remains exact to the current
engine and schema, so an `.2` extraction cannot satisfy an `.3` cache lookup or
appear to have been produced by `.3`.

The shared inspection catalog lists reports for the supported schema across
engine versions and exposes each original version. Benchmark rows link to an
exact model, immutable revision, artifact selection, and supported schema. When
several compatible stored reports exist, the current-engine report wins; among
reports from the same engine, the newest inspection wins. A historical report is
used only when no current-engine match exists. Stored reports still pass the
registry's content digest, report identity, schema, and engine-version checks
before they open as visitor-owned analyses.

## Implemented extraction and access repairs

Helper calls now bind to their inspected Python signatures, preserving keyword
names, primitive defaults, keyword-only arguments and variadic members. Static
and class methods receive the correct receiver binding. Unresolved argument
expansion and active chunking retain dependencies and explicit warnings. The
interpreter restores its caller state even when helper analysis raises.

The analyzer identifies actual Torch dropout, activation and softmax functions
by callable identity and records their operator contract and version provenance.
Shadowed names, malformed calls and dtype-changing softmax remain opaque. These
are static contracts; neither forward execution nor checkpoint payload loading
was enabled.

Optional standard `HF_TOKEN` credentials can authorize public gated repositories
when the token owner has access. Credentials are limited to canonical HTTPS
Hugging Face requests, removed on cross-origin redirects, and excluded from the
frontend environment. Private repositories and authenticated repositories without
an explicit public flag are rejected before caching. No approved token was
available for this run, so gated success is not claimed. Configure credentials
through the worker environment, never through browser input or committed files.

## Final targeted recheck

The frozen-source 22-case capture on engine `2026.10.20.3` produced 4 complete
static reports, 12 partial graphs, 1 artifact inspection, 4 HTTP access failures
and 1 artifact error. All 17 successful reports passed graph/IR round-trip,
service and storage contracts, with zero checkpoint payload bytes and no forward
execution. All previous outcome classifications were retained.

| Pinned model | Previous report warnings | `.3` report warnings |
| --- | ---: | ---: |
| bigcode/starcoder2-3b | 3 | 1 |
| sentence-transformers/all-MiniLM-L6-v2 | 4 | 3 |
| google-bert/bert-base-uncased | 5 | 4 |
| FacebookAI/roberta-base | 6 | 5 |
| openai/whisper-tiny | 9 | 4 |
| laion/clap-htsat-fused | 41 | 40 |
| answerdotai/ModernBERT-base | 9 | 7 |

These seven reports remain partial. T5/Flan-T5, Pythia, VideoMAE and SegFormer
retained their warning counts; XCLIP retained artifact-only status. SmolLM2,
Qwen2.5, GPT-2 and ViT retained complete static status. The full catalog retains
16 complete, 87 partial, 142 inspected, four HTTP failures and one artifact error.
The original baseline and earlier eight repair observations remain unchanged;
22 new observations bring the recorded recheck attempts to 30.

Exact pins, case identities, engine/runner source hashes and dependencies are in
[the final manifest](benchmarks/model-matrix-250-coverage-manifest.json) and
[validated portable summary](benchmarks/model-matrix-250-coverage-rechecks.json).
The committed catalog is rebuilt by validating all three runs in order.

The 17 new `.3` reports were uploaded to private Supabase Storage, verified by
digest and read-back, and registered with no failed imports. All 245 catalog
report links remain available. Publication
`5994e657b281069309ac01af48190ac1fa638a8ab09ae03b7b240572adf2356a`
retains all 250 rows and the three source runs. Seven capture/import/acceptance
artifacts, including the exact frozen backend/runner source bytes, were also
archived privately and verified by read-back.

Live API acceptance opened reports from 56 task categories with ownership
isolation, graph/node or artifact inspection as appropriate. It independently
opened historical `.2` and current `.3` reports, completed a fresh pinned `.3`
SmolLM2 reconstruction, verified the subsequent current-engine cache hit and six
export formats. Database, private storage and worker health passed on port 8000;
the preview runs on 5176. Optional AI remains explicitly unavailable. The safe
receipt is [model-coverage-live-acceptance.json](benchmarks/model-coverage-live-acceptance.json).

A benchmark row's engine identifies its measured observation. Its linked saved
inspection can be newer; the registry and opened report expose that inspection's
own engine. The acceptance check uses these separate identities.

Deterministic verification: `pytest -m "not integration" -q` passed **562** tests
with 1 skipped and 18 integration tests deselected. Ruff across backend/scripts/
infra tests, Python compilation, frontend TypeScript/build and `git diff --check`
passed. The changed catalog JSON passes Prettier. The broader inherited frontend
format gate still fails on 84 unchanged files; this backend increment does not
reformat those files. The serial desktop/mobile catalog and live-evidence
Playwright suites passed **32** tests, including all 250 rendered rows, measured
outcomes and pinned/stored inspection actions.

The source publication default now includes
`docs/benchmarks/model-matrix-250-coverage-rechecks.json` as the third private
run payload. The third payload validates against its captured run descriptor
before publication.

## Reproduction

Audit local capture records and digest-named reports with:

```powershell
python scripts/audit_model_coverage.py --records .local/supabase-alignment-runtime/capture.ndjson --report-dir .local/supabase-alignment-runtime/reports --output docs/benchmarks/model-coverage-audit-250.json
```

The publisher's `--store-only` default includes the baseline, existing repair
rechecks, and coverage-repair rechecks. It validates and imports all three run
payloads with the catalog publication; a missing or invalid run stops import.
