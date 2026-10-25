# 250-model compatibility benchmark

Status: all **250 distinct pinned repositories** were attempted on frozen engine `2026.10.20.1`. **8 targeted rechecks** on repaired engine `2026.10.20.2` retain the baseline history. These are measured static compatibility outcomes; 250 attempts do not mean 250 complete reconstructions.

This increment combines the database-backed catalog in PR #34 and the validated
engine in PR #32. The initial 120 case definitions are preserved; 130 distinct
repositories extend coverage to 57 intended task categories. Specialist selection
is curated: biomedical protein NER is not protein-sequence modeling, and an
author name containing "tabular" is not evidence of a tabular task. Exact pinned
repository metadata and optional artifact filenames were checked through the
public Hugging Face API. Publisher task labels remain declarations.

## Measured outcomes

| Outcome | Baseline | Latest observation |
| --- | ---: | ---: |
| Complete static report | 16 | 16 |
| Partial graph | 83 | 87 |
| Artifacts inspected | 141 | 142 |
| HF request failed | 4 | 4 |
| Artifact processing failed | 2 | 1 |
| Report validation failed | 4 | 0 |

The latest catalog preserves 250 repository rows and attributes every replacement to its own engine,
source hash, timestamp and prior status. CLAP accepts its valid hierarchical audio attention schedule,
including omitted native defaults, while rejecting malformed schedules. CLIP/XCLIP source
configuration now survives native config construction unchanged, preserving JSON round trips.
These repairs do not turn incomplete graph semantics into complete anatomy.

The remaining HF access failures and strict metadata failures remain visible. The pinned
`coqui/XTTS-v2` config contains a literal `Infinity` token and is not valid finite JSON.
Access failures do not establish that an upstream repository is invalid.

One Windows runner interruption occurred after 198 saved observations. The unchanged runner
resumed from matching case, source and environment identities; incomplete/unrecorded work was retried.
The full 250-case baseline was validated before publication. Earlier pre-pinned-URL observations
remain excluded.

Portable evidence:

- [Full baseline](benchmarks/model-matrix-250-baseline.json)
- [Repair recheck manifest](benchmarks/model-matrix-250-recheck-manifest.json)
- [Repair recheck observations](benchmarks/model-matrix-250-rechecks.json)
- [Live HTTP acceptance](benchmarks/model-matrix-250-live-http.json)
- [Published catalog](../frontend/public/benchmark-catalog.json)

## Reproduce

```powershell
python scripts/model_matrix.py --manifest benchmarks/model-matrix-250.json --output .local/model-matrix-250/results.ndjson --summary .local/model-matrix-250/summary.json --resume
python -m scripts.publish_benchmark_catalog --manifest benchmarks/model-matrix-250.json --summary .local/model-matrix-250/summary.json --output frontend/public/benchmark-catalog.json
```

Apply the retained targeted rechecks with their own recorded engine/source identity:

```powershell
python scripts/model_matrix.py --manifest docs/benchmarks/model-matrix-250-recheck-manifest.json --output .local/model-matrix-250/rechecks.ndjson --summary .local/model-matrix-250/rechecks.json --allow-small --resume
python -m scripts.publish_benchmark_catalog --manifest benchmarks/model-matrix-250.json --summary .local/model-matrix-250/summary.json --recheck-manifest docs/benchmarks/model-matrix-250-recheck-manifest.json --recheck-summary .local/model-matrix-250/rechecks.json --output frontend/public/benchmark-catalog.json
```

The recorded baseline is from the pre-repair engine, not a claim that all 250 cases ran on
the repaired version. Running the full manifest on a newer engine produces new observations.
Keep the recorded baseline unchanged when reproducing a targeted repair overlay.

The runner gives each case a fresh process, at most 120 seconds wall time and a
sampled 4 GiB process-tree RSS ceiling. The Windows memory ceiling is sampled,
not an OS allocation guarantee. Cases run serially. Report JSON must round-trip,
graphs must validate, public graph/artifact schemas must validate, and an owned
node/tensor response is sampled where available. Serialized reports must fit the
32 MiB service storage contract. Weight payloads and forward execution are not
part of this compatibility benchmark. The engine's recorded zero-payload and
no-forward fields are source-backed assertions, not an independent network trace.
These measurements do not establish task accuracy, inference throughput or
runtime numerical equivalence.

A complete static result requires all four bounded coverage ratios to equal one
and no warnings, unknown nodes or unmapped tensors. Partial graphs, artifact-only
inspection and failed attempts remain separate outcomes. An artifact-processing
error alone does not prove invalid upstream metadata. No publisher Python or RL
policy archive is executed to fill a support gap.

The manifest fingerprint hashes sorted, compact UTF-8 JSON (`canonical-json-v1`),
so checkout line endings do not change its identity.

Publication requires exactly 250 unique case IDs, matching pinned revisions,
case identities, runner/engine hashes and environment budgets. Mixed or missing
records, inflated completion claims and invalid service/storage contracts stop
publication. Failed attempts retain status and a fixed stage label but do not publish
unvalidated architecture metrics or raw exception text. Public diagnostics use
fixed reason messages; the portable evidence also omits raw diagnostic strings,
which remain in the local run records. Public rows are evidence records,
not saved reports or visitor analyses.

## Catalog and pinned inspection

`/catalog` defaults to measured outcomes with 24-row pagination, search, category
and outcome filters, revision links, warnings and coverage details. Stored
inspections remain a separate database-backed view. A benchmark action opens the
normal submission flow at its exact revision; submitting creates a real
visitor-owned analysis. It never opens a fabricated
analysis from the benchmark snapshot.

Live HTTP verification found that the existing API rejected repository URLs of
the form `/tree/<commit>`. API and worker parsers now accept this form for a
lowercase 40-character commit only, preserving automatic checkpoint selection.
Explicit artifact references retain their existing selection semantics. Regression
tests cover URL rejection boundaries, API acceptance and exact-commit cache reuse.
A fresh live inspection completed, reopened from the registry, and returned 404 to
a different visitor; see the companion HTTP evidence JSON.

## Validation

- Full backend and infra offline suite: **496 passed**, one platform skip, 18 integrations
  deselected. The existing Starlette/httpx deprecation warning remains.
- Ruff over backend/scripts/infra and Python compilation passed.
- Frontend typecheck (production build) and production build passed. Vite retains its
  large-chunk advisory. Focused formatting of the edited catalog files passed.
- **36 desktop/mobile browser tests passed** with one worker. Actual-data acceptance
  verified all 250 published rows across 11 pages, their recorded statuses/facts/diagnostics,
  320px layout, exact selected-artifact/revision actions, and recheck history labels.
- Live engine `2026.10.20.2` on port 8000 completed a fresh uncached pinned inspection,
  reused the exact saved revision through the registry, and returned 404 to a foreign visitor.
- Repository-wide frontend `format:check` still fails on **77 pre-existing files**.
  All 77 are unchanged relative to pre-repair commit `51fc4a6`; edited catalog files pass
  their focused check. The PR remains a draft pending that broader frontend formatting gate.

The earlier 24 completed observations collected before the pinned-URL fix are
retained locally for diagnosis and are excluded from this final 250-case run.

## Task-category observations

Categories describe the curated test intent; publisher task labels remain attributed declarations.
These counts do not establish task accuracy, inference performance or complete family semantics.

| Category | Repositories | Complete static | Partial graph | Artifacts inspected | Failed attempt |
| --- | ---: | ---: | ---: | ---: | ---: |
| audio-classification | 4 | 0 | 2 | 2 | 0 |
| audio-to-audio | 4 | 0 | 0 | 4 | 0 |
| automatic-speech-recognition | 4 | 0 | 1 | 3 | 0 |
| causal-lm | 13 | 7 | 5 | 0 | 1 |
| code-lm | 3 | 2 | 1 | 0 | 0 |
| depth-estimation | 6 | 0 | 3 | 3 | 0 |
| document-understanding | 1 | 0 | 0 | 1 | 0 |
| embedding | 9 | 0 | 8 | 1 | 0 |
| extractive-qa | 2 | 0 | 2 | 0 | 0 |
| feature-extraction | 4 | 1 | 3 | 0 | 0 |
| fill-mask | 5 | 0 | 4 | 1 | 0 |
| gguf-quantized | 4 | 0 | 0 | 4 | 0 |
| image-captioning | 1 | 0 | 0 | 1 | 0 |
| image-classification | 11 | 2 | 4 | 5 | 0 |
| image-segmentation | 9 | 0 | 3 | 6 | 0 |
| image-text-embedding | 2 | 0 | 1 | 1 | 0 |
| image-text-to-text | 8 | 0 | 4 | 4 | 0 |
| image-to-image | 4 | 0 | 0 | 4 | 0 |
| image-to-text | 4 | 0 | 1 | 3 | 0 |
| image-to-video | 5 | 0 | 0 | 5 | 0 |
| neural-audio-codec | 1 | 0 | 0 | 1 | 0 |
| object-detection | 8 | 0 | 3 | 5 | 0 |
| peft-adapter | 1 | 0 | 0 | 1 | 0 |
| protein-model | 4 | 0 | 2 | 2 | 0 |
| question-answering | 4 | 0 | 3 | 1 | 0 |
| reranker | 2 | 0 | 2 | 0 | 0 |
| rl-policy | 2 | 0 | 0 | 2 | 0 |
| rl-policy-decision-transformer | 3 | 0 | 0 | 3 | 0 |
| rlhf-reward-model | 2 | 0 | 0 | 2 | 0 |
| rlhf-sft-lm | 2 | 0 | 1 | 1 | 0 |
| robotics-policy | 4 | 0 | 0 | 4 | 0 |
| sentence-similarity | 4 | 0 | 2 | 2 | 0 |
| sentiment-classification | 2 | 0 | 1 | 1 | 0 |
| seq2seq | 2 | 0 | 2 | 0 | 0 |
| seq2seq-instruct | 2 | 0 | 2 | 0 | 0 |
| speech-recognition | 5 | 0 | 2 | 3 | 0 |
| speech-representation | 1 | 0 | 0 | 1 | 0 |
| speech-to-speech-translation | 1 | 0 | 0 | 1 | 0 |
| summarization | 6 | 0 | 2 | 4 | 0 |
| tabular-model | 4 | 0 | 0 | 4 | 0 |
| text-classification | 6 | 0 | 5 | 0 | 1 |
| text-classification-encoder | 2 | 0 | 2 | 0 | 0 |
| text-generation | 5 | 4 | 1 | 0 | 0 |
| text-to-audio | 4 | 0 | 0 | 4 | 0 |
| text-to-image | 12 | 0 | 0 | 10 | 2 |
| text-to-speech | 10 | 0 | 0 | 10 | 0 |
| text-to-video | 8 | 0 | 0 | 8 | 0 |
| time-series-forecasting | 4 | 0 | 2 | 2 | 0 |
| token-classification | 4 | 0 | 3 | 1 | 0 |
| translation | 8 | 0 | 2 | 6 | 0 |
| unconditional-image-generation | 4 | 0 | 0 | 4 | 0 |
| video-classification | 6 | 0 | 5 | 1 | 0 |
| video-text-retrieval | 1 | 0 | 0 | 1 | 0 |
| visual-question-answering | 6 | 0 | 1 | 5 | 0 |
| voice-cloning-tts | 1 | 0 | 0 | 0 | 1 |
| voice-conversion | 2 | 0 | 0 | 2 | 0 |
| zero-shot-image-classification | 4 | 0 | 2 | 2 | 0 |
