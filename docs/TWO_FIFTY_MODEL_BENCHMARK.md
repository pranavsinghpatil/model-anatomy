# 250-model compatibility benchmark

Status: fresh execution is in progress on engine `2026.10.20.1`. Do not read the
250-case manifest as 250 successful reconstructions. The final measured summary
and public catalog export are added only after all cases finish.

This increment combines the database-backed catalog in PR #34 and the validated
engine in PR #32. The initial 120 case definitions are preserved; 130 distinct
repositories extend coverage to 57 intended task categories. Specialist selection
is curated: biomedical protein NER is not protein-sequence modeling, and an
author name containing "tabular" is not evidence of a tabular task. Exact pinned
repository metadata and optional artifact filenames were checked through the
public Hugging Face API. Publisher task labels remain declarations.

## Reproduce

```powershell
python scripts/model_matrix.py --manifest benchmarks/model-matrix-250.json --output .local/model-matrix-250/results.ndjson --summary .local/model-matrix-250/summary.json --resume
python -m scripts.publish_benchmark_catalog --manifest benchmarks/model-matrix-250.json --summary .local/model-matrix-250/summary.json --output frontend/public/benchmark-catalog.json
```

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

Publication requires exactly 250 unique case IDs, matching pinned revisions,
case identities, runner/engine hashes and environment budgets. Mixed or missing
records, inflated completion claims and invalid service/storage contracts stop
publication. Failed attempts retain status and stage but do not publish unvalidated
architecture metrics or raw exception traces. Public rows are evidence records,
not saved reports or visitor analyses.

## Catalog and pinned inspection

`/catalog` defaults to measured outcomes with 24-row pagination, search, category
and outcome filters, revision links, warnings and coverage details. Stored
inspections remain a separate database-backed view. A benchmark action submits a
real visitor-owned analysis at its exact revision; it never opens a fabricated
analysis from the benchmark snapshot.

Live HTTP verification found that the existing API rejected repository URLs of
the form `/tree/<commit>`. API and worker parsers now accept this form for a
lowercase 40-character commit only, preserving automatic checkpoint selection.
Explicit artifact references retain their existing selection semantics. Regression
tests cover URL rejection boundaries, API acceptance and exact-commit cache reuse.
A fresh live inspection completed, reopened from the registry, and returned 404 to
a different visitor; see the companion HTTP evidence JSON.

## Validation

- Backend and infra offline suite: 463 passed, one platform skip, 18 integrations
  deselected. Existing Starlette/httpx deprecation warning remains.
- Ruff and Python compilation passed.
- Frontend typecheck and production build passed; Vite retains a large-chunk advisory.
- Catalog desktop/mobile tests are rerun after final outcome labels and data export.
- Repository-wide frontend formatting is not clean on inherited files. Changed
  catalog files are formatted separately; do not infer a clean repository-wide
  formatting result from those focused checks.

The earlier 24 completed observations collected before the pinned-URL fix are
retained locally for diagnosis and are excluded from this final 250-case run.
