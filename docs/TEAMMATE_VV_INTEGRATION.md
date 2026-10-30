# Teammate feat/vv integration

This increment selectively integrates useful compatible work from
[`feat/vv` at 56ffad8f67afb83076c3067a90d846de1e88d83d](https://github.com/pranavsinghpatil/model-anatomy/tree/56ffad8f67afb83076c3067a90d846de1e88d83d).
It imports the validation evidence and scope decisions into catalog discovery,
and adapts staged report-file publication with rollback for ordinary write errors.

## Source evidence and scope

The October 25 audit found that later integration commits replaced
`output/hf_model_parse_results.csv` with a newer run. The original publication
below remains pinned to `56ffad8`; its exact bytes are now retained separately
as `backend/tests/fixtures/vv-validation-56ffad8.csv` for reproducible tests.
The current output CSV is preserved unchanged. Do not apply the older digest or
counts to the newer run, or overwrite a historical publication with it.


The source files are retained byte-for-byte, including their original newlines
and whitespace inside diagnostic fields. These two evidence CSVs are exempt from
Git newline normalization and whitespace lint through `.gitattributes`:

| File | SHA-256 |
| --- | --- |
| `output/hf_model_parse_results.csv` | `53332c2c68c8ff23b837ebb7fd299bd0d9834b056a39e3e7174d5bdf897bd9a4` |
| `data/hf_model_ids.csv` | `b95d5ddbcd6e5746cd5bc5bc09f8e767a88be7c91d823c93f7cd439c067d6a20` |

The results contain 163 input rows and 162 distinct models/observations. Rows
139 and 140 repeat `microsoft/unispeech-sat-base`; one catalog observation retains
both source row numbers. No source row is silently discarded.

| Source result | Source rows | Distinct observations | Current target scope |
| --- | ---: | ---: | --- |
| Reported complete module anatomy | 48 | 48 | In scope |
| Reported partial module anatomy | 10 | 10 | In scope |
| No publisher Safetensors checkpoint | 81 | 80 | Excluded |
| HTTP 401 access restriction | 7 | 7 | Excluded |
| Other failures | 17 | 17 | In scope, unresolved |
| Total | 163 | 162 | 75 in scope, 87 distinct exclusions |

The 17 remaining failures comprise eight artifact errors, eight native
architecture limitations, and one unavailable architecture configuration.
Exclusions are recorded separately from parser failures. An access exclusion
records the source attempt's circumstances; it does not claim the model will
always be inaccessible. Three access rows lack a resolved revision and remain
explicitly unpinned.

A fresh bounded, metadata-only check confirmed no `.safetensors` files in the
pinned publisher inventories for all **80 distinct no-Safetensors models**.
The check read **218,960 JSON metadata bytes** and **zero weight payload bytes**.
See [inventory verification](benchmarks/vv-safetensors-inventory.json).
Compound source reasons may also mention config/backbone limitations; these do
not make a checkpoint-free model eligible for this Safetensors target. No full
weights are downloaded or converted, and no timm/native adapter is added for
these excluded cases.

## Persistence and discovery

Migration `0005_external_validations` adds a server-only immutable publication
registry with RLS and browser-role grants revoked. Import validates the complete
bounded CSV before writing, retains source commit/hash provenance, deduplicates
identical observations with source row attribution, and is idempotent. It exposes
only allowlisted fields. Raw provider error prose, suggested fixes, local graph
paths and local diagnostic paths are not published through the API.

The discovery feed attaches external observations to the exact model/revision
selection. Existing saved-report and benchmark facts retain priority, status,
metrics and links. New source-only models become discoverable with distinct
`reported_complete`, `reported_partial`, `reported_failure` or `out_of_scope`
outcomes. A reported graph/node count is labeled as reported evidence. Missing
parameter, tensor and coverage facts stay unknown. The CSV cannot create a
cached report or a visitor analysis: its referenced full reports live in omitted
teammate-local directories and are not included in the branch.

The source's 48 complete and 10 partial outcomes describe
`static_module_architecture`: registered modules, checkpoint mapping, native
selection and bounded module-call topology. They do not establish the current
engine's full tensor-expression/activation-shape closure. Source provenance and
that mode remain explicit. Importing this evidence never rewrites the immutable
250-model benchmark publication or promotes an existing partial report.

Example trusted-server import after applying migrations:

```powershell
python scripts/import_model_validation_csv.py output/hf_model_parse_results.csv --source-repository https://github.com/pranavsinghpatil/model-anatomy --source-commit 56ffad8f67afb83076c3067a90d846de1e88d83d --source-path output/hf_model_parse_results.csv --env-file D:/neb-hf/infra/.env
```

## Selected code improvement

The adapted `output_transaction.py` stages all report files before publication.
The renderer publishes model JSON, source config, artifact audit, diagnostics,
Mermaid graphs, preview and stale graph cleanup together, restoring the previous
file set if a caught filesystem error interrupts publication. Existing semantic
wording, output names, graph identities and source config serialization remain
compatible. Existing symlink targets are rejected. This protects against ordinary
caught write/rename/unlink errors; it is not a crash-safe or concurrent-writer
filesystem transaction.

## Compatibility choices

The whole teammate branch is not merged. A trial merge produced twelve conflicts
in core extraction, service and test code. The incoming branch disables quota
rejection and uses a different extraction/completion contract. Its dependency pin
to Transformers 5.20.2 also differs from the installed 4.57 source-fingerprinted
contracts. The current API/catalog/registry, quotas, metadata redirect/auth/JSON
protections, semantic extraction and native contracts are retained.

The module-topology extractor, native class selection and construction guards
are recorded as candidates for a separately scoped integration with versioned
contracts and pinned comparisons. Importing their results is not evidence that
our engine has reproduced those 58 graphs. Polars is unnecessary for this bounded
CSV import; no new engine dependency or cache identity is introduced here.

## Verification

Acceptance completed on **2026-10-22** against the Supabase-backed app on API
port 8000 and frontend port 5176.

- Full offline Python suite: **852 passed, 2 skipped, 18 network integration tests
  deselected**. Skips cover Linux-specific behavior and Windows symlink creation.
- Desktop/mobile catalog regressions: **46 passed** across the discovery and
  benchmark suites. This includes source links, deduplicated history, scope
  filters, native metric priority and narrow-screen layout.
- Changed Python passes Ruff and compilation; frontend formatting, TypeScript
  checking and production build pass. Vite retains its existing >500 kB bundle
  advisory. `git diff --check` passes.
- Import is idempotent under publication
  `904a42a2dbd476814557f2dc18a03633c47b174fbae0397fc8386a6411a47d60`.
  All **400 prior reports**, four benchmark publications and their cases retain
  identical identities/content digests. The feed includes **380 discovery
  records**, all **250 benchmark models**, and all **162 external observations**.
- Live source filters show **75 in-scope observations and 87 exclusions**. RLS is
  enabled and neither `anon` nor `authenticated` has table SELECT privileges.
- A fresh inspection of `sail/poolformer_s12` at
  `01bcf57546aceaccba581172497434bb0b377749` produced a **partial** native
  report: **19 graphs, 132 tensors and 11,915,176 unique parameters**. It saved
  automatically and became the catalog's primary result, bringing durable report
  count to **401**. Its teammate-reported complete outcome and **49 reported
  graphs** remain separate history; they do not promote the native report.
- Exact cached reopening and a separate visitor's reopening succeeded, with
  visitor histories isolated. Expanded external history fits a 320 px viewport.

A transient Supabase DNS/connection failure stopped the worker during acceptance.
After DNS recovered, the stopped worker was restarted with the existing private
profile; database, storage and worker health all passed before live inspection.
No engine retry behavior is changed by this increment. Universal reconstruction
completeness remains unproven; the partial native outcome above is retained.

Sanitized machine-readable results:
[cloud preservation](benchmarks/vv-cloud-acceptance.json) and
[live catalog acceptance](benchmarks/vv-live-acceptance.json).
