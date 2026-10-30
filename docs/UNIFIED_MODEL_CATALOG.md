# Unified model catalog

The catalog now presents actual saved inspections, recorded terminal failures and
published benchmark observations in one discoverable list. New worker results
save automatically and appear through the same feed. Complete static reports,
partial graphs, artifact inspection and unsuccessful attempts retain distinct
labels. Metrics describe static artifact inspection, not inference quality or
speed.

## Data and identity

`GET /api/v1/catalog/discovery` combines compatible `model_reports`, the latest
immutable `benchmark_publications`/`benchmark_cases`, and safe
`inspection_failures`. The existing paged report and benchmark APIs remain
compatible. Every original benchmark case remains attributable to its publication.

A discovery record identifies model, immutable revision when known, artifact and
explicit selection. Multiple revisions or artifact selections remain separate.
The newest observation supplies that card's status, engine, metrics and report
link together; metrics never come from a different observation. Benchmark
category/task metadata can fill an unknown discovery label with explicit source
attribution. History preserves prior observations and saved-report reopening.

The normal successful finalization path saves private report bytes and durable
report metadata. It now projects warning, graph and tensor counts plus declared
task metadata for discovery. A bounded, explicit backfill reads digest-verified
historical reports, checks identity and enriches missing summary metrics. It
preserves original status, coverage, identity, timestamps and private blob content.
Unknown values remain unknown; artifact inspection has zero architecture graphs
and makes no unique-parameter or semantic-coverage claim.

Terminal failures persist safe identity, generic outcome and timestamps separately
from visitor jobs. They do not contain visitor IDs, job IDs or provider messages.
A requested immutable revision stays requested evidence if resolution failed;
its existence is not inferred. Unverified failed visitor artifact paths are withheld from public fields; an
opaque selection hash keeps distinct attempts separate. Retry links retain the
known/requested revision and open repository inspection without publishing the
unverified filename. Published benchmark references remain attributed evidence. Transient retries and cancellations do not publish
failure records. Newer failures keep earlier valid reports available in history.
Apply migration `0004_inspection_failures` before starting the updated Supabase
profile. PostgreSQL enables RLS and denies browser-role table access; public reads
pass through FastAPI's model-fact projection.

## Browser behavior

`/catalog`, including legacy view query links, shows one list with status totals,
search, category/outcome filters, real metrics and expandable provenance/history.
Search includes model ID, task, architecture, revision and artifact. Cards open
exact saved reports as new visitor-owned analyses, or start a new inspection from
the recorded selector. No visitor's original analysis becomes public.

Metadata refetches on entry/focus and every 15 seconds while visible. Manual
refresh is available, background refresh keeps existing data visible, and filter
changes reset pagination. Historical reports from older engines remain attributed
to those engines; new inspections do not rewrite benchmark measurements.

## Verification

The full offline Python suite passed: **830 passed**, one platform skip and 18
network integration cases deselected. A sandboxed first run stalled before
progress; the completed run used the project virtualenv with plugin autoload
disabled and traceback diagnostics. The final focused discovery, registry, benchmark service, worker and migration
suite passed **64 tests**, with one platform skip, after the review fixes. Ruff, compilation and
`git diff --check` pass.

The Supabase backfill enriched **398 of 398 reports with zero failures**. A
before/after registry snapshot confirmed every original report ID, digest,
revision, engine, inspection timestamp and result status remained unchanged.
Migration head is `0004_inspection_failures`; live checks confirmed RLS is
enabled and both `anon` and `authenticated` lack table SELECT privileges.

The catalog browser suites passed **20 desktop and 20 mobile checks**, including
new-record refresh, outcomes/metrics, provenance, prior saved-report reopening,
redacted failed selections and 320px publication-hash wrapping. TypeScript, the
production build and changed-file Prettier checks pass. The existing large vendor
chunk advisory remains. The final read-only review found the requested privacy,
backfill and history issues closed.

Live acceptance through the Supabase-backed worker and API passed:

- All **250 benchmark models** remain present with their original publication
  provenance. The unified feed now contains **255 records**: 19 complete static
  reports, 88 partial graphs, 142 artifact inspections and six failed outcomes.
- A new `EleutherAI/pythia-70m-deduped` inspection at
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` produced a partial report with
  **5 graphs, 94 tensors and 70,426,624 unique parameters**. Those displayed
  metrics matched the exported report. It appeared automatically, survived
  reload, and reopened as an exact cached analysis for two independent visitors.
  A different visitor received 404 for the original visitor-owned analysis.
- A preceding fresh `HuggingFaceTB/SmolLM-135M` inspection at
  `1d461723eec654e65efdc40cf49301c89c0c92f4` saved a complete report. The first
  live run then caught a 320px overflow caused by the full publication hash. The
  hash now wraps, and the final new-model run passed at 320px.
- The real gated `meta-llama/Llama-3.2-3B` attempt at
  `13afe5124825b4f3751f836b40dafda64c1ed062` recorded an artifact-request failure
  with unknown result metrics. Its pin and safe failure record survived API
  restart. This failure does not claim a model artifact defect or a report.
- API **8000**, frontend **5176** and the worker run this branch. After restart,
  database, private storage and worker checks are all healthy. Engine/cache
  identity remains `2026.10.21.2`; IR remains `1.1`.

[Portable live acceptance](benchmarks/unified-catalog-acceptance.json) contains
the measured facts. Local scripts, screenshots and logs are under
`D:/neb-hf/.local/unified-model-catalog-runtime/`. No source/log archive upload
is part of this change.

## Practical bounds

This increment returns the complete deduplicated discovery set and paginates in
the browser. Server-side filtering/pagination and database aggregation are a
separate scaling improvement for a much larger registry; no claim of unbounded
catalog throughput is made. The recorded benchmark remains a measured 250-model
cohort. Adding catalog records does not expand reconstruction coverage or make
partial architecture reports complete.
