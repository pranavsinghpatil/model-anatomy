# Immutable Report Cache Acceptance

Verification started October 16 and continued October 17, 2026, local time.
This is a backend persistence increment, not a hosted-production readiness claim.

## Implemented

- Keep real inspected reports by default, independently of 24-hour visitor jobs.
- Preserve legacy cache deadlines that have not already been cleaned up.
- Resolve the requested upstream reference before lookup; retain exact model,
  commit, engine, schema and artifact-selection identity.
- Skip both extraction and upload on a cache hit. Persist a distinct owned job.
- Keep optional operator-selected TTL eviction and unreferenced-object cleanup.
- Fence completion if the reserved object is deleted or marked for retirement.
- Emit correlated `analysis_cache_hit` and `analysis_cache_miss` worker events.
- Document backend-only Supabase setup, private storage and backup requirements.

No API/schema, extraction, frontend, quota or production-timeout changes.
No cloud resources provisioned, credentials configured or running app replaced.

## Evidence

The focused jobs/cache/settings suite passed **32 tests**, including 90-day
simulated history expiry plus restart, concurrent requests using one extraction,
new-commit invalidation, unchanged identity reuse, zero repeated uploads, missing
and corrupt reports, provider outages and completion/cleanup fencing. Existing
identity tests distinguish engine, report schema and explicit artifact selection.

An initial full regression run had **369 passed, 19 skipped, 1 failed**. The
failure was the existing spawned-success-child test exceeding its ten-second
test budget. Both spawned-child cases passed in a separate rerun without
changing the test or the production timeout. The final full suite passed
**372 tests, with 19 skipped**, in 72.97 seconds. Skipped tests include opt-in
network checks and the Linux-only production resource check on this Windows host.

Ruff lint, formatting, compilation and `git diff --check` passed. The first
sandboxed storage run was blocked by
Windows hard-link permissions; storage tests were rerun with permission rather
than weakening atomic storage or mocking its writes.

After the regression suite completed, the isolated real HTTP/worker check passed
in **53.05 seconds**. A fresh SmolLM2-360M inspection produced a completed
architecture with **361,821,120 parameters and 290 tensors**, nested graph and
component/tensor details, JSON/ZIP exports, visitor isolation and quota checks.
Its next submission logged `analysis_cache_hit`, returned the same pinned
revision and skipped extraction. Child execution took 406 ms; the parent recorded
5,094 ms including spawn/coordination, so this is not a subsecond end-to-end claim.
A nonexistent repository returned a genuine provider failure, not a fallback.

## Release Blockers

A fresh, isolated real HTTP/worker run for
`HuggingFaceTB/SmolLM2-360M@f8027fd0eaeea54caa13c31d31b9fdc459c38b49`
resolved the real commit, logged a cache miss, then timed out at the unchanged
120-second job deadline while its last recorded stage was `reading_metadata`.
It produced no completed report and therefore did not verify the live cache-hit
leg in that run. The successful isolated rerun above does not erase this result.
Do not describe the first run as successful or raise the timeout to hide it.
The isolated temporary API/worker were stopped by the test fixture.

This still needs a cold-start/import/provider-timing investigation and deployed
load validation before a production-ready claim. Useful evidence is in the
ignored local `cache-http-tests` and `cache-http-isolated-recheck` directories'
API/worker logs. The successful second run is smoke-test evidence, not proof
of reliable startup or capacity under load.

Supabase PostgreSQL/private-bucket operation, privacy checks, deployed restart,
database-plus-object restore and Linux resource/capacity validation still require
the hosted environment. The database/storage adapters existing in source does
not establish those gates. See `DATABASE_SETUP.md` for setup and manual checks.
