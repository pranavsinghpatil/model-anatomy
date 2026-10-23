# Top integration validation — 2026-10-23

`codex/top-all-preserved` combines the current catalog/reconstruction lineage,
the value-specific dtype and helper-binding work, and the preserved edits from
the earlier checkout. The earlier checkout and `vv/*` refs were left intact.
The first merge commit is `9b35a0f`; the follow-up integration commit is in
this branch's history.

## Live application

The integrated app was launched with the configured Supabase profile at
`http://127.0.0.1:5176/` (API `http://127.0.0.1:8001/`). The `/health` response
reported `status: ok`, with database, storage and worker checks true. The live
`/api/v1/catalog/discovery` response returned **381 distinct catalog records**
and publication ID `947cbb5fb2274c8d2fd9b928c7bd1972c48f67d80220198199be8c8c32dbcf64`.
The browser catalog displayed 381 records, 24 per page, with provenance and
outcome filters. Local launch logs are under `.local/app-v1/supabase/`.

A subsequent Supabase DNS/connection interruption caused the original worker to
exit while releasing its PostgreSQL advisory lock. The worker now retries only
database transport failures with a 1–30 second backoff, closes the stale engine,
and reacquires the lifetime lock and worker lease before handling jobs. SQL,
lease, and configuration errors still fail immediately. The replacement worker's
log is `.local/app-v1/supabase/worker-reconnect-error.log`; the original
failure is retained in `worker-error.log`. After restart, `/health` again
reported database, storage, and worker true, and the catalog still returned
381 records. This is a local runtime check, not a guarantee against future
network outages.

These are catalog records, not 381 complete graphs. At validation time the API
reported 19 completed static reports, 89 partial graphs, 143 artifact
inspections, and separately attributed external/benchmark outcomes. Restricted,
unsupported, and failed models retain their honest statuses.

## Verification

- Backend suite after the reconnect fix: **945 passed, 22 skipped**; one
  external Starlette/httpx deprecation warning. The prior suite log is
  `.local/validation-backend-pytest-final.log`.
- Frontend type check and production build: passed.
- Ruff and Python compile check: passed.
- Changed desktop browser suites: **45 passed**. See
  `.local/validation-frontend-e2e-final.log`.
- `git diff --check`: passed.

The repository-wide Prettier check reports pre-existing formatting differences
across many untouched files; it is not used as evidence that this integration
passes formatting. The combined branch was built and run from its own worktree.
