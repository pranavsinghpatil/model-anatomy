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
- After the combined UI exposed stale scale and color assertions, the affected
  desktop/mobile suites passed **14 tests**, and the shared anatomy scale check
  passed **2 tests**. The workspace navigation gap was reduced to prevent a
  three-pixel horizontal overflow; the current UI scale remains 90%.
- `git diff --check`: passed.

The Windows checkout's CRLF conversion makes a repository-wide Prettier check
report broad differences; CI on Linux isolated one actual formatting mismatch,
which was fixed. The changed UI files pass Prettier. The combined branch was
built and run from its own worktree.

## Branch and PR cleanup

After confirming their heads were ancestors of `codex/top-all-preserved`, PRs
#27 and #30–32 were closed and 19 redundant remote branches were deleted with
exact-SHA leases. `feat/vv`, `codex/integrate-vv-updates`, the active top branch,
and the original dirty checkout branch were preserved. PRs #2, #24, #28, and
#33 retain commits absent from the top branch and remain open. Existing local
worktrees with checked-out branches were left intact, including the running
older previews; four un-checked-out local refs already contained by the top
branch were removed.
