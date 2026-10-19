# Database-backed inspections and catalog

The catalog is built from actual completed inspections. It starts empty; there
are no seeded model cards, prepared reports, or fallback architecture facts.
PostgreSQL stores report identities and projected metadata. Full reports stay in
the private Supabase Storage bucket, addressed by their content digest.

## Inspection requests

`POST /api/v1/analyses` checks the durable registry before admitting reconstruction
work. A compatible stored report creates a new analysis owned by the requesting
visitor and returns `cached: true`. It does not contact Hugging Face, run the
engine, or require an available worker. Admission quotas still apply.

Compatibility includes the model, requested reference, resolved immutable
revision, checkpoint filename and explicit selection, engine version, and IR
schema version. A plain model ID reuses its most recent stored inspection requested from `main`; this
is a saved revision, not a claim that Hugging Face's current `main` is unchanged.
Use `refresh: true` to request a new inspection. Explicit revisions and different
checkpoints cannot silently reuse another result.

On a miss, the existing worker resolves the real Hugging Face artifacts,
inspects them, uploads the resulting report, and publishes its metadata to the
registry. Failed jobs do not become catalog entries. Partial architecture
reports and artifact-only reports retain their actual status and limitations.

## Catalog

- `GET /api/v1/catalog?q=&offset=0&limit=24` searches stored metadata with bounded
  pagination. The response is `{models, total, offset, limit}`.
- `POST /api/v1/catalog/{id}/inspect` opens that exact stored revision and
  checkpoint as a new visitor-owned analysis.

Catalog entries expose model facts, inspection time, and pinned identity. They
do not expose guest identifiers, analysis history, storage credentials, or raw
private-storage URLs. Unknown parameter counts and dimensions remain unknown.
The browser shows loading, empty, and error states instead of sample models.

Visitor history still expires after 24 hours. The durable registry and its
referenced report objects survive that cleanup and the short-lived worker
cache's expiry. Existing engine/schema versions remain part of identity; a
version change requires a compatible new inspection.

## Supabase runtime

Apply the repository's Alembic migrations before starting the hosted profile.
PostgreSQL startup verifies the actual migration head and never creates tables
automatically. The registry table has RLS enabled and browser-role grants
revoked; trusted server SQL accesses it. The frontend calls FastAPI rather than
receiving a Supabase server key.

The Supabase launcher reads credentials from the ignored environment file and
fails if they are missing. It must not silently select SQLite or local storage.
The separate SQLite development profile remains explicit. Windows development
does not establish Linux production memory-limit or deployment acceptance.

## Acceptance evidence

Live Supabase verification on October 20, 2026:

- `/health` returned HTTP 200 with database, storage, and worker healthy and
  `storage_backend: supabase`; the catalog initially contained zero records.
- A real inspection of `HuggingFaceTB/SmolLM2-135M` completed with one worker
  attempt at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Supabase saved its private report and catalog metadata: `LlamaForCausalLM`,
  134,515,008 parameters, 30 layers, hidden dimension 576, and nine heads.
- A second visitor request returned the completed saved revision with
  `cached: true`. Opening the catalog record also returned that exact revision.
- A different visitor received HTTP 404 when requesting another visitor's
  analysis. Cloud table probes confirmed browser-role access is denied.
- Supabase security advisors reported only the expected `rls_enabled_no_policy`
  informational notices: no browser policies are intentional for server-only
  tables ([explanation](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy)).

The report describes static reconstruction; forward execution and runtime
tracing were not performed. No runtime model records were seeded. Automated
regression tests use controlled fixtures separately from the live catalog.

The development preview for this worktree used API port 8002 and frontend port
5175. To start it again from this checkout with the shared private environment:

```powershell
./scripts/start-local.ps1 -Supabase -EnvFile D:/neb-hf/infra/.env -ApiPort 8002 -FrontendPort 5175 -PythonPath D:/neb-hf/.venv/Scripts/python.exe
```

Use free ports and one worker per database. See [Supabase setup](SUPABASE_SETUP.md)
for a normal checkout and fresh-project provisioning.

## Automated verification

- Full offline Python suite: 334 passed, one skipped, 16 network integration
  cases deselected. Migration upgrade/schema alignment/downgrade, registry reuse,
  forced refresh, retention, quotas, visitor isolation, and existing SQLite
  upgrades are included. A process-start timeout observed during concurrent
  browser load did not recur in the final full Python run.
- Ruff, Python compilation, PowerShell parsing, and `git diff --check` passed.
- Frontend TypeScript and production build passed; focused catalog and landing
  browser checks passed on desktop and mobile.
- The broad browser run recorded 136 passed, 26 skipped, and 24 failed. It exposed
  outdated request/health assertions alongside layout and timeout failures.
  Updated change-specific checks are verified separately; this run is not a
  clean browser acceptance result.
- Repository-wide formatting remains failing on existing files. Targeted
  formatting checks cover the changed TypeScript and generated API contract.

The live Supabase behavior above is verified independently of browser test
fixtures. Linux container deployment and a clean full browser acceptance run
remain separate release gates.
