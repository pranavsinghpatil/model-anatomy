# Supabase catalog alignment

The 250 measured repository outcomes now live in PostgreSQL. The catalog reads
`GET /api/v1/benchmarks/catalog`; the committed JSON export remains reproducible
evidence and is no longer the page's runtime data source.

All 250 original case definitions, the baseline and repair run payloads, and
their published provenance are retained in private database tables. Publication
IDs are canonical JSON SHA-256 digests; importing the same publication is a no-op.
The original five evidence files and subsequent report-capture/import receipts
are also preserved in private Supabase Storage with verified content hashes.

Full reports were captured separately with pinned revisions and bounded static
inspection. Of the 250 cases, 245 produced validated reports: 16 complete static
reports, 87 partial graphs and 142 artifact inspections. Four access failures
and one malformed-artifact failure remain recorded without fabricated reports.
These are anatomy inspections, not inference quality, latency or generation tests.

Each imported report passes content-digest/size, pinned identity, engine/schema,
IR round-trip and recorded service-contract checks. Uploads are read back before
the report registry is committed. Re-running the importer verifies existing
content before reuse. Errors record safe stage/type diagnostics and resume per case.

The API attaches an exact compatible `stored_report_id` to each of the 245
successful cases. **Open stored report** creates a visitor-owned cached analysis
through the existing registry API and opens the workspace's actual query-based
architecture route. Cases without reports retain their exact pinned inspection
action. API/loading errors remain visible, with no static catalog fallback.

The preview runs on `http://127.0.0.1:5176/catalog`, backed by port 8000, PostgreSQL
and private Supabase Storage. Its previous 12 browser analyses and seven report
blobs were copied and verified before switching, preserving ownership and IDs.
New reports and future worker results use the same database/storage configuration.

## Repeating the import

Use a backend environment with the project's API and engine dependencies. Apply
`alembic upgrade head` against the intended database first. Load private database
configuration through the existing environment mechanism; never put credentials
in commands or Git. From the checkout:

```powershell
python scripts/publish_benchmark_catalog.py --store-only --env-file D:/neb-hf/infra/.env
python scripts/store_benchmark_reports.py --capture-only --manifest benchmarks/model-matrix-250.json --output output/benchmark-report-capture.ndjson --report-dir output/benchmark-reports --resume
python scripts/store_benchmark_reports.py --store-only --manifest benchmarks/model-matrix-250.json --output output/benchmark-report-capture.ndjson --report-dir output/benchmark-reports --env-file D:/neb-hf/infra/.env --store-log output/benchmark-report-store.ndjson
```

Start the preview using the existing launcher with free ports:

```powershell
./scripts/start-local.ps1 -Supabase -EnvFile D:/neb-hf/infra/.env -ApiPort 8000 -FrontendPort 5176 -PythonPath D:/neb-hf/.venv/Scripts/python.exe
```

The private tables use RLS with no browser-role grants. Full reports are never
public bucket URLs. Existing visitor ownership checks guard report reads/exports.

## Verification and remaining boundaries

- Final full offline backend/infrastructure suite: 511 passed, one Linux-only
  skip and 18 network integration cases deselected. Ruff, compilation and diff
  checks passed. Earlier response-schema/index mismatches were fixed; a
  concurrent-load spawn timeout passed in isolation and in the final full run.
- Every one of the 245 imported report blobs passed read-back verification; all
  250 catalog rows and both original run histories were queried from Supabase.
- Live HTTP checks exercised saved reports across all 56 categories that had a
  report, including graph/node or artifact views and foreign-visitor rejection.
- Fresh pinned SmolLM2 reconstruction completed with `cached: false`; subsequent
  registry reopening returned `cached: true`. JSON, ZIP, Mermaid, Markdown,
  diagnostics and audit exports all passed; the ZIP integrity check passed.
- The catalog browser suite passed 32 desktop/mobile checks, including all 250
  published rows, pagination, exact pins, saved-report routing, failures and 320px layout.
- Frontend production build, regenerated OpenAPI contract, typecheck and focused
  formatting passed. The existing large-chunk build advisory remains.

The optional AI explanation provider is unconfigured; health reports
`ai_available: false`, and questions return the explicit `ai_unavailable` error.
Its unavailable state does not block inspection, saved reports or exports.
Existing repository-wide frontend formatting findings and public HTTPS/Linux
deployment acceptance remain separate release gates.

See [cloud verification evidence](benchmarks/model-matrix-250-cloud.json) for
counts, model/category checks and the private archive object digests.
