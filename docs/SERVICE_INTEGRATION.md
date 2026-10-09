# Application Integration

## Boundaries

FastAPI is the application backend, not a terminal-output wrapper. Its separate
inspection worker calls the same Python engine as the CLI. HTTP startup, graph
views and exports do not import Torch or Transformers. Stored IR is the only
source of graph facts; React renders it without guessing model structure.

| Responsibility | Implementation |
| --- | --- |
| HTTP | FastAPI, Pydantic, Uvicorn; versioned analyses API |
| Jobs | SQLAlchemy; SQLite locally, Supabase PostgreSQL in production |
| Reports | SHA-256-addressed local objects or private Supabase Storage |
| Extraction | One globally leased worker; fresh subprocess per job |
| Browser | React, TypeScript, Vite, React Flow; ELK layout in a Web Worker |
| Questions | NVIDIA Nemotron through Nebius Token Factory; evidence-only context |
| Deployment | Planned CPU-only API/worker images; Docker Compose and Caddy HTTPS |

## Public Contract

Application routes use `/api/v1`. Frontend types are generated from FastAPI's
OpenAPI document, not from a separately maintained specification.

| Route | Behavior |
| --- | --- |
| `GET /analyses` | This anonymous visitor's unexpired jobs |
| `POST /analyses` | Persist an admitted analysis before returning 202 |
| `GET /analyses/{id}` | Status, progress, warnings, revision and dimensions |
| `DELETE /analyses/{id}` | Request cancellation of this visitor's analysis |
| `GET /analyses/{id}/graphs/{graph_id}` | One shallow graph with nested references |
| `GET /analyses/{id}/nodes/{graph_id}/{node_id}` | Facts, paginated tensors and evidence |
| `GET /analyses/{id}/exports/{format}` | JSON, Mermaid, Markdown, diagnostics, audit or ZIP |
| `POST /analyses/{id}/questions` | AI-generated answer with validated report references |

`GET /health` is the single operational endpoint, outside `/api/v1`. It checks
the actual database, migrations, private storage and current worker lease. Its
typed body includes dependency checks, `analysis_available`, configured
`ai_available` and a sanitized failure reason. HTTP 200 means new analyses can
run; 503 means degraded. AI is optional and never changes core health status.
The former health/readiness/capabilities routes are removed, with no aliases.
Existing owned reports remain usable when only the worker or AI is unavailable,
provided their database and storage remain reachable. There is no demo fallback.
Configured AI availability is not proof of successful inference; a deployment
smoke test must establish that separately.

The browser opens a visitor's recent job, or an empty analysis workspace. Prepared
examples, public sample IDs and `/api/v1/examples` have been removed. Every POST
persists a distinct visitor-owned job, resolves the
input's actual immutable revision, and either runs the engine or reuses an exact
revision cache entry marked `cached=true`. Missing repositories fail honestly.
Swagger's same-origin POST/DELETE requests work without adding the API's own URL
to the cross-origin allowlist; untrusted hosts and foreign origins remain rejected.

## Ownership and Spend

An HttpOnly, SameSite visitor cookie contains a random opaque token. The database
stores its hash. Job reads and cancellation require matching ownership; another
visitor's job returns 404. No login is required. Clearing cookies creates a new
visitor; global caps, not cookies, are the hard spend bound.

Defaults: two inspections and ten questions per visitor per hour; twenty
inspections and one hundred questions globally per day. Counters and admission
share a transaction. Ten jobs can wait behind one running inspection. Budgets
are configurable. Visitor jobs expire after 24 hours; reusable reports have a
seven-day cache TTL. There are no separately retained demo reports. Cleanup runs
through the worker.

## Durable Execution

The worker has an exclusive database lease and heartbeats while its child runs.
Jobs record revision, stage, attempts, deadline and lease owner. A stale worker
cannot finish a recovered job. Interrupted jobs retry within a bounded allowance,
preserving the original SHA. Cancellation and timeouts terminate the child.
An idle process that resumes after suspension can renew its expired lease while
holding the exclusive lifetime lock. It cannot steal a different live lease or
silently let a stale running child publish results.

Cache identity includes canonical model ID, exact commit SHA, engine version,
IR schema version and artifact selection. Objects are immutable and checked
against their content hash. Cloud calls occur outside global database locks.
Database and storage are separate systems: crash-orphaned objects require
reconciliation, never blind deletion of live reports.

The production Linux worker has a two-minute deadline and a 4 GiB memory budget,
also bounded by its container. Windows local runs cannot establish Linux memory
enforcement; validate production limits in Docker before release.

## Trust and Failure

Native Transformers implementations are inspected on the meta device. Remote
model code is disabled. Header ranges, shard counts and aggregate metadata have
bounds. No checkpoint weight payload is downloaded. Unknown dimensions,
structural relationships and static-analysis limitations remain visible.

`completed` and `partial` both permit inspection. `unsupported` means outside
the supported native scope. `failed`, `timed_out` and `cancelled` do not become
valid reports. Errors include a code, readable message and request ID without
returning credential-bearing exception details.

Nemotron receives a bounded selection of report facts and original evidence.
Model content and questions are untrusted data, not instructions. Returned
citation IDs must exist; quantitative claims are checked against cited values.
This does not prove every qualitative claim. Answers are labelled AI-generated
and never modify IR. Invalid responses and outages fail explicitly while graphs
continue working.

## Release Evidence

Run the opt-in real HTTP test with `RUN_HTTP_INTEGRATION=1` and
`pytest -q -s backend/tests/test_live_http.py`. It launches a separate API and
worker with a fresh database, submits
`HuggingFaceTB/SmolLM2-360M`, compares dimensions with the exact upstream config,
and checks nested graphs, real tensors, exports, cookie isolation, quotas and
exact-revision cache reuse. No HTTP responses or engine functions are mocked.
Test-only saved engine outputs and mock transports remain under tests/e2e to
exercise deterministic errors and safety checks; they are not runtime data sources.

See [engine acceptance](ENGINE_ACCEPTANCE.md) and
[live HTTP acceptance](LIVE_BACKEND_ACCEPTANCE.md). Local tests do not establish
Supabase connectivity, usable Nebius credits, model availability, HTTPS hosting
or availability during judging. Those remain deployment gates.

Use `pranav/` branches and PRs into `b-mvp`. Promotion to `main` requires a
separate reviewed release PR.
