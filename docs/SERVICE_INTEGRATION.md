# Service integration

## Current implementation

The FastAPI application factory lives in `backend/model_anatomy/api/app.py`.
The existing reconstruction scaffold remains separate. Starting the HTTP service
does not import Torch, Transformers, or Safetensors; a subprocess test verifies
this property. The package still declares these existing engine dependencies.

| Endpoint | Behavior |
| --- | --- |
| `GET /healthz` | 200 when the HTTP process can answer |
| `GET /readyz` | 503 with `reconstruction_not_implemented` |
| `GET /api/v1/capabilities` | API version and explicit unavailable reconstruction |
| `GET /docs` | Generated OpenAPI explorer; configurable off |

The service generates request IDs and logs method, response status, and elapsed
time, without logging request bodies or URLs. The local launcher binds to
loopback, disables Uvicorn access logs, and caps HTTP concurrency at 32. This is
an initial operational bound, not a measured optimum or a reconstruction limit.
CORS allows the two configured local Vite origins, without credentials. Host
validation rejects unconfigured hostnames. Neither CORS nor host validation
provides authentication.

This increment is a service foundation. It does not yet provide reconstruction,
job persistence, frontend integration, result caching, cancellation, or public
deployment. A live HTTP process must not be described as a ready model service.

## Recommended stack

| Responsibility | Local application | Hosted expansion, when needed |
| --- | --- | --- |
| HTTP contracts | FastAPI, Pydantic, Uvicorn | Same API behind TLS and authentication |
| Inspector | React, TypeScript, Vite | Same static frontend served with API |
| Graph rendering | React Flow; ELK layout in a Web Worker | Same renderer |
| Client data | Generated OpenAPI types; TanStack Query | Same contract and cache |
| Job/result metadata | SQLite, SQLAlchemy, Alembic migrations | PostgreSQL |
| Reconstruction | Separate bounded worker process | Dedicated workers; durable broker if justified |
| Artifacts | Local files with content hashes and atomic writes | Object storage |
| Verification | pytest, HTTPX, Ruff; Playwright for explorer | CI plus deployment smoke and load checks |

FastAPI is an HTTP boundary around deterministic reconstruction. Heavy model
work should run in a separate process with a resource budget. Adding API workers
without measuring memory can duplicate model allocations. FastAPI's documentation
covers [worker memory](https://fastapi.tiangolo.com/deployment/concepts/) and
[heavy background work](https://fastapi.tiangolo.com/tutorial/background-tasks/).

React Flow supports the required inspectable graph. Collapse repeated blocks and
memoize node components; benchmark viewport rendering rather than enabling it
blindly. See its [performance guidance](https://reactflow.dev/learn/advanced-use/performance).

Keep the first installation local and straightforward. Redis, Celery, Kubernetes,
and a desktop wrapper are optional future choices, not dependencies for the first
end-to-end model. Local operation does not imply offline operation: resolving a
new Hugging Face model requires network access; an offline mode requires a complete,
verified cache and explicit cache-miss errors.

## Proposed next API contract

These routes are proposed, not implemented by this PR:

| Route | Contract |
| --- | --- |
| `POST /api/v1/reconstructions` | Validate reference, admit bounded job, return 202 with ID/status URL |
| `GET /api/v1/reconstructions/{id}` | Persisted state, stage, warnings, or structured failure |
| `GET /api/v1/reconstructions/{id}/graph` | Versioned IR, provenance, cache validator |
| `GET /api/v1/reconstructions/{id}/nodes/{node_id}` | Focused tensor and evidence details |
| `DELETE /api/v1/reconstructions/{id}` | Request cooperative cancellation; terminal state persisted |

Start with polling. Add SSE progress if measured UX needs justify it. Do not
stream large tensor metadata through a progress channel. Generate frontend types
from the actual OpenAPI document after these contracts are implemented.

## Production gates and optimization

1. Integrate and verify one pinned model. Audit any code brought from `feat/vv`
   in a separate PR; the current `b-mvp` resolver and reconstructor are stubs.
2. Persist jobs before returning 202. Bound queue depth and worker concurrency;
   start with one reconstruction worker. Define restart recovery, lease expiry,
   cancellation, timeout, and partial/unsupported results with tests.
3. Keep artifact resolution restricted to Hugging Face references and pinned
   revisions. Disable remote model code by default. Set download/header and
   artifact size limits; avoid weight payloads when metadata is sufficient.
4. Key reusable results by canonical model ID, pinned SHA, engine version,
   IR schema version, and extraction options. Use atomic writes and hashes;
   a mutable branch name alone is insufficient cache identity.
5. Page tensor lists and fetch node details on demand. Measure graph layout and
   interaction using the primary model on desktop and mobile viewports.
6. Record cold/warm reconstruction time, peak worker memory, cache size, HTTP
   latency under concurrent jobs, and failure recovery. Set budgets from these
   measurements and retain a reproducible benchmark manifest.
7. Before public hosting, add user authentication, job/result ownership,
   per-user admission limits, TLS, migrations, backups, deployment health checks,
   and log/metric retention. Bind publicly only through a reviewed deployment.

Promote each increment through a topic-branch PR into `b-mvp`. Promote to `main`
through a separate release PR after the one-model end-to-end and benchmark gates.
