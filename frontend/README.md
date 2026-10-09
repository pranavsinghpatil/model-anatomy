# Model Anatomy Frontend

React + TypeScript + Vite workspace for the service's IR 1.1 architecture graphs.
The primary canvas uses React Flow and ELK layout in a bundled WebWorker. Only
backend-returned shallow graphs are rendered; nested graphs are fetched on demand.
No model artifacts or fabricated graph fallback are bundled with the application.

## Local Run

```powershell
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. `/api` and `/health` proxy to `http://127.0.0.1:8001` by default.
Set `API_PROXY_TARGET` in the environment or `.env.local` to change the upstream.
Browser requests use `/api/v1` and root `/health` with same-origin guest cookies.
In production, serve `dist/` and reverse-proxy both `/api` and `/health` to the HTTP service.
On entry, the workspace restores the visitor's most recent job when one exists.
Otherwise it opens with an empty canvas and model input.

The canvas starts at a readable scale rather than shrinking a long architecture
to fit the screen. Fit graph gives the complete overview; Readable view restores
the inspection scale. Search and the minimap navigate components outside the
viewport. Desktop history is collapsible, and the inspector opens on selection
instead of reserving space for an empty panel. Mobile keeps separate workspace
views. Architecture facts and graph direction are unchanged by these controls.

## Verification

```powershell
npm run typecheck
npm run build
npx playwright install chromium
npm run test:e2e
```

Playwright runs desktop and mobile Chromium checks on port 5174, with the `job-test`
route fixture derived from the repository's real `output/gpt2` IR artifact.
Lifecycle/error scenarios are test-only. The production app has no
mock mode. With the HTTP service running on 8001, set `E2E_LIVE_API=1` to include
the separate `e2e/live.spec.ts` integration tests, which use no route interception.
Before starting the local HTTP service for live tests, include
the test server's port 5174 in its explicit allowed origins:

```powershell
$env:MODEL_ANATOMY_CORS_ORIGINS = '["http://127.0.0.1:5173","http://localhost:5173","http://127.0.0.1:5174"]'
```

Restart the HTTP process after changing this setting. The proxy preserves the
browser Origin header, and the API rejects unlisted mutation origins. These are
explicit local development origins, not production settings. `E2E_BASE_URL` can
override the browser URL (for example, port 5173 with the default API configuration).
Both a running HTTP service and worker are required for live submission. Each
viewport submits one SmolLM2-360M job under its own guest cookie and
verifies its real parameter count and recent-job restoration. Screenshots and traces go to the ignored
`test-results/` directory.

## Contract Integration

`src/api/generated.ts` is generated from the actual service's `/openapi.json` using
`npm run generate:api`. This command includes formatting, so an unchanged schema
produces the same file for CI drift checks. For an offline schema exported from
the actual Python app's `app.openapi()`, use `npm run generate:api -- path/to/openapi.json`.
The offline input is a generated contract, not application data or a graph fallback.
`src/api/contract.ts` contains aliases
imported from that generated file; no handwritten response schemas remain.
Regenerate after backend schema changes, then type-check. `src/api/client.ts` is the
single HTTP boundary and uses generated request and response schemas.

Readiness and optional AI configuration come from the single root `GET /health`.
A 503 health response uses the same health schema as 200; its check results and
reason are retained, and the UI indicates degradation without hiding stored
reports. Only new analyses require confirmed healthy database, storage, and
worker checks plus `analysis_available`. AI availability is independent.

Other required endpoints use `/api/v1`: analyses (list/create), analysis
detail/cancel, graph, paginated node inspection, questions, and exports
(`json`, `zip`, `mermaid`, `markdown`, `diagnostics`, `audit`). HTTP failures must
provide `{error:{code,message,request_id}}`, except for the structured 503 health
response. A visitor with no stored jobs sees an empty workspace. Cached analyses
remain visitor-owned reports and are marked as cached.

Graph/node identities stay scoped by graph ID. Repeated blocks open only the
engine-provided representative child graph; collapsing returns to its parent.
The UI never expands repeated instances into invented nodes. Null dimensions,
parameter counts, dtypes, and shapes remain unknown; numeric zero remains zero.

Citations resolve through exact evidence source/kind/locator matches, or the
service's `stored_ir` locator forms `node:{graph_id}/{node_id}` and `metadata`.
Navigation follows at most 100 returned child graphs; an unresolved association
shows an honest error while retaining the citation source and locator. AI text is
rendered as plain text and cannot change architecture data.

## Ownership

All changes are confined to `frontend/**`. No backend, infrastructure, root
dependency files, branches, commits, or pull requests are created by this workstream.
