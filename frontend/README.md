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
Configure the production server to return `index.html` for application routes.
The landing page is `/`, submission and visitor-owned history are at `/app`, and
reports use `/app/analyses/:id?view=overview|architecture|artifacts|report`.
Legacy `/?analysis=:id` links redirect to the report. Recent analyses require an
explicit Continue action; a new visitor sees an empty history.

Research mode opens Architecture; Learn opens Overview and offers an authored
guide beside the same report data. Theme (System by default) and mode (Research
by default) are remembered locally. Graph selection, breadcrumbs, viewports,
inspector tab and panel sizes are retained per analysis during the browser session.
Switching mode keeps this context. No model report is stored in localStorage.

The canvas starts at a readable scale. Fit graph gives the complete overview;
Readable view restores the inspection scale. Search is scoped to the current
graph. Selecting a component opens its inspector; opening a nested graph requires
an explicit action. The component list provides a keyboard alternative to the
canvas. Desktop panels resize; narrower screens use a component drawer, and
below 1024px the graph and inspector have separate views with return navigation.

Light and dark semantic tokens, local Inter/JetBrains Mono fonts, Radix controls,
and reduced-motion support form the shared design system. See
[UI_REDESIGN.md](../docs/UI_REDESIGN.md) and [THIRD_PARTY.md](THIRD_PARTY.md).

## Verification

```powershell
npm run typecheck
npm run build
npm run format:check
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
viewport exercises both Research and Learn journeys with real SmolLM2-360M jobs,
graph/component/tensor/evidence inspection, JSON export, and explicit recent
analysis continuation. Other `live-*.spec.ts` files cover artifact-only, GGUF and
ViT results. Screenshots and traces go to ignored `test-results*/` directories.

The deterministic suite includes keyboard interaction, scoped citations,
pagination, cancellation and recovery, unavailable AI, clipboard/export failures,
and responsive axe checks. These route fixtures are test data only. Their evidence
does not replace the separate live service acceptance run.

On 2026-10-13 the redesign passed the production build, formatting and unchanged
generated-contract checks, **92 deterministic desktop/mobile tests**, and all
**14 separate live cases**. Both modes and themes passed the six target widths
from 320 to 1920px, long identifiers and a 200% CSS zoom regression. Twelve axe
scans reported no violations. See the [verification record](../docs/UI_REDESIGN.md#verification-record)
for model evidence and the limits of local Chromium validation.

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

## Scope

This redesign changes frontend presentation and project documentation. Backend
API contracts, reconstruction logic, ownership checks and report schemas remain
unchanged. Artifact-only results expose files, tensors, metadata and supported
exports; they do not gain a fabricated architecture, runtime verification or AI
answers. Model comparison, Model Autopsy and inference remain separate work.
