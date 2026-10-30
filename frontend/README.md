# Model Anatomy Frontend

React + TypeScript + Vite explorer for the service's IR 1.1 architecture reports.
React Flow renders the graph; ELK computes layout in a bundled WebWorker.
Graphs and module attributes come from backend reports. Nested graphs are fetched
on demand. No production graph fixture or fabricated fallback is bundled.

## Local run

Start the API and worker in two separate terminals using the
[root README](../README.md#run-the-application). In a third terminal:

```bash
cd frontend
npm ci
npm run dev
```

Skip `npm ci` when dependencies are already installed and unchanged. Use Node.js
20.19+ on the 20.x line or Node.js 22.12+. The default browser address is
`http://127.0.0.1:5173`; `/api` and `/health` proxy to
`http://127.0.0.1:8001`. New inspections require a running worker. Saved reports
can remain available while the worker is offline.

Optional persistent configuration belongs in `frontend/.env`. Copy
[.env.example](.env.example) if no local file exists, then edit as needed:

```dotenv
FRONTEND_PORT=5173
API_PROXY_TARGET=http://127.0.0.1:8001
```

Missing files use defaults; existing process environment variables take priority.
Vite also supports its standard mode-specific and local environment files.
Restart Vite after editing configuration. If the browser port changes, update
`MODEL_ANATOMY_CORS_ORIGINS` in `backend/.env` and restart the API. Backend
credentials belong only in backend configuration.

For production, serve `dist/`, reverse-proxy both `/api` and `/health` to the HTTP
service, and return `index.html` for application routes. Hosted deployment has
separate acceptance requirements.

## Explore a report

Both `/` and `/app` open the model submission and recent-analysis screen. A new
visitor sees an empty history; opening a recent report is explicit. Report views
use `/app/analyses/:id?view=architecture|overview|artifacts|report`.
Legacy `/?analysis=:id` links redirect to the report.

The Architecture workspace follows the report hierarchy from overview to basic
modules such as Linear, Embedding, LayerNorm, Activation, Conv and Dropout. Basic
modules are leaves. Individual tensor expressions are outside this view.

- Select a component to open its inspector. Opening its child graph is a separate
  action; breadcrumbs return to an ancestor.
- **Current graph** lists the fetched graph. **Model tree** expands the full
  registered hierarchy lazily and provides keyboard navigation.
- **Find in current graph** searches visible graph records. **Find across model**
  in Model tree searches names, kinds and module paths after two characters.
  Results use the lightweight location index; selecting one fetches its target
  graph, without fetching every intervening graph. Longer lists offer Show more.
- Graph and selected-node identities are encoded in the URL as `graph` and
  `node`. Refresh, browser history and switching report views preserve that
  location. Invalid locations show an explicit recovery action.
- Fit frames the graph; Readable focuses a component. Direction, card detail,
  connection labels and neighbor focus are available in graph controls. Desktop
  panels resize; narrow screens use a component drawer and separate graph and
  inspector views with Back to graph navigation.

Graph coordinates and inspector/panel state are retained in memory for recently
visited analyses in the current page session. Layout preferences and Graph/Anatomy
choice are also saved in sessionStorage. URL identities restore graph/node
location after reload; exact pan/zoom coordinates are not persisted across reload.
Theme preferences persist separately. Model reports are not stored in localStorage.

Native module paths distinguish repeated class names. Every layer instance in the
returned graph remains available. The UI opens only backend-provided child graphs
and representative repetition records; it does not create extra instances or
merge layers based on their names.

## Inspect modules and relationships

The Component tab prioritizes recorded native module type, scalar attributes and
basic-module boundary. Tensors shows checkpoint shapes, storage dtypes and
parameter ownership; Evidence shows recorded provenance. Missing attributes stay
unreported. Optional input/output specifications are collapsed for static module
architecture reports. Missing activation shapes are not fabricated.

Connections preserve their backend relation:

| Relation                 | Graph appearance | Meaning                                                            |
| ------------------------ | ---------------- | ------------------------------------------------------------------ |
| `dependency`             | Dashed           | Recorded module dependency with intermediate expressions omitted.  |
| `conditional_dependency` | Dotted           | Dependency whose recorded source conditions remain inspectable.    |
| `dataflow`               | Solid            | A relation supplied by the report, including older report formats. |

The legend reflects the current graph. Selecting an edge exposes its relationship,
ports and recorded conditions. Other returned relations remain distinguishable;
the UI does not turn dependencies into runtime tensor flow. Unconnected modules
appear separately, without invented connections.

Complete and partial are backend decisions. The report shows native module
inventory, checkpoint parameter mapping, native architecture selection and module
topology checks, plus blocking diagnostics and warnings. Unknown intermediate
math expressions or absent activation shapes alone do not make a static module
architecture partial. Artifact-only results show files and metadata without an
architecture graph; publisher reference topology remains explicitly partial.

Edit model and retry uses the API's original `model_input`, preserving an explicit
checkpoint file or pinned reference. Older responses fall back to a browser-saved
submission, then a pinned repository reference when available. Tensor artifact
links open a validated, pinned Hugging Face preview page instead of a direct
weight-download URL. Export downloads the backend report, not model weights.

## Anatomy presentation

Architecture also offers **Anatomy** (`scene=anatomy`): an SVG isometric
illustration of the fetched graph. Its geometry is explanatory, not model
hardware, neuron counts, activation values or measured execution.

Each page contains up to 24 actual components. Previous/Next components exposes
the remaining pages; selecting a component moves to its page. Links retain the
reported dependency type and are drawn only when both endpoints are on the
current page. A page's illustrated-link count can therefore be smaller than the
graph's total relationship count. Graph remains available for the full diagram.
Selection, nesting, tensors and evidence share the same inspector. Camera pose
and separate/reassemble controls are local presentation state.

## API integration

`src/api/generated.ts` is generated from the actual service OpenAPI schema.
`src/api/contract.ts` provides aliases; `src/api/client.ts` is the HTTP boundary.
After backend schema changes, start the API and run from `frontend/`:

```bash
npm run generate:api
npm run typecheck
```

The generator defaults to `http://127.0.0.1:8001/openapi.json`. Supply a different
OpenAPI URL or an existing schema exported from `app.openapi()` as an argument:

```bash
npm run generate:api -- path/to/openapi.json
```

The offline file is a contract, not application data. Generation includes formatting.
Restart the API after updating backend code so it exposes the current schema.

Browser requests use `/api/v1` and root `/health` with same-origin guest cookies.
Health HTTP 200 and 503 share the same response schema. New analyses require
healthy database, storage and worker checks plus `analysis_available`; optional
AI availability is independent. Degradation does not hide reachable stored reports.

Required endpoints include analyses, detail/cancel, graphs, paginated node and
artifact inspection, catalogs, questions and exports. Two navigation contracts
support the current explorer:

- `GET /api/v1/analyses/{id}` includes nullable `model_input`, the original
  submitted reference used for retry.
- `GET /api/v1/analyses/{id}/graph-index` returns existing graph parent links,
  node locations and evidence locators. It copies no tensor payload or raw source,
  performs no reconstruction and enforces the existing visitor ownership checks.
  Artifact-only reports have no architecture index.

Graph/node identities remain scoped by graph ID. Citation navigation uses exact
source/kind/locator matches or the stored IR node locator
`node:{graph_id}/{node_id}`. It uses the index without the former 100-graph
traversal limit. Unresolved associations remain explicit. AI text is rendered as
plain text and cannot create or alter architecture facts.

## Performance and accessibility

Layouts are reused per immutable fetched graph, direction and card density.
Graphs with more than 150 nodes render only visible canvas elements; the model
tree loads children on demand. The catalog refreshes once per minute or manually,
and recovers its current page when records shrink. Decorative animation pauses
when the page is hidden and honors reduced-motion preferences.

Light/dark tokens, local Inter/JetBrains Mono fonts, keyboard controls and Radix
primitives form the shared design system. Build output still includes a chunk
above Vite's size-warning threshold; the build succeeds, but further bundle
splitting remains possible.

## Verification

From `frontend/`, the normal development checks are:

```bash
npm run typecheck
npm run build
npm run format:check
npm run test:e2e
```

Install Playwright Chromium beforehand if it is unavailable. The shared Playwright
configuration starts or reuses a test frontend on port 5174. Default artifacts go
to Playwright's test-results and HTML report directories. Local artifact routing
can be supplied by a local runner without changing shared project configuration.

`e2e/service.ts` supplies self-contained HTTP contract fixtures, including legacy
repetition/dataflow examples. It no longer requires `output/gpt2/model.json`.
Those fixtures verify UI behavior, not parser completeness. Production has no
mock mode.

Separate `live-*.spec.ts` cases opt in through the test process's `E2E_LIVE_API`
setting and require real API/worker services. `E2E_BASE_URL` can override the test
browser address. These Playwright process settings are not automatically loaded
from application `.env` files. For live tests on port 5174, add that exact browser
origin to `MODEL_ANATOMY_CORS_ORIGINS` in `backend/.env`, preserving existing
origins, and restart the API. Test quotas and external metadata access can affect
live results.

The latest frontend integration verification passed 146 selected deterministic
desktop/mobile cases, 18 additional local regression cases, type checking and a
production build. A separate fresh-database run exercised the actual API, spawned
worker and frozen parser using pinned cached GPT-2 and GPT-Neo configuration and
header metadata. It verified nesting, search, refresh, tensors, evidence, reports,
exports and mobile navigation, while preserving the parser's complete/partial
outcomes. No weight payloads or forward execution were used. This is local
integration evidence, not a new all-model or hosted-deployment acceptance claim.

Historical design and verification records remain in
[CANVAS_WORKSPACE.md](../docs/CANVAS_WORKSPACE.md),
[ANATOMY_ILLUSTRATION.md](../docs/ANATOMY_ILLUSTRATION.md),
[UI_REDESIGN.md](../docs/UI_REDESIGN.md),
[FRONTEND_ART_DIRECTION.md](../docs/FRONTEND_ART_DIRECTION.md) and
[WHITE_WORKSPACE.md](../docs/WHITE_WORKSPACE.md).
See [THIRD_PARTY.md](THIRD_PARTY.md) for asset and dependency notices.
