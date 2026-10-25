# Model Anatomy

> Turn the artifacts of an open-weight Hugging Face model into a trustworthy, inspectable architecture graph.

Model Anatomy lets an engineer understand a specific checkpoint without manually joining its configuration, `nn.Module` hierarchy, source, graph traces, and Safetensors metadata. The graph is generated from evidence; it is not a hand-authored template or an LLM guess.

## Product path

```text
model ID / URL → artifact inventory → Raw IR → Semantic IR → graph → inspection
```

The explorer follows the registered module hierarchy from a model overview down
to basic modules such as Linear, Embedding, LayerNorm, Activation, Conv and
Dropout. Basic modules are the expansion boundary. Nodes expose recorded module
attributes, checkpoint tensor metadata, parameter ownership and evidence.
Intermediate expressions such as individual `log` or `exp` operations are outside
this architecture view. Model Autopsy is a later layer built on this representation.

## Current state

The working CLI engine has been integrated from `feat/vv`. Fresh metadata-only
runs verify GPT-2 and the primary target `HuggingFaceTB/SmolLM2-135M`, with no
weight payload downloads or forward execution. See [engine acceptance](docs/ENGINE_ACCEPTANCE.md)
for pinned revisions, tests, and the remaining static-analysis limitations.
The application includes durable anonymous analyses, a separate inspection worker,
an interactive nested graph and tensor inspector, exports, and an evidence-bound
Nemotron explanation integration. Hosted operation still requires cloud setup
and deployment verification; configured integration is not live inference evidence.

The frontend provides a lazy model tree, search across the model hierarchy,
nested graph/node links that survive refresh, per-graph view restoration, and
Graph and Anatomy presentations. Anatomy pages through all components in groups
of 24. Dependency and conditional-dependency edges retain backend evidence; they
do not establish runtime tensor flow. See the [frontend README](frontend/README.md)
for navigation, inspection and API integration details.

Repositories whose architecture cannot be reconstructed can return a distinct
**Artifacts inspected** result with pinned files and verified tensor headers,
never a guessed graph or model parameter total. See
[artifact acceptance and retesting](docs/ARTIFACT_INSPECTION_ACCEPTANCE.md).

GGUF repositories can be inspected separately from architecture reconstruction.
Choose the exact file when a repository contains multiple variants; inspect its
bounded header metadata and GGML-order tensor dimensions without inventing a
graph. See [GGUF checks and manual testing](docs/GGUF_ACCEPTANCE.md).

Category-aware routing now covers additional native text, image, speech, video
and image-text checkpoints, with explicit partial-report limitations. Diffusion
pipelines require component selection. See
[dynamic routing evidence and manual checks](docs/DYNAMIC_ROUTING_ACCEPTANCE.md).

## Inspect a model

Run from the project root:

```bash
uv sync --frozen --extra engine
uv run --no-sync model-anatomy openai-community/gpt2 --inspect
```

The CLI writes configuration to `config/<slug>/`, JSON and Mermaid graphs to
`output/<slug>/`, and an English Markdown preview to `preview/`. It also prints
the resolved revision, output paths and, with `--inspect`, diagnostics. See the
[CLI README](backend/model_anatomy/src/README.md) for all arguments, filenames,
completion rules and the optional `--artifacts` inspection mode.

The architecture support target is anonymously readable, publisher-provided
Safetensors checkpoints with a compatible installed native implementation.
Repositories without Safetensors and models requiring private/gated access are
outside that target. Fresh metadata inspection requires Hugging Face network
access. Weight payload downloads and forward execution are disabled.

## Run the application

Use three terminals for normal local development. Prerequisites are Python 3.11+,
`uv`, and Node.js 20.19+ on the 20.x line or Node.js 22.12+.
Run the Python commands from the project root.

Install once, or when dependencies change:

```bash
uv sync --frozen --extra api --extra engine
cd frontend
npm ci
```

Select only the extras needed by each environment. A CLI environment needs
`engine`; an API-only environment needs `api`; a worker needs both. A shared
local API/worker environment needs both extras as above. The `dev` extra is
optional for development tools. Keep the required extras when synchronizing
again: plain `uv sync` does not select them. API-only deployment does not require
Torch. API and worker startup provide readable missing-dependency checks.

**Terminal 1 — API**

```bash
uv run --no-sync model-anatomy-api
```

**Terminal 2 — worker**

```bash
uv run --no-sync python -m model_anatomy.service.worker
```

**Terminal 3 — frontend**

```bash
cd frontend
npm run dev
```

The API serves HTTP requests, visitor-owned analyses, saved reports and the job
queue. The worker claims queued jobs, runs reconstruction in a separate child
process, and stores the resulting report. Both use the same backend settings and
database. One worker owns the execution lease; starting a second worker against
the same local database is rejected.

Open [the application](http://127.0.0.1:5173) or [API docs](http://127.0.0.1:8001/docs).
No sign-in is required. There are no prepared reports or demo endpoints: paste
a model ID or URL to create an analysis. Exact-revision caching may reuse a
compatible saved report. `/health` checks the database, storage and worker and
reports configured AI availability. HTTP 503 means new analyses are unavailable;
stored reports remain inspectable when their database and storage are reachable.
These commands bind only to loopback.

### Persistent local configuration

Configuration files are optional. Missing files use defaults; existing process
environment variables take precedence. To customize settings, copy
`backend/.env.example` to `backend/.env` and `frontend/.env.example` to
`frontend/.env` if those files do not already exist. Edit existing files when
present. No temporary shell environment variables are required for normal use.

| File            | Setting                      | Default                               |
| --------------- | ---------------------------- | ------------------------------------- |
| `backend/.env`  | `MODEL_ANATOMY_PORT`         | `8001`                                |
| `backend/.env`  | `MODEL_ANATOMY_DATA_DIR`     | `.local/model-anatomy`                |
| `backend/.env`  | `MODEL_ANATOMY_CORS_ORIGINS` | Local frontend origins on port `5173` |
| `frontend/.env` | `FRONTEND_PORT`              | `5173`                                |
| `frontend/.env` | `API_PROXY_TARGET`           | `http://127.0.0.1:8001`               |

If the API port changes, update `API_PROXY_TARGET` to match. If the frontend port
changes, update backend allowed origins to match. Restart affected services
after editing configuration. Backend credentials belong only in backend
configuration, never in frontend environment files.

Local defaults use SQLite and private local report objects. Web inspections store
job records in `.local/model-anatomy/jobs.sqlite3` and immutable report objects
under `.local/model-anatomy/objects/`. They do not populate the CLI's `config/`,
`output/` or `preview/` directories. Download web reports using Export.

Production uses Supabase PostgreSQL and a private Storage bucket. Backend-only
`NEBIUS_API_KEY` and `NEBIUS_MODEL` enable Nemotron explanations; choose a real
available NVIDIA Nemotron ID, never a placeholder. Graph facts do not depend on
AI. Do not commit secrets or model weights.

### Optional launchers

On Linux, after installing dependencies, the local launcher runs all three
processes in one terminal:

```bash
uv run --no-sync python scripts/start-local.py
```

It reads the same per-directory configuration, refuses occupied ports or an
active worker lease, writes logs under the configured data directory's
`local-launcher/`, and stops only its own processes on Ctrl+C. It does not install
packages or submit models. Do not run it alongside the three manual processes.

The existing Windows launcher is
`powershell -ExecutionPolicy Bypass -File scripts/start-local.ps1`. It uses a
separate local testing profile under `.local/app-v1`, defaults to API port `8000`
and frontend port `5173`, and provides explicit port/quota options. Its defaults
are distinct from the normal API entry point and Linux launcher.

### Development checks

From `frontend`, run `npm run typecheck`, `npm run build` and `npm run test:e2e`.
The API must be running before `npm run generate:api` can refresh the generated
contract from its default OpenAPI URL. See the frontend README for offline schema
generation, deterministic fixtures and the separate opt-in live tests.
Completed reports describe static architecture, not executed model predictions.

See [service integration](docs/SERVICE_INTEGRATION.md) for the trust boundary and
release gates, and [live backend acceptance](docs/LIVE_BACKEND_ACCEPTANCE.md) for
HTTP-to-worker evidence. Historical acceptance documents describe the code and
configuration used at the time of each run; use the commands above for current
local defaults. Cloud deployment is separate: see [deployment](docs/DEPLOYMENT.md),
[benchmarks](docs/BENCHMARKS.md) and the
[submission checklist](docs/HACKATHON_SUBMISSION.md).

## Hackathon fit

For the Nebius × NVIDIA Global AI Hackathon, reconstruction remains deterministic. A clearly separated, optional Nebius Token Factory + NVIDIA model layer may explain already-established evidence in the explorer; it may never create or alter architectural facts. This makes the required model-service usage visible without weakening the product’s trust boundary.

## Documentation

- [Project ground](docs/00-GROUND.md) — promise, scope, and non-negotiable boundaries
- [Architecture](docs/ARCHITECTURE.md) — pipeline, IR, evidence, and failure model
- [Roadmap](docs/ROADMAP.md) — milestones, dates, release gates, and current status
- [Quality](docs/QUALITY.md) — PR, benchmark, demo, and submission requirements
- [Decision log](docs/DECISION_LOG.md) — binding product and architecture decisions
- [Live backend acceptance](docs/LIVE_BACKEND_ACCEPTANCE.md) - real submissions, artifacts and remaining gates

## Development workflow

The active integration base is `b-mvp`. Use `pranav/` topic branches exclusively,
test each increment and open PRs targeting `b-mvp`. Promotion to `main` requires
a separate reviewed PR. See [CONTRIBUTING.md](CONTRIBUTING.md).
