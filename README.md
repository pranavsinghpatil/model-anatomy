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

## Features

The working CLI engine has been integrated from `feat/vv`. Fresh metadata-only
runs verify GPT-2 and the primary target `HuggingFaceTB/SmolLM2-135M`, with no
weight payload downloads or forward execution. See [engine acceptance](docs/ENGINE_ACCEPTANCE.md)
for pinned revisions, tests, and the remaining static-analysis limitations.
The application includes durable anonymous analyses, a separate inspection worker,
an interactive nested graph and tensor inspector, exports, and an evidence-bound
NVIDIA Nemotron question answering through Nebius Token Factory for researchers,
developers, and students exploring model architecture.

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

Use three terminals: one for the API, one for the worker, and one for the
frontend. Prerequisites are Python 3.11+, `uv`, and Node.js 20.19+ on the 20.x line
or Node.js 22.12+. Start every command block from the repository root; only the
frontend block changes into `frontend/`.
These commands use Bash on Linux, macOS, or WSL.

Install once, or when dependencies change:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" UV_PYTHON_INSTALL_DIR="$PWD/.tmp/python" \
  uv --cache-dir ./.tmp/uv-cache sync --frozen --extra api --extra engine
TMPDIR="$PWD/.tmp" npm --cache ./.tmp/npm-cache ci --prefix frontend
```

Select only the extras needed by each environment. A CLI environment needs
`engine`; an API-only environment needs `api`; a worker needs both. A shared
local API/worker environment needs both extras as above. The `dev` extra is
optional for development tools. Keep the required extras when synchronizing
again: plain `uv sync` does not select them. API-only deployment does not require
Torch. API and worker startup provide readable missing-dependency checks.

**Terminal 1 — API**

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-sync python -B -m model_anatomy.api
```

**Terminal 2 — worker**

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-sync python -B -m model_anatomy.service.worker
```

**Terminal 3 — frontend**

```bash
mkdir -p .tmp
cd frontend
TMPDIR="$PWD/../.tmp" VITE_DEMO_CONTROLS=false npm run dev
```

Keep all three terminals running while using the app. Stop each process with
`Ctrl+C` in its own terminal. The frontend command runs the local application
with cloud Sandbox controls hidden.

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

Local graph inspection uses SQLite and local files without Supabase credentials.
Keep an existing database/storage profile when restarting. The API and worker
must use the same profile so saved analyses remain accessible.

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
local defaults. See [benchmarks](docs/BENCHMARKS.md) for measured reconstruction
behavior and its limitations.

## NVIDIA Nemotron and Nebius Token Factory

Model Anatomy uses an NVIDIA Nemotron model served through Nebius Token Factory
to help researchers, developers, and students ask questions about an inspected
model. Questions can focus on the architecture, a selected component, or recorded
tensor information. The backend supplies relevant inspection facts and evidence
to the model, then returns an explanation with references to that evidence.

NVIDIA Nemotron provides the language-model explanations. Nebius Token Factory
provides hosted inference through its API, so the question-answering feature
does not require running a separate LLM or provisioning a local GPU. This brings
contextual help into the inspection workflow: users can ask about the component
they are exploring while keeping the underlying report available for checking.

Architecture reconstruction remains deterministic. The LLM explains the stored
report; it cannot create or change the graph, tensor metadata, or parameter
counts. The service checks cited evidence IDs and numerical claims against the
supplied context. Explanatory prose remains an AI interpretation, and missing
evidence must stay explicit.

### Configure question answering

Create `backend/.env` from `backend/.env.example` if it does not exist. Otherwise,
edit the existing file without replacing its database or storage configuration.
Add your private Token Factory settings:

```dotenv
NEBIUS_API_KEY=YOUR_NEBIUS_TOKEN_FACTORY_API_KEY
NEBIUS_MODEL=YOUR_AVAILABLE_NVIDIA_NEMOTRON_MODEL_ID
```

Replace both placeholders. Set `NEBIUS_MODEL` to the exact NVIDIA Nemotron model
ID available in your Token Factory account; the application does not assume a
particular Nemotron variant. The default API base URL is
`https://api.tokenfactory.nebius.com/v1`. If your account uses another supported
endpoint, set `MODEL_ANATOMY_NEBIUS_BASE_URL` in the same private backend file.

Restart the API after editing these settings. API keys stay server-side in the
Git-ignored `backend/.env`; never place them in frontend files or `VITE_`
variables. Graph inspection works independently of these AI settings. The
`/health` response reports configured AI availability, which is separate from
checking that your provider accepts a particular model request.

After an architecture inspection completes, open its report and the **Questions**
tab. For example, ask which tensors belong to the selected component or
what a recorded tensor shape means. Review the answer's evidence references
alongside the graph and tensor inspector.

The Token Factory request path is implemented in
[`backend/model_anatomy/service/ai.py`](backend/model_anatomy/service/ai.py), and
the application endpoint is
`POST /api/v1/analyses/{analysis_id}/questions`. This is the Nebius service used
for question answering; local reconstruction runs in the Python worker.

## License

Model Anatomy is released under the [MIT License](LICENSE).

## Documentation

- [Project ground](docs/00-GROUND.md) — promise, scope, and non-negotiable boundaries
- [Architecture](docs/ARCHITECTURE.md) — pipeline, IR, evidence, and failure model
- [Roadmap](docs/ROADMAP.md) — milestones, dates, release gates, and current status
- [Quality](docs/QUALITY.md) — PR, benchmark, demo, and submission requirements
- [Decision log](docs/DECISION_LOG.md) — binding product and architecture decisions
- [Live backend acceptance](docs/LIVE_BACKEND_ACCEPTANCE.md) - real submissions, artifacts and remaining gates

## Development workflow

The active integration base is `feat/vv`. Use `codex/` topic branches by default,
test each increment and open PRs targeting `feat/vv`. Promotion to `main` requires
a separate reviewed PR. See [CONTRIBUTING.md](CONTRIBUTING.md).
