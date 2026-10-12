# Model Anatomy

> Turn the artifacts of an open-weight Hugging Face model into a trustworthy, inspectable architecture graph.

Model Anatomy lets an engineer understand a specific checkpoint without manually joining its configuration, `nn.Module` hierarchy, source, graph traces, and Safetensors metadata. The graph is generated from evidence; it is not a hand-authored template or an LLM guess.

## Product path

```text
model ID / URL → artifact inventory → Raw IR → Semantic IR → graph → inspection
```

The first release proves that path for one decoder-only Transformer. A node must lead to its module path, tensors, dimensions, parameter ownership, and the evidence that established it. Model Autopsy is a later layer built on this representation.

## Current state

The working CLI engine has been integrated from `feat/vv`. Fresh metadata-only
runs verify GPT-2 and the primary target `HuggingFaceTB/SmolLM2-135M`, with no
weight payload downloads or forward execution. See [engine acceptance](docs/ENGINE_ACCEPTANCE.md)
for pinned revisions, tests, and the remaining static-analysis limitations.
The application includes durable anonymous analyses, a separate inspection worker,
an interactive nested graph and tensor inspector, exports, and an evidence-bound
Nemotron explanation integration. Hosted operation still requires cloud setup
and deployment verification; configured integration is not live inference evidence.

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

```powershell
uv sync --frozen --extra engine --extra api --extra dev
uv run --no-sync model-anatomy openai-community/gpt2 --inspect
```

The command produces a model report, nested Mermaid diagrams, diagnostics,
artifact audit, and Markdown preview. Use `--revision` to pin the model and
`--preview-dir` to isolate previews. Network access to Hugging Face is required.

## Run the Application

```powershell
uv sync --frozen --extra api --extra engine --extra dev
uv run --no-sync uvicorn model_anatomy.api.app:create_app --factory --host 127.0.0.1 --port 8001 --no-access-log
```

Run the worker in a second terminal:

```powershell
uv run --no-sync python -m model_anatomy.service.worker
```

Run the browser application in a third terminal:

```powershell
cd frontend
npm ci
npm run dev
```

For manual testing with higher local quotas, stop the existing API, worker and frontend,
then run `powershell -ExecutionPolicy Bypass -File scripts/start-local.ps1`.
It starts the API, worker, and frontend in hidden windows, uses
the existing `.local/app-v1` data, and writes logs beneath that directory. The
local profile allows up to 100 jobs per visitor/hour, 500 jobs globally/day,
100 AI questions per visitor/hour, and 500 AI questions globally/day. Each
limit can be lowered or raised up to 1,000 with the script parameters. This
does not change the normal 2/20 job and 10/100 AI defaults. The script refuses
to reuse listeners on ports 8001/5173 or an active worker lease, since those processes would
keep their current settings; it does not stop processes or clear quota records.

Open [the application](http://127.0.0.1:5173) or [API docs](http://127.0.0.1:8001/docs).
No sign-in is required. There are no prepared reports or demo endpoints: paste
a model ID or URL to create a real analysis. New models require Hugging Face network access.
`/health` checks the actual database, storage and worker and reports configured
AI availability. HTTP 503 means new analyses are unavailable; stored reports
remain inspectable when their database and storage are reachable. The app starts with your recent job or an empty workspace,
not a fixed model. These commands bind only to loopback.

Configuration uses `MODEL_ANATOMY_` environment variables. For example:

```powershell
$env:MODEL_ANATOMY_PORT = '8001'
$env:MODEL_ANATOMY_CORS_ORIGINS = '["http://localhost:5173"]'
```

Local defaults use SQLite and private local report objects under `.local/`.
Production uses Supabase PostgreSQL and a private Storage bucket. Backend-only
`NEBIUS_API_KEY` and `NEBIUS_MODEL` enable Nemotron explanations; choose a real
available NVIDIA Nemotron ID, never a placeholder. Graph facts do not depend on AI.
Do not commit secrets or model weights.

For a step-by-step manual check and expected real-model facts, use the
[local testing guide](docs/LOCAL_TESTING.md). Completed reports still describe
canonical static architecture, not executed model predictions.

```powershell
uv run --no-sync pytest -q backend/tests
uv run --no-sync ruff check backend
$env:RUN_HTTP_INTEGRATION = '1'
uv run --no-sync pytest -q -s backend/tests/test_live_http.py
```

From `frontend`, run `npm run generate:api`, `npm run build` and `npm run test:e2e`.
See [service integration](docs/SERVICE_INTEGRATION.md) for configuration, the trust
boundary and release gates, and [live backend acceptance](docs/LIVE_BACKEND_ACCEPTANCE.md)
for real HTTP-to-worker evidence. Cloud deployment remains a separate delivery.
The [CPU deployment candidate](docs/DEPLOYMENT.md), [real-job benchmarks](docs/BENCHMARKS.md)
and [submission checklist](docs/HACKATHON_SUBMISSION.md) describe setup and the
unmet hosted-release gates. No cloud credentials are needed for local inspection.

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
