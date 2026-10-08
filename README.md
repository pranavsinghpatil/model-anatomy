# Model Anatomy

> Turn the artifacts of an open-weight Hugging Face model into a trustworthy, inspectable architecture graph.

Model Anatomy lets an engineer understand a specific checkpoint without manually joining its configuration, `nn.Module` hierarchy, source, graph traces, and Safetensors metadata. The graph is generated from evidence; it is not a hand-authored template or an LLM guess.

## Product path

```text
model ID / URL → artifact inventory → Raw IR → Semantic IR → graph → inspection
```

The first release proves that path for one decoder-only Transformer. A node must lead to its module path, tensors, dimensions, parameter ownership, and the evidence that established it. Model Autopsy is a later layer built on this representation.

## Current state

The repository contains a Python reconstruction scaffold and a tested FastAPI service foundation. The resolver and reconstruction pipeline are not implemented yet on `b-mvp`. The API reports that limitation explicitly. The immediate target is `HuggingFaceTB/SmolLM2-135M`, a compact, Apache-2.0 decoder model with Safetensors support. See the [roadmap](docs/ROADMAP.md) for the dated release gates.

## Local API

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[api,dev]'
model-anatomy-api
```

Open <http://127.0.0.1:8000/docs>. `GET /healthz` checks the HTTP process;
`GET /readyz` returns 503 until reconstruction is integrated.
`GET /api/v1/capabilities` reports the current supported functionality.
The launcher binds to `127.0.0.1`; it does not expose a public service.

Configuration uses `MODEL_ANATOMY_` environment variables. For example:

```powershell
$env:MODEL_ANATOMY_PORT = '8001'
$env:MODEL_ANATOMY_CORS_ORIGINS = '["http://localhost:5173"]'
```

Run `python -m pytest -q` and
`ruff check backend/model_anatomy/api backend/tests/test_api.py` before opening a PR.
See [service integration](docs/SERVICE_INTEGRATION.md) for the proposed production
path, resource controls, and remaining implementation gates.

## Hackathon fit

For the Nebius × NVIDIA Global AI Hackathon, reconstruction remains deterministic. A clearly separated, optional Nebius Token Factory + NVIDIA model layer may explain already-established evidence in the explorer; it may never create or alter architectural facts. This makes the required model-service usage visible without weakening the product’s trust boundary.

## Documentation

- [Project ground](docs/00-GROUND.md) — promise, scope, and non-negotiable boundaries
- [Architecture](docs/ARCHITECTURE.md) — pipeline, IR, evidence, and failure model
- [Roadmap](docs/ROADMAP.md) — milestones, dates, release gates, and current status
- [Quality](docs/QUALITY.md) — PR, benchmark, demo, and submission requirements
- [Decision log](docs/DECISION_LOG.md) — binding product and architecture decisions

## Development workflow

The active MVP base is `b-mvp`. Create a topic branch from it, test the smallest useful change, and open a PR targeting `b-mvp`. Promotion to `main` requires a separate reviewed PR. Full expectations are in [CONTRIBUTING.md](CONTRIBUTING.md).
