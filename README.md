# Model Anatomy

> Turn the artifacts of an open-weight Hugging Face model into a trustworthy, inspectable architecture graph.

Model Anatomy lets an engineer understand a specific checkpoint without manually joining its configuration, `nn.Module` hierarchy, source, graph traces, and Safetensors metadata. The graph is generated from evidence; it is not a hand-authored template or an LLM guess.

## Product path

```text
model ID / URL → artifact inventory → Raw IR → Semantic IR → graph → inspection
```

The first release proves that path for one decoder-only Transformer. A node must lead to its module path, tensors, dimensions, parameter ownership, and the evidence that established it. Model Autopsy is a later layer built on this representation.

## Current state

The repository contains a Python scaffold and a smoke test; the resolver and reconstruction pipeline are not implemented yet. The immediate target is `HuggingFaceTB/SmolLM2-135M`, a compact, Apache-2.0 decoder model with Safetensors support. See the [roadmap](docs/ROADMAP.md) for the dated release gates.

## Hackathon fit

For the Nebius × NVIDIA Global AI Hackathon, reconstruction remains deterministic. A clearly separated, optional Nebius Token Factory + NVIDIA model layer may explain already-established evidence in the explorer; it may never create or alter architectural facts. This makes the required model-service usage visible without weakening the product’s trust boundary.

## Documentation

- [Project ground](docs/00-GROUND.md) — promise, scope, and non-negotiable boundaries
- [Architecture](docs/ARCHITECTURE.md) — pipeline, IR, evidence, and failure model
- [Roadmap](docs/ROADMAP.md) — milestones, dates, release gates, and current status
- [Quality](docs/QUALITY.md) — PR, benchmark, demo, and submission requirements
- [Decision log](docs/DECISION_LOG.md) — binding product and architecture decisions

## Development workflow

`main` is integration-only. Create a topic branch, test the smallest useful change, open a PR, and merge only after CI and review. Full expectations are in [CONTRIBUTING.md](CONTRIBUTING.md).
