# AGENTS.md

## Mission
Build Model Anatomy as a serious engineering project, not a collection of disconnected demo features.

> Given an open-weight Hugging Face model, reconstruct a trustworthy, semantic, tensor-aware representation of the actual model artifacts, render an inspectable architecture graph, and later use that representation for deeper model analysis.

## Working rules
1. `docs/00-GROUND.md` and `docs/DECISION_LOG.md` are the source of truth.
2. **Never commit directly to `main`. All work reaches `main` through a pull request.**
3. Work in small, reviewable increments.
4. Evidence over assumptions.
5. No silent scope creep.
6. No black-box LLM behavior as the core reconstruction mechanism.
7. One real model end-to-end before broad support.
8. Tests accompany core logic.
9. Unsupported/custom models fail honestly.
10. Keep documentation aligned with implementation.
11. Every work update states: done, evidence, blocked, next.
12. Start substantial changes on a new topic branch and open a PR. Bundle related minor fixes into that increment instead of opening a PR for every small correction.

## Current build order
HF model → artifact resolver → config / AutoConfig → instantiated nn.Module → module tree + graph tracing where possible → Safetensors metadata → Raw Architecture IR → Semantic Architecture IR → interactive graph → tensor / parameter inspection → Model Autopsy

## Current scope
- First target: decoder-only Transformer.
- First milestone: HF URL → trustworthy architecture graph.
- Model Autopsy is a second-layer capability.

## Engineering standards
- Python 3.11+
- Type hints for core interfaces
- Small modules with clear responsibility
- Deterministic extraction where possible
- Stable IR schemas
- Explicit evidence/provenance
- Prefer metadata inspection before full weight loading
- Never commit model weights, credentials, or secrets

## Definition of done
A change is done only when:
- implementation exists
- meaningful tests exist
- docs/status are updated when relevant
- limitations are understood
- the change is in a PR with clear evidence
