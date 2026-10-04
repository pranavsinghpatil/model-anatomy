# AGENTS.md

## Mission
Build Model Anatomy as a serious engineering project, not a collection of disconnected demo features.

> Given an open-weight Hugging Face model, reconstruct a trustworthy, semantic, tensor-aware representation of the actual model artifacts, render an inspectable architecture graph, and later use that representation for deeper model analysis.

## Team rules
1. Ground and decision log are the source of truth.
2. Work in small, reviewable increments.
3. Evidence over assumptions.
4. No silent scope creep.
5. No black-box LLM behavior as the core reconstruction mechanism.
6. One real model end-to-end before broad support.
7. Tests accompany core logic.
8. Unsupported/custom models fail honestly.
9. Keep docs current.
10. Every update states: done, evidence, blocked, next.

## Current build order
HF model
→ artifact resolver
→ config / AutoConfig
→ instantiated nn.Module
→ module tree + graph tracing where possible
→ Safetensors metadata
→ Raw Architecture IR
→ Semantic Architecture IR
→ interactive graph
→ tensor / parameter inspection
→ Model Autopsy

## Current scope
- First target: decoder-only Transformer.
- First milestone: HF URL → trustworthy architecture graph.
- Model Autopsy is a later analysis layer.

## Engineering
- Python 3.11+
- Type hints for core interfaces
- Small modules
- Deterministic extraction where possible
- Stable IR
- Explicit evidence/provenance
- Avoid unnecessary dependencies
- Prefer metadata inspection before full weight loading
- Never commit weights or secrets

## Definition of done
- implementation
- tests
- docs/status updated
- limitations understood
- clear commit
