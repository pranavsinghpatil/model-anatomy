# AGENTS.md

## Mission
Build Model Anatomy as a serious engineering project, not a collection of disconnected demo features.

> Given an open-weight Hugging Face model, reconstruct a trustworthy, semantic, tensor-aware representation of the actual model artifacts, render an inspectable architecture graph, and later use that representation for deeper model analysis.

## Team rules
1. Ground and decision log are the source of truth.
2. Never commit directly to main. All work must be raised as a pull request.
3. Work in small, reviewable increments.
4. Evidence over assumptions.
5. No silent scope creep.
6. No black-box LLM behavior as the core reconstruction mechanism.
7. One real model end-to-end before broad support.
8. Tests accompany core logic.
9. Unsupported/custom models fail honestly.
10. Keep docs current.
11. Every update states: done, evidence, blocked, next.

## Current build order
HF model → artifact resolver → config / AutoConfig → instantiated nn.Module → module tree + graph tracing where possible → Safetensors metadata → Raw Architecture IR → Semantic Architecture IR → interactive graph → tensor / parameter inspection → Model Autopsy

## Definition of done
- implementation
- tests
- docs/status updated
- limitations understood
- clear commit