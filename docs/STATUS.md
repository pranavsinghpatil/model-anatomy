# Project Status

**Last updated:** 2026-10-04

## Current phase
Phase 1 — Reconstruction foundation

## Ground
Model Anatomy reconstructs an open-weight Hugging Face model into a semantic, tensor-aware architecture representation and interactive graph.

## Current target
HF model ID / URL
→ resolve artifacts
→ load/inspect model
→ inspect module hierarchy
→ inspect Safetensors metadata
→ create Raw Architecture IR

## Done
- Product ground established
- Repository scaffold established
- Operating documentation established

## In progress
- Resolver design
- First model selection
- Raw IR schema

## Not started
- robust graph tracing
- Semantic IR
- frontend graph
- tensor explorer
- Model Autopsy

## Next milestone
**M1: URL → Raw Architecture IR for one real decoder-only Transformer.**

## Milestone success
1. resolve relevant artifacts
2. safely inspect/load model
3. enumerate module hierarchy
4. inspect tensor metadata
5. emit deterministic Raw IR
