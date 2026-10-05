# Project Status

**Last updated:** 2026-10-05

## Current phase
Phase 1 — Reconstruction foundation

## Repository control
main is the protected integration branch. Work is PR-only.

## Current target
HF model ID / URL → resolve artifacts → load/inspect model → inspect module hierarchy → inspect Safetensors metadata → create Raw Architecture IR

## Done
- product ground
- repository scaffold
- operating documentation
- benchmark/release tracking structure

## In progress
- resolver design
- first model selection
- Raw IR schema
- benchmark harness definition
- CI / branch policy

## Next milestone
M1: URL → Raw Architecture IR for one real decoder-only Transformer.

## Milestone success
1. resolve relevant artifacts
2. safely inspect/load model
3. enumerate module hierarchy
4. inspect tensor metadata
5. emit deterministic Raw IR

## Update rule
Update this file whenever milestone status changes.