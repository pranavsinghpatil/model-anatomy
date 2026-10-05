# Project Tracker

| ID | Milestone | Status | Exit condition |
|---|---|---|---|
| M0 | Ground + repository | ✅ | Base repo and operating docs exist |
| M1 | Model resolver | ⬜ | Model ID resolves relevant artifacts |
| M2 | Raw Architecture IR | ⬜ | One model represented deterministically |
| M3 | Graph reconstruction | ⬜ | Traceable graph/edges extracted |
| M4 | Semantic IR | ⬜ | Human-readable architecture semantics |
| M5 | Interactive explorer | ⬜ | Node → details → evidence works |
| M6 | Tensor explorer | ⬜ | Module ↔ tensor inspection works |
| M7 | Model Autopsy v0 | ⬜ | One real forensic finding is reproducible |
| M8 | Benchmark + demo hardening | ⬜ | Reproducible benchmark + judge-ready demo |
| M9 | Submission release | ⬜ | Frozen release package and submission evidence |

## Current focus
M1 — Model resolver

## Integration rule
All implementation and documentation changes enter main through pull requests.

## Work streams
Core: resolver → extraction → IR → reconstruction
Frontend: graph → node inspection → evidence
Research: coverage → traceability → forensics
Demo: one reliable end-to-end path