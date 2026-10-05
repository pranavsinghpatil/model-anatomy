# Release Plan

## R0 — Internal engineering
Prove reconstruction on one model. Resolver, Raw IR, tests.

## R1 — Technical demo
Generated architecture, Semantic IR, graph UI, tensor inspection, evidence.

## R2 — Benchmark release
Pinned benchmark manifest, reproducible run, results and failures documented.

## R3 — Submission release
Final demo, backup path, benchmark evidence, architecture diagram, README, limitations, credits/licenses, release tag, clean main.

## Freeze rule
Once R3 begins: no opportunistic features; only release-blocking fixes; every change through PR; rerun benchmark after core changes.