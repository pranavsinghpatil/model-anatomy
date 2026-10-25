# Roadmap and release train

**Last updated:** 2026-10-25 · **External deadline:** [2026-10-30 10:00 PDT](https://nebiusglobalaihackathon.devpost.com/rules) (22:30 IST)

## Current truth

The October 25 submission audit starts from Van's integrated `feat/vv`
commit `7ce638e`, including PR #57 and its follow-ups. The release remains
**not ready to submit**: the listed public demo returns 404, the repository is
private, and live Nebius/NVIDIA inference and the public video are unverified.
Current checks, fixes and ordered release gates are recorded in
[submission readiness](SUBMISSION_READINESS.md). Earlier counts below are
dated historical evidence, not a certification of this candidate.

The current native completion report is scoped to registered module architecture,
checkpoint binding and bounded module-call dependencies. A complete result does
not certify intermediate activation shapes, runtime branch selection, or
numerical equivalence to publisher custom code.


The October 19 compatibility increment attempts 120 distinct pinned Hugging Face
repositories, then reruns 22 affected/control cases after measured fixes. Latest
observations are ten complete static reports, 45 partial graphs, 61 artifact-only
inspections, three gated-file blocks and one invalid JSON configuration. This is
baseline plus targeted follow-up evidence, not universal support or a full rerun
on the new engine. The backend remains healthy on port 8000. See
[the matrix and verification record](HUNDRED_MODEL_BENCHMARK.md) and
[family-specific next gates](MODEL_FAMILY_STRATEGIES.md).

The real static engine is integrated from `feat/vv`, with committed regression
tests and fresh pinned-model acceptance for GPT-2, SmolLM2-135M, Qwen3-4B and ViT. Resolver,
module inspection, checkpoint metadata, and hierarchical semantic reports work
through the CLI. Static warnings and unknown shapes remain explicit; this is not
a runtime trace. See [engine acceptance](ENGINE_ACCEPTANCE.md).

Durable jobs, anonymous ownership and budgets, private-storage integration,
the public graph inspector and evidence-bound Nemotron integration are implemented
in reviewable PR increments. Local verification does not complete a hosted release
gate. Supabase setup, usable Nebius credits, live NVIDIA inference, container/HTTPS
deployment and the public video remain release requirements. No cloud credentials
are configured yet. Runtime examples have been removed: the application serves
only real submitted analyses. See [local testing](LOCAL_TESTING.md) and
[current acceptance gaps](SYSTEM_GAPS.md) before adding features or claiming a
completed hosted release.

The UI redesign is implemented on the GGUF increment: routed landing/submission,
Learn and Research workspaces, resizable inspection panels, artifacts and report
views. Local verification passes 92 deterministic browser tests and 14 separate
live cases across the primary text, existing ViT, artifact-only and selected GGUF
paths. Both themes and responsive layouts are checked. See
[the redesign verification record](UI_REDESIGN.md#verification-record). This UI
work is delivered as historical dependent PR increments targeting `b-mvp`; it does not close
the hosted or live AI release requirements above.

## Milestones

| ID | Target date | Outcome | Exit evidence |
|---|---:|---|---|
| M1 | Oct 8 | Artifact resolver | normalized reference, pinned revision, file inventory, config metadata, tests |
| M2 | Oct 12 | Raw Architecture IR | module tree, Safetensors metadata, ownership and provenance for the primary model |
| R0 `v0.1.0` | Oct 13 | Reconstruction proof | reproducible command/output for M1–M2; tagged release |
| M3 | Oct 17 | Semantic graph | repeated-block grouping and traceable edges; unsupported-path diagnostics |
| M4 | Oct 21 | Inspector demo | generated graph plus node → module/tensor/evidence drill-down |
| R1 `v0.2.0` | Oct 22 | Demo candidate | complete primary flow, recorded screen demo, Nebius/NVIDIA integration visible |
| M5 | Oct 25 | Benchmark and hardening | pinned manifest, expected facts, results and failure cases |
| R2 `v0.3.0-rc.1` | Oct 26 | Release candidate | tests, benchmark, README, deployment and demo rehearsal pass |
| R3 `v1.0.0` | Oct 28 | Submission freeze | only release-blocking fixes after this point |
| Submission | Oct 29, 18:00 IST | Submit with buffer | public repository, working URL, ≤3-minute video, Devpost fields complete |

Dates before the external deadline are internal commitments. If an exit gate fails, cut scope before moving a date: protect the one end-to-end primary-model path.

## Release gates

**R0 — reconstruction proof:** resolver and Raw IR are deterministic, tests pass, and the revision is pinned.

**R1 — demo candidate:** a user can follow the primary path without developer intervention; the Nebius Token Factory or AI Cloud + NVIDIA model interaction is demonstrable and clearly separated from reconstruction.

**R2 — release candidate:** benchmark results include successful, partial, and unsupported cases; failure behavior and resource limits are documented; the demo has a recorded fallback.

**R3 — submission freeze:** release tag, public open-source license, setup instructions, deployed/testable URL, ≤3-minute public video, architecture image, credits/licenses, and documented significant hackathon-period work.

## Operating cadence

Every work item ends with: **done, evidence, blocked, next, PR**. Update this file only when milestone state, target, or release risk changes; record permanent direction changes in the decision log. Do not let a broad Model Autopsy or multi-model feature displace an unmet release gate.
