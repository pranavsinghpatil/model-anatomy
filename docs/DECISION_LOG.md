# Decision Log

Use this for binding product or architecture decisions. Do not rewrite history; append new decisions.

## D-001 — Reconstruction first
**2026-10-04 · accepted**

Build the architecture reconstruction engine before Model Autopsy.

**Reason:** the representation is the foundation for visualization, inspection and deeper analysis.

## D-002 — Diagram is generated
**2026-10-04 · accepted**

The architecture diagram must be produced from actual model artifacts rather than manually authored templates.

## D-003 — Fuse artifacts
**2026-10-04 · accepted**

Use config/AutoConfig, instantiated `nn.Module`, module hierarchy, graph tracing where possible, source inspection, and Safetensors metadata as complementary evidence.

## D-004 — Model-specific semantic IR
**2026-10-04 · accepted**

Do not force every architecture into a single Transformer template.

## D-005 — Evidence is first-class
**2026-10-04 · accepted**

Every important fact should be traceable to direct evidence or a documented derivation.

## D-006 — First scope: decoder-only Transformer
**2026-10-04 · accepted**

Start with one real decoder-only model and expand only after reliability is demonstrated.

## D-007 — Model Autopsy is second layer
**2026-10-04 · accepted**

Reference comparison, checkpoint differences, anomalies, contradictions and localized provenance depend on the reconstructed representation.

## D-008 — No generic LLM wrapper
**2026-10-04 · accepted**

LLMs may explain established facts, but they must not invent the underlying model graph.

## D-009 — One model before breadth
**2026-10-04 · accepted**

The first engineering milestone is URL → inspect → IR → graph.

## D-010 — Demo standard
**2026-10-04 · accepted**

A judge should be able to inspect a component and reach technical details and evidence.

## D-011 — Main is PR-only
**2026-10-05 · accepted**

`main` is the integration and release branch. Development happens on topic branches and enters `main` only through pull requests.

Required repository policy:
- pull request required before merge
- CI required before merge
- force pushes blocked
- branch deletion blocked
- administrators should not bypass the rules for normal work

## D-012 — Benchmark is a release gate
**2026-10-05 · accepted**

Milestones are not complete merely because code exists. The benchmark must be reproducible and its failures documented before final release.

## D-013 — Evidence-bound Nebius/NVIDIA layer
**2026-10-05 · accepted**

To meet the hackathon platform requirement without compromising reconstruction, use Nebius Token Factory or AI Cloud with an NVIDIA open-source model only after deterministic reconstruction. The layer may explain IR facts and evidence, but may not create, label, or modify architecture facts.

## D-014 — Release train protects the primary path
**2026-10-05 · accepted**

The Oct 30, 2026 submission uses dated internal releases in `ROADMAP.md`. If a release gate fails, reduce optional scope before delaying the primary model’s reliable end-to-end path.

## D-015 - FastAPI boundary and MVP integration base
**2026-10-08 - accepted**

Use FastAPI to frame the local application around the deterministic engine.
Keep HTTP startup independent of model-library imports and report unavailable
reconstruction honestly. Develop on topic branches from `b-mvp` and open PRs
targeting `b-mvp`; promote to `main` only through a separate reviewed PR.

The initial HTTP-only foundation has been superseded by D-016 and D-017.

## D-016 - Integrate the working engine and preserve static evidence
**2026-10-08 - accepted**

Use the working `feat/vv` engine as the application's core and retain the CLI
as a diagnostic entry point. The CLI and HTTP worker share extraction logic and
canonical IR; the HTTP service must not execute CLI commands or parse terminal
output. Fix extraction gaps during application integration and preserve partial
results rather than inventing missing architecture.

Use `pranav/` topic branches exclusively and raise PRs into `b-mvp`. Public
visitors need no sign-in; anonymous ownership and usage limits belong to the
service. Supabase PostgreSQL/private storage and evidence-bound Nemotron on
Nebius are the accepted hosted direction.

## D-017 - Durable anonymous application and bounded explanations
**2026-10-08 - accepted**

Persist accepted analyses before returning 202. Use one fenced worker, ten waiting
jobs, a 120-second deadline and a 4 GiB production memory budget. Visitor budgets
are hourly; hard global spend caps are daily. Exact revisions and engine/schema
versions define reusable cache identity. Keep visitor jobs for 24 hours and
cache reports for seven days; prepared examples survive provider outages.

NVIDIA Nemotron explanations use selected stored report facts and validated
citations. AI cannot create or change graph facts. Cookie ownership is not
sign-in or an anti-Sybil guarantee. Do not claim production readiness before
real cloud configuration, container checks, live inference and deployment evidence.

## D-018 - Live analyses first and one operational health contract
**2026-10-09 - accepted**

New analyses must inspect the submitted Hugging Face repository's actual pinned
artifacts. Prepared snapshots are explicit optional examples, never an automatic
default result or substitute for failed inspections. The browser begins with a
visitor's recent job or an empty analysis workspace. Production has no mock
response fallback; test-only mocks remain useful for deterministic failure tests.

Consolidate liveness, readiness and configured availability into `GET /health`.
It reports real database/storage/worker checks and returns 503 when inspection
is unavailable. Optional AI configuration is separate from core readiness. Remove
the former three operational endpoints instead of keeping redundant aliases.
Allow exact same-origin Swagger mutations while retaining host validation and
the explicit foreign-origin allowlist. Verify the HTTP-to-worker path against a
real nonprepared model with prepared reports absent, not only saved snapshots.

## D-019 - Expand through explicit family and scale acceptance gates
**2026-10-09 - accepted**

Broaden Model Anatomy step by step across model families, preserving the working
text inspection path. Track family semantics, checkpoint ownership, input/output
shapes, runtime budgets and cloud readiness separately. A successful native meta
construction or large text checkpoint is not proof of image/video support. Unknown
components remain visible, and unsupported artifacts never become an invented
graph. `SYSTEM_GAPS.md` records the outstanding gates, not a universal support claim.

Make the graph workspace readable before adding further UI surface: default to an
inspection scale with explicit whole-graph framing, collapsible history, and an
inspector that opens on selection. Keep the canonical graph unchanged by layout
controls, and verify desktop/mobile inspection and export workflows.
