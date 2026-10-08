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

The worker, persistence, and frontend choices in `SERVICE_INTEGRATION.md` are
recommendations for subsequent increments, not implemented capabilities.
