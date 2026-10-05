# Decision Log

## D-001 — Reconstruction first
**2026-10-04 · accepted**
Build the architecture reconstruction engine before Model Autopsy.

## D-002 — Diagram is generated
**2026-10-04 · accepted**
The architecture diagram must come from actual artifacts.

## D-003 — Fuse artifacts
**2026-10-04 · accepted**
Use relevant evidence from config/AutoConfig, nn.Module, module hierarchy, graph tracing, source inspection and Safetensors metadata.

## D-004 — Model-specific semantic IR
**2026-10-04 · accepted**
Do not force every model into one Transformer template.

## D-005 — Evidence is first-class
**2026-10-04 · accepted**
Facts should be traceable to their source or derivation.

## D-006 — First scope: decoder-only Transformer
**2026-10-04 · accepted**
Start with one real decoder-only model.

## D-007 — Autopsy is second layer
**2026-10-04 · accepted**
Comparison, anomalies, contradictions and provenance build on the reconstructed representation.

## D-008 — No generic LLM wrapper
**2026-10-04 · accepted**
LLMs can explain established facts; they do not invent the graph.

## D-009 — One model before breadth
**2026-10-04 · accepted**
First milestone is URL → inspect → IR → graph.

## D-010 — Demo standard
**2026-10-04 · accepted**
A judge must be able to inspect a component and reach technical details and evidence.

## D-011 — Main is PR-only
**2026-10-05 · accepted**
main is the integration and release branch. Development happens on topic branches and must enter main through a pull request.
Required GitHub protection: pull request required; CI required; force pushes blocked; deletion blocked; administrators included where appropriate.