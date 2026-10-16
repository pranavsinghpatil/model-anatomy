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

## D-020 - Close canonical ViT semantics with bounded source shape algebra
**2026-10-09 - accepted**

Capture patch flatten/transpose, CLS expansion/concatenation and token selection
as explicit source operations for native ViT. Derive fixed-resolution dimensions
from native configuration and projection metadata, retaining unknowns for
unsupported transformations. Classify ViT output projections from native types
and affine/dropout topology; source analysis alone establishes residual wiring.
Expose standalone CLS/position parameter ownership through component inspection.
Keep this refinement scoped to ViT until other families have acceptance evidence.
Increment the engine cache identity without changing IR schema. Do not equate
complete native-module classification with runtime-verified computation, image
preprocessing, universal vision support or production deployment readiness.

## D-021 - Completion follows established report facts, not runtime execution
**2026-10-09 - accepted**

A canonical static report can complete without executing weights when all scoped
facts and shapes are established and no extraction warnings remain. General
method limitations remain visible but are not missing model facts. Preserve known
Q/K/V tensor specs on named graph output ports rather than introducing artificial
unknowns. Do not relax coverage thresholds or relabel genuinely partial reports.
Old saved reports remain immutable; extraction corrections invalidate cache
identity and require a new analysis.

## D-022 - Separate report problems, method scope and artifact identity
**2026-10-09 - accepted**

Analysis status, model revision and engine version are independent facts. Expose
recorded coverage and specific incomplete checks rather than implying that a
completed badge proves every runtime variant. Extraction warnings remain warnings;
general method limitations belong to analysis scope. Missing coverage/count checks remain
unknown, not default to success. Normalize legacy presentation from immutable
report bytes and reassess terminal completion from those same facts, without
rewriting stored jobs, reports or ownership. Keep summaries
small and bounded. Pause expansion while current report and system gaps are being
closed; document unresolved engine warnings instead of declaring the system perfect.

## D-023 - Serve submitted analyses only, with no prepared examples
**2026-10-09 - accepted**

Remove the examples endpoint, prepared snapshot resolver/settings, public sample
IDs, prepared response types/flags, and example UI/client paths. Every analysis,
graph, component, export and question uses a persisted job and visitor ownership.
New visitors start empty; failed or unavailable submissions never substitute a
saved sample. Exact-revision caching remains because its content comes from real
inspections. This supersedes earlier decisions retaining prepared reports.
Keep test-only fixtures and mocks for deterministic safety/failure regression
tests, plus separate real HTTP/worker acceptance; neither becomes runtime data.
Preserve historical CLI reports as evidence, not served product content. Pause
new features until this removal is verified and the user can test it locally.

## D-024 - Close current correctness and bounded-runtime gaps before extras
**2026-10-12 - accepted**

Resume development against the documented first-release agenda, not additional
features or universal model support. Resolve known native tracing predicates by
callable identity under the canonical static profile, retaining source hashes
and assumptions; never suppress arbitrary opaque functions by name. Unknown
conditions with returning branches must retain alternatives and report uncertainty.
Derive report applicability from recorded input facts, not architecture-name
heuristics. Explicit nested configuration, tensor/native record and graph budgets
supplement, but do not replace, worker isolation and Linux resource enforcement.
Invalidate extraction cache identity without rewriting saved reports.

The user does not yet have cloud credentials. Local validation must remain usable
without them. Deployment preparation and a real visitor-owned benchmark can be
reviewed now; Supabase, Nebius/NVIDIA inference, HTTPS and hosted load/recovery
verification remain explicit release blockers until setup and real checks occur.

## D-025 - Verified artifact inspection is distinct from architecture reconstruction
**2026-10-12 - accepted by user**

The user approved showing verified artifact inspection when architecture support
is unavailable, with no invented graph. Add a terminal `inspected` outcome,
separate from completed/partial architecture reports. Pin the repository revision,
list its actual files, and read bounded Safetensors headers only when checkpoint
selection is unambiguous. Do not execute custom model code, fetch weights, infer
dataflow from tensor names, or equate tensor elements with unique model parameters.
Missing format/configuration and unsupported native classes may produce this
explicit result; malformed configuration, corrupt headers and transport failures
remain failures. Graphs, diagram exports and architecture questions are unavailable
for these reports; verified artifacts and non-diagram exports remain accessible
only to the owning visitor. Keep the CLI strict and invalidate engine cache identity.

Use an explicit local launcher profile with bounded higher allowances for manual
testing. Do not clear rate records, replace visitor identity, or weaken deployment
defaults. Keep support claims tied to fresh pinned model evidence.

## D-026 - Inspect GGUF metadata without claiming architecture reconstruction
**2026-10-12 - implementation increment**

The reported Bonsai repository uses GGUF rather than Safetensors and has separate
language-model variants and vision projectors. Support explicit pinned GGUF file
URLs and file selection from the inspected inventory. Never choose a quantization
variant arbitrarily or merge a projector into a language-model checkpoint.

Read bounded GGUF metadata and tensor descriptors through exact HTTP ranges;
do not load weights or execute publisher code. Use upstream GGUF type definitions,
retain unknown quantization IDs as numeric facts, and label GGML dimension order
explicitly. Header facts do not prove payload integrity or computation dataflow.
Keep the CLI architecture path strict. Return the existing graph-free `inspected`
outcome, and invalidate cache identity without rewriting previous analyses.
Full Bonsai hybrid-attention reconstruction and broader modality semantics remain
separate acceptance gates; this decision does not claim universal support.

## D-027 - Route by artifact layout and upstream native mappings
**2026-10-12 - compatibility increment**

The user requests broader categories and investigation of real failures. Keep
publisher task/library declarations as hints, not architecture facts. Discover
native wrappers from upstream mapping names; unavailable declared heads must not
silently become another backbone. Pipeline manifests require component selection
with sibling configs and directory-relative indexes. Preserve native blockers and
non-PyTorch Safetensors as artifact evidence. Keep remote code and weights disabled.

Preserve staged dimensions instead of failing valid API reports. Establish
VideoMAE temporal input order from native source; leave other unverified layouts
unknown. Add pinned cross-category and diffusion component checks. Partial reports
record remaining semantic work, not full modality support. See
`DYNAMIC_ROUTING_ACCEPTANCE.md`.

## D-028 - Organize inspection around orientation, architecture, artifacts and evidence
**2026-10-12 - accepted by user; implemented 2026-10-13**

Introduce a routed technical workspace with Learn and Research modes using the
same real report. Keep submission on Start, explicit visitor-owned continuation,
theme preferences, and session graph/selection/viewport state. Separate Overview,
Architecture, Artifacts and Report to contain dense details. Use free Radix
primitives, resizable panels and local fonts. Learn definitions are authored
general guidance, never model-specific inferred facts. Artifact-only results
retain graph-free behavior and supported exports. API and IR schemas do not
change. See `UI_REDESIGN.md` for responsive layouts and domain boundaries.

## D-030 - Make the inspection canvas primary and add a bounded explanatory assembly
**2026-10-14 - usability revision**

User feedback supersedes the separate landing/start direction in D-028. Both
entry routes now accept a model immediately. Both modes open the same architecture;
artifact-only reports still open Artifacts. Compact technical nodes, visible fit
and readable controls, scoped search and separate disconnected inventory keep the
graph navigable without supplying missing relationships.

Offer a separate Anatomy presentation of the current shallow graph. Use original
isometric SVG geometry, with mechanism choice based only on recorded component
kind. Preserve selection, inspector tabs, child-graph loading and per-graph view
state across presentations. Bound illustrated nodes and repetition samples;
unknown kinds stay sealed. Separation illustrates construction, never inference.
Reported dataflow alone supplies assembly links. Mechanical dimensions, fixed
inserts and materials are explanatory artwork rather than model facts.

Keep exact shapes, provenance and other relationship types in the technical graph
and shared inspector. No backend/schema changes, new paid dependencies, third-party
drawing service or model data upload are introduced. See
`CANVAS_WORKSPACE.md` and `ANATOMY_ILLUSTRATION.md` for validation evidence.

## D-031 - Use the working product to carry the frontend design
**2026-10-16 - frontend usability revision**

Apply the user's Backdoor, Fanout and ClickUp references to information hierarchy:
a strong introduction beside the actual model form, horizontal named workspace
views, reserved graph controls and readable technical details. Preserve direct
entry, the same inspected reports, graph state and explicit unavailable outcomes.
Avoid marketing proof without evidence and effects that compete with inspection.
Retain existing fonts, Radix controls and the teal accent; add no UI dependencies.
See `FRONTEND_ART_DIRECTION.md` for the skill direction, adaptations, critique and
validation limits. This decision changes no backend, ownership or IR contract.

## D-032 - Make the white canvas and contextual detail primary
**2026-10-16 - user-directed layout replacement**

Supersede the tinted split entry and stacked workspace bands in D-031. Use white
as the default appearance, retaining explicit theme preferences. Place the actual
entry form in one centered column. Separate global identity/view navigation from
one local graph bar and a reserved bottom dock. Move dimensions and advanced
graph settings behind explicit disclosures, and start the component navigator
closed for new analyses. Retain its accessible list and all graph/inspection
state. Research collapses type guidance; Learn expands it. These are information
hierarchy changes, not backend or evidence changes. See `WHITE_WORKSPACE.md`.
