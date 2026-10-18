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

## D-033 - Use compact controls and purposeful state motion
**2026-10-17 - user-directed density refinement**

Reduce desktop entry typography, control heights and section gaps at normal
browser zoom. Keep touch targets at 44px and preserve technical values. Respond
to canvas container width so graph controls remain usable beside resized panels;
share phone toolbar rows instead of reducing touch targets. Align inspector port
facts in rows. Use transform/opacity feedback for mode selection, copy/theme
icons, menus and panels, honoring reduced motion. Fanout and Smooth UI inform
the presentation; reconstruction and API contracts remain unchanged. See the
compact-scale section of `WHITE_WORKSPACE.md` for evidence and limitations.

## D-034 - Use sourced free components and restrained material separation
**2026-10-17 - user-directed component refinement**

Adapt Magic UI DotPattern and SmoothUI Button Copy / Fluid Hover Highlight from
pinned MIT sources. Keep the compact workspace, normal browser zoom, solid reading
surfaces and explicit evidence. Use a static pattern outside technical content,
subtle panel depth, labeled graph category cues and honest clipboard feedback.
Load Motion's DOM features separately and honor reduced motion. No Pro components,
decorative model facts, backend changes or new routes are introduced. The live
browser audit resumed after a connection-error/security blocker cleared; six
captured steps and deterministic regression checks are separate evidence.
See `INTERFACE_COMPONENTS.md`.

## D-035 - Use one inspection flow at 75% interface scale
**2026-10-17 - explicit user request**

Remove Learn/Research controls, mode persistence, conditional copy and the global
learning footer. Retain authored component definitions behind the inspector's
existing disclosure. Apply a persistent 0.75 root page scale and compensate the
workspace height so it fills the viewport. Keep graph zoom independent of page
scale and retain theme, graph path, selection, tabs and panel state. This supersedes
the former two-mode presentation and normal-scale density preference; no backend
or report contract changes are required.

## D-036 - Elevated professional engineering dashboard aesthetic
**2026-10-17 - explicit user request**

Refine the UI/UX across the landing page, unified workspace, graph canvas dock,
component navigator, and inspector to achieve a calm, restrained, high-density
engineering dashboard aesthetic inspired by TextBackdoor, Fanout, and ClickUp.
Avoid generic AI aesthetic cliches (neon glows, gradient borders, rainbow badges).
Implement a 3-pillar engineering Bento Grid on the landing page, clean radial
ambient lighting, tree guide lines in the component navigator, tactile button
micro-interactions, tabular monospace typography for tensor dimensions and counts,
and acrylic floating control bars. Retain all existing contracts, accessibility hooks,
and Playwright test bounds. Development conducted on topic branch `pranav/frontend-revamp`.

## D-037 - Cosmic obsidian & electric indigo research studio theme with illuminated dual-grid canvas and shiny components
**2026-10-17 - explicit user request**

Replace the muted slate palette with a modern, rich, shiny research engineering palette:
deep cosmic obsidian canvas (`#08090d`), midnight research plates (`#0e121d`), electric indigo
primary core (`#6366f1` / `#818cf8`), and luminous category semantics (sky cyan `#38bdf8`,
violet `#818cf8`, golden amber `#fbbf24`, mint emerald `#34d399`).
Add an atmospheric radial ambient aura and a dual-layer technical grid (major grid lines +
subtle dots) to the architecture graph canvas and landing page backdrop. Upgrade buttons,
omnibox, bento cards, and graph nodes with specular top highlights (`inset 0 1px 0 rgba(255,255,255,0.15-0.35)`),
rich subtle gradients, glowing category indicator stripes, metallic repeat badges, and
radiant selected states. Retain all existing contracts, test bounds, and accessibility attributes.

## D-038 - Architectural blueprint & porcelain light theme optimization
**2026-10-17 - explicit user request**

Overhaul the light theme from a washed-out, flat white-on-white presentation into a crisp, authoritative research engineering aesthetic:
- **Architectural Paper Backdrop & Blueprint Canvas**: Shift the global background to a clean architectural paper tone (`#f1f4f9`) and the graph canvas to an architectural blueprint tone (`#eef2f8`) with dual-layer grid lines (`--graph-grid`: `rgba(15,23,42,0.09)`, `--graph-grid-major`: `rgba(79,70,229,0.08)`) and ambient radial focal lighting.
- **Porcelain White Surfaces with Specular Depth**: Render cards, workbench containers, inspector panels, and graph nodes in brilliant porcelain white (`#ffffff`) with subtle top highlights (`inset 0 1px 0 #ffffff`), tactile drop shadows (`0 1px 3px rgba(15,23,42,0.04)`, `0 16px 36px -4px rgba(15,23,42,0.08)`), and sharp borders (`#d1d9e6` / `#cbd5e1`).
- **Eliminate Repeated Node Dark Mud Bug**: Replace the hardcoded dark mode gradient on `.model-node.is-repeated::before/after` with stacked porcelain white cards (`linear-gradient(180deg, #ffffff 0%, #f4f7fc 100%)`), amber border accents, and a warm gold repeat badge (`#fef3c7` / `#92400e`), completely eliminating the dark muddy patch on `Decoder Block`.
- **Specular Action Controls**: Style light mode primary buttons with royal electric indigo gradients (`#4f46e5` to `#4338ca`), top bevel highlights, and authoritative hover elevations. Secondary buttons use porcelain bevels with crisp indigo hover borders.
- **Deep Obsidian Ink Typography**: Elevate light theme text to deep obsidian ink (`#090d16`) and secondary labels to slate blue (`#5b677a`), ensuring razor-sharp contrast across all surfaces.
- **Preserved Contracts**: Dark theme remains untouched; all Playwright tests, accessibility labels, and backend/IR contracts remain 100% intact.

## D-039 - Default-on workspace sidebars, inspector executive overview, and enriched light theme studio aesthetics
**2026-10-17 - explicit user request**

Resolve UX disconnect and elevate the light theme into a modern, tactile research engineering studio:
- **Default-On Workspace Sidebars**: Update `useWorkspaceSession` and `ArchitectureWorkspace` breakpoint so both the Component Navigator (left) and Inspector (right) default to **open** on desktop and laptop screens (`>= 1024px`), eliminating the empty floating void and presenting an immediate, dense, professional IDE experience.
- **Inspector Executive Model Architecture Overview**: Replace the bare empty state in `Inspector` with a rich overview card displaying the Model ID, Architecture pill, Parameter Count, Status badge, Tensor count, Revision hash, and an interactive component selection guide.
- **Luminous Node & Canvas Aesthetics**:
  - Add multi-stage specular drop shadows and subtle tone-aware ambient illumination to `.model-node`.
  - Refine `.anatomy-node-icon` with jewel-toned container badges and crisp drop shadows.
  - Upgrade `.model-node.is-repeated` into an architectural 3D card stack with warm amber borders (`#f59e0b`), golden amber top border accent, and metallic gold repetition pill.
  - Laser-contour input and output boundary nodes with cyan and emerald dashed framing.
  - Introduce Segmented Control pill tabs in the Inspector and header dock.
  - Add frosted glass acrylic styling with hairline borders and top specular bevels.
- **Landing Page Polish**: Add an announcement studio badge (`Tensor & Architecture Inspection Studio`), refine submission card elevation, and sharpen typography.
- **Full Verification**: 100% passing Playwright test suite (10/10) across desktop and mobile suites.

## D-040 - Smooth, modern, rich, minimal, and professional light theme overhaul
**2026-10-17 - explicit user request**

Eliminate dated, cluttered graph paper visual patterns and transition to an ultra-clean, minimal, and rich porcelain studio design:
- **Clean Porcelain Canvas**: Removed noisy math-notebook grid lines (`--graph-grid-major: transparent`) in light theme. Retained ultra-fine, quiet micro-dots (`--graph-grid: rgba(148, 163, 184, 0.24)`) on a warm porcelain surface (`#fafbfc`) with a soft top radial spotlight.
- **Node Card Redesign**:
  - Eliminated the dated vertical left bar strip on `.model-node`.
  - Upgraded cards with modern `14px` border-radius, clean slate borders (`rgba(226, 232, 240, 0.95)`), and silky multi-stage drop shadows.
  - Re-styled category labels (`.node-kind`) as crisp, uppercase pill micro-badges with tinted jewel icon containers.
  - Deep obsidian slate typography (`#0f172a`) with tight letter-spacing.
- **Solid Modern Boundary Cards**: Replaced wireframe dashed borders on input and output nodes with sleek solid pill borders and subtle ice-blue (`#bfdbfe`) and mint (`#bbf7d0`) gradients.
- **3D Layered Repeat Deck**: Polished repeated decoder blocks with an isometric card stack appearance, warm amber accents, and an integrated `× 30 layers` pill badge.
- **Fluid Edges & Framed Mini-Map**: Refined SVG connector paths with smooth hover feedback and glowing selection; framed the mini-map in an elevated porcelain card.
- **Bento Overview Card**: Enriched the executive scorecard with stat cards, hover elevation, and interactive discovery banner.
- **Verification**: 10/10 Playwright tests passing cleanly across desktop and mobile suites.

## D-041 - Aceternity spotlight backdrop, Bento scorecard with pulsing status, and specular porcelain component architecture
**2026-10-17 - explicit user request**

Incorporate design inspirations from the user-provided design repository (Aceternity UI, BentoGrids, 21st.dev) to completely transform the light theme into a smooth, modern, rich, and minimal research engineering studio:
- **Aceternity Spotlight Canvas & Backdrop**:
  - Replaced flat white with a warm porcelain slate canvas (`#f8fafc`) illuminated by an Aceternity-inspired ambient spotlight beam (`radial-gradient(ellipse 85% 55% at 50% -8%, rgba(99, 102, 241, 0.1) 0%, rgba(56, 189, 248, 0.04) 40%, transparent 75%)`).
  - Layered ultra-fine architectural grid lines (88px spacing at `rgba(15, 23, 42, 0.035)`) with precision micro-dots (22px spacing at `rgba(99, 102, 241, 0.12)`) for depth without visual noise.
- **Specular Porcelain Node Cards**:
  - Eliminated dated vertical left strips in favor of top specular accent indicators (`border-top: 2.5px solid var(--node-tone)`), mirroring the clean, modern component design of Linear and Raycast.
  - Re-styled category headers (`.node-kind`) as crisp uppercase pill micro-badges with jewel icon backgrounds.
  - Enhanced repeated decoder blocks (`.is-repeated`) with an isometric 3D stacked-card depth, warm golden amber borders (`#f59e0b`/`#d97706`), and a metallic gold repeat counter pill.
  - Replaced wireframe dashed outlines on Input and Output boundary nodes with solid, polished pill borders (`#7dd3fc` and `#6ee7b7`) and delicate gradient tints.
  - Refined SVG connection handles with interactive scale (`1.15`) and radiant indigo hover glow.
- **BentoGrids Executive Scorecard with Pulsing Status**:
  - Transformed the Model Architecture Overview in `Inspector.tsx` into a modern Bento grid scorecard featuring dedicated micro-icons (`Cpu`, `Activity`, `Layers`, `GitCommit`).
  - Added a live pulsing emerald status indicator dot (`.status-indicator-dot` with `@keyframes pulse-dot`).
  - Added an interactive discovery card with subtle gradient background and search icon.
- **Component Navigator Smooth Micro-Interactions**:
  - Added smooth horizontal drift (`transform: translateX(2.5px)`) on item hover and an active indicator gradient for selected components.
- **Verification**:
  - Full Vite production build succeeded with zero errors (2,332 modules).
  - 100% of the Playwright e2e test suite passed (10/10 tests across desktop and mobile).
  - High-resolution visual screenshots captured in light mode verifying studio canvas, selected node tensor details, and the landing page.

## D-042 - Interactive architecture blueprint, animated border beam, bento preview widgets, and model catalog
**2026-10-18 - explicit user request**

Elevate the UI and UX of Model Anatomy from minimal to a rich, immersive, and professional research studio:
- **Interactive Architecture Blueprint Teaser (`<ArchitectureTeaser />`)**:
  - Embedded an interactive blueprint teaser directly on the landing page with tabbed architecture switching (`SmolLM2 · 135M`, `GPT-2 · 124M`, `BERT · Base`).
  - Connected sequential node flow (`Token IDs` → `Token Embedding` → `Decoder Block ×30` → `Final RMSNorm` → `LM Head` → `Output Logits`) with interactive click selection.
  - Live inspector card displays authentic tensor dimensions (`[batch_size, seq_len, 576]`), parameter counts (`85,054,464`), model share (`63.2%`), precision dtype (`bfloat16`), and a 1-click "Inspect this model" CTA.
- **Magic UI Animated Border Beam (`.border-beam`)**:
  - Implemented an animated rotating gradient border beam around the primary model input card inspired by Magic UI / Aceternity.
  - Added `@media (prefers-reduced-motion: reduce)` fallback to cleanly pause and hide the animation for accessibility and test stability.
- **4-Metric System Capability Ribbon**:
  - Added a tactile 4-stat ribbon below the hero: `0 MB Weights Downloaded`, `< 85ms SafeTensors Parse`, `100% Deterministic Evidence`, `134M - 70B+ Scale Range`.
- **Expanded Bento Grid with Interactive Visual Previews**:
  - 4 capability cards featuring live UI widgets:
    1. *Semantic Graph Reconstruction*: visual PyTorch dataflow preview with Self-Attention and SwiGLU MLP blocks.
    2. *SafeTensors Header Streaming*: live header stream active badge with hex offset ticker (`offset: 0x0041a0 · BF16 · 82.4 KB`).
    3. *Executive Scorecard*: multi-colored parameter allocation split bar (Attention 38%, MLP 44%, Embed 18%).
    4. *Deterministic Evidence*: verified commit SHA `93efa2f0` and SHA-256 audit badge.
- **Supported Architecture Catalog**:
  - 6 model cards (`SmolLM2-135M`, `GPT-2`, `BERT-base`, `Llama-3.2-1B`, `Mistral-7B`, `Qwen-2.5-0.5B`) with family tags, parameter pills, layer/head/context specs, and direct "Load into inspector" action buttons.
- **Graph Node & Inspector Polish**:
  - Added `.node-header-row` with `.node-mini-param` chips to `.model-node` in `AnatomyNode.tsx` without breaking `.node-kind` text matchers.
  - Added executive parameter allocation bar in `Inspector.tsx` overview and verified SafeTensors percentage share banner in selected view.
- **Verification**:
  - Clean `tsc --noEmit` and Vite production build (1m 39s).
  - 10/10 Playwright tests passing cleanly across desktop and mobile suites (`e2e/entry-rebuild.spec.ts` and `e2e/graph-rebuild.spec.ts`).
  - Zero modifications to backend APIs or IR schema contracts; dark theme remains 100% preserved.



