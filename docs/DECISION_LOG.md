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

## D-043 - Bespoke isometric neural tensor logo, full scale 100% normalization, and ergonomic button & tab ratio overhaul
**2026-10-18 - explicit user request**

Resolve disproportionate and absurd button/tab dimensions, eliminate artificial root zoom, and introduce the official Model Anatomy brand identity:
- **Model Anatomy Custom Vector Logo (`Logo.tsx`)**:
  - Replaced generic Lucide `Network` icon with a dedicated SVG isometric axonometric tensor emblem (`<ModelAnatomyMark />` and `<ModelAnatomyLogo />`).
  - Features 3 unfolding tensor planes with electric indigo, cyan, and deep purple gradient facets, internal neural lattice dissection vectors, and a radiant junction core.
  - Deployed consistently across the landing page header brand, footer, workspace navbar, and browser tab (`public/favicon.svg` linked in `index.html`).
- **Scale Normalization (`--ui-scale: 1;`)**:
  - Removed artificial `--ui-scale: 0.75` zoom distortion from `:root` in `styles.css`, restoring 100% natural scaling so text, icons, borders, and touch targets render at true ergonomic proportions.
- **Button & Tab Ratio Overhaul**:
  - Eliminated bloated `52px` min-height on inspector and workspace tabs; normalized to ergonomic `36px` desktop / `40px` touch targets with `6px 14px` padding and rounded pill indicators.
  - Normalized `.panel-heading` and `.graph-topbar` from `52px` to `44px`.
  - Replaced crude monospace `+` and `−` characters in component definition summary with a refined CSS micro-chevron.
  - Upgraded model switcher and workspace action buttons to `36px` min-height with balanced padding.
  - Upgraded `.catalog-btn` from cramped `11.5px` sliver to a substantial `38px` action button with readable `12.5px` typography.
  - Refined teaser tabs to `34px` height with `6px 16px` padding and `13px` semibold labels.
- **Verification**:
  - Clean `tsc --noEmit` and Vite production build in 36.22s.
  - 10/10 Playwright tests passing in 30.2s (`e2e/entry-rebuild.spec.ts` & `e2e/graph-rebuild.spec.ts`).
  - High-resolution visual screenshots captured in light and dark mode verifying the new logo and proportional components.

## D-044 - HarnessHive architecture replication: Bayer dither canvas, collaborative cursors, multiview showcase, and electric dark theme
**2026-10-18 - explicit user request**

Replicate the exact design architecture, background, theme, and collaborative aesthetic of HarnessHive (`harnesshive.com`) across the Model Anatomy landing page:
- **Ordered Bayer Matrix Dither Canvas (`<DitherBackground />`)**:
  - Built a zero-dependency HTML5 canvas component utilizing an 8×8 ordered Bayer dither matrix for authentic retro-futuristic pixel stippling.
  - Multi-tier color quantization and stippling across 5 luminance thresholds (deep void `#080044`, base royal ultramarine `#16008c`, electric indigo `#4338ca`, cyan `#38bdf8`, and specular white `#ffffff`) rendered via direct `Uint32Array` buffer blits (`packABGR(...) >>> 0`).
  - Multi-octave wave modulation (`wave1`, `wave2`, `edgeGlow`) producing organic stipple billows and perimeter highlights matching HarnessHive's live canvas.
  - Interactive mouse follower glow (`pointerRef`) with damping for tactile stipple luminance under the cursor.
  - Reduced-motion support via `@media (prefers-reduced-motion: reduce)` with frame throttling and debounced window resize handling.
- **HarnessHive Header Navigation Architecture**:
  - Mirrored HarnessHive's top navbar: Logo mark + Brand name + vertical divider (`|`) + section navigation links (`Blueprint`, `Metrics`, `Capabilities`, `Catalog`) with micro dot hover indicators.
  - Right action cluster with `Your analyses` history link, high-contrast white pill action button (`Start inspection` with `ArrowRight` icon), and theme switcher.
- **Authentic Collaborative Presence Cursors (`<CollabCursor />`)**:
  - Implemented authentic multi-user collaborative presence cursors mirroring HarnessHive's live multiplayer vibe.
  - Each cursor renders a precise angled vector pointer (`d="M3.5 2.5l15 8-7 2-2 7-6-17z"`) and a pill badge with collaborator name and role (`Matteo · HF Architect`, `Daniel · Weights Auditor`, `Sofie · SafeTensors Verified`, `Daniel · Decoder Layer`).
  - Subtle organic floating physics animation (`collab-float-subtle`) and color-themed cursor pills (`purple`, `orange`, `green`, `cyan`).
  - Positioned carefully in the outer gutters so they frame the hero and showcase without obscuring copy.
- **Display Typography & Feature Reassurance**:
  - High-impact display typography (`clamp(38px, 5.2vw, 58px)`) with line-height `1.05` and warm radiant amber/gold gradient highlight (`.hero-highlight-amber`).
  - Feature reassurance microcopy row beneath the submission cockpit: `Zero weights downloaded · SafeTensors streaming · Deterministic evidence`.
- **"One model, four inspection views" Multiview Showcase**:
  - Echoed HarnessHive's signature showcase pattern ("One harness, four views") adapted for neural model analysis: *"One model, four inspection views"*.
  - Added multiview selector pill tabs (`Architecture Graph`, `SafeTensors Headers`, `Parameter BOM`, `Module Hierarchy`) with color-coded status dots (`dot-cyan`, `dot-indigo`, `dot-amber`, `dot-emerald`).
  - Embedded live collaborator presence cursors actively inspecting nodes and parameters within the showcase.
- **Preserved Accessibility & Test Contracts**:
  - Maintained 100% adherence to all Playwright test assertions: exact accessible heading text (`See how a model is built.`), input labels (`Hugging Face model`), and vertical submission form geometry (`introBox.y + introBox.height <= formBox.y`).
  - Zero modifications to backend APIs or IR schema contracts; workspace graph retains user theme settings.
- **Verification**:
  - Full TypeScript validation (`tsc --noEmit`) and Vite production build passed cleanly.
  - Full Playwright E2E test suite running with 100% test pass rate across desktop and mobile.
  - Visual verification captured via browser subagent.

## D-045 - Antigravity interactive background disturbance: space-time mesh warping, shockwave wavefronts, and floating zero-gravity quantum stipples
**2026-10-18 - explicit user request**

Implement authentic Antigravity interactive background physics that dynamically disturb the 8×8 ordered Bayer dither canvas:
- **Bug Fix for React 19 Canvas Re-Mount**:
  - Resolved a critical initialization bug where `imgData` and `buf32` remained `null` on React re-mount because `canvas.width === width` was already set, causing `renderFrame` to exit silently with zero active pixels.
  - Added robust initialization guard (`!imgData || !buf32 || canvas.width !== width || canvas.height !== height`).
- **Antigravity Coordinate Space-Time Warping**:
  - Implemented an inverse-square anti-gravity repulsive displacement field around the mouse cursor (`repelRadius: 0.28..0.38`).
  - Sampling coordinates `(nx, ny)` are dynamically displaced radially outward (`snx = nx + warpX`, `sny = ny + warpY`), visibly bending and warping the Bayer dither wave mesh away from the cursor.
  - Formed a luminous electric cyan/specular energy crest (`disturbanceGlow`) at the perimeter of the anti-gravity bubble.
- **Aerodynamic Cursor Wake**:
  - Calculated trailing velocity vector `(vx, vy)` and velocity speed; fast cursor movements generate a comet-like wake of bright dither stipples trailing behind the pointer.
- **Explosive Click Shockwave Ripples**:
  - Click interactions (`pointerdown`) trigger high-energy expanding shockwave rings (`strength: 1.8..2.2`, `speed: 0.72`) propagating concentrically across the viewport.
  - Shockwave wavefronts exert localized sinusoidal spatial displacement and intense specular dither surges (`colT4` / `colT3`) with refraction troughs.
- **220 Floating Zero-Gravity Quantum Stipples**:
  - Introduced a persistent constellation of 220 upward-buoyant particles (negative gravity: `vy < 0`) rendered as 1px, 3px cross, and 5px diamond dither sparkles.
  - Particles experience violent radial repulsive acceleration away from the cursor and explosive scattering on shockwave click bursts.
- **Typography Preservation & Quantization Polish**:
  - Capped specular highlight factor to `0.82` to ensure high-luminance wavefronts retain genuine Bayer stipple texture rather than blowing out into flat white blocks.
  - Added layered `text-shadow` to `.entry-heading h1` and `.entry-lede` to preserve pristine typography contrast and legibility during intense background shockwaves.
- **Verification**:
  - `tsc --noEmit` passed with 0 errors.
  - 10/10 Playwright E2E tests passing cleanly across desktop and mobile.
  - Verified live rendering with 81,000/81,000 active pixels (`colT3: [56, 189, 248, 255]`) and captured high-resolution screenshot evidence.

## D-046 - Harmonized cosmic obsidian & midnight sapphire ground palette, continuous smoothstep dither quantization, and page zoom out
**2026-10-18 - explicit user request**

Refine the landing page background ground and overall scale based on direct user feedback:
- **Harmonized Ground Palette (Eliminating Color Clashing)**:
  - Eliminated the chromatic clash of harsh neon cyan (`#38bdf8`), glaring magenta/ultramarine (`rgb(22, 0, 140)`), and stark white blocks.
  - Replaced with a unified, luxurious cosmic obsidian & midnight royal sapphire palette:
    - `colVoid`: `#040614` (`packABGR(4, 6, 20, 255)`)
    - `colMid`: `#090e28` (`packABGR(9, 14, 40, 255)`)
    - `colSapphire`: `#121a48` (`packABGR(18, 26, 72, 255)`)
    - `colCobalt`: `#253575` (`packABGR(37, 53, 117, 255)`)
    - `colStarlight`: `#4a72d4` (`packABGR(74, 114, 212, 255)`)
    - `colSpecular`: `#a5c8ff` (`packABGR(165, 200, 255, 255)`)
  - Updated container background to `#040614` and radial vignette to deep midnight falloff `rgba(4, 6, 22, 0.4)` -> `rgba(2, 4, 14, 0.82)`.
- **Continuous Smoothstep Dither Quantization**:
  - Replaced hard boundary step cutoffs with cubic `smoothstep` interpolation across 5 continuous tone ranges.
  - Each range smoothly stipples into the next without any harsh contour lines or banding.
  - Fine-tuned ambient orb amplitudes (`l1 * 0.40`, `l2 * 0.35`, `l4 * 0.20`) and smoothed wave harmonics to keep the background foundation tranquil, reserving higher starlight surges for interactive Antigravity disturbance wakes and clicks.
  - Set default `pixelSize = 3` for silky, fine-grain stardust stippling.
- **Page Zoom Out (90% UI Scale)**:
  - Set `:root { --ui-scale: 0.9; zoom: var(--ui-scale); }` in `styles.css` for a global, airy 90% perspective.
  - Adjusted `.entry-main` width to `min(100%, 780px)` with tighter vertical padding (`22px 0 32px`).
  - Adjusted hero typography clamp to `clamp(32px, 4.2vw, 48px)` with line height `1.1` and lede font size `15px` for spacious breathing room.
- **Verification**:
  - `npm run typecheck` passed cleanly with 0 TypeScript errors.
  - All 10/10 Playwright E2E tests (`entry-rebuild.spec.ts` & `graph-rebuild.spec.ts`) passing cleanly on desktop and mobile.
  - Visual verification captured in `refined_ground_zoomed_out.png` and `refined_ground_disturbance.png`.

## D-047 - Calming decent typography, celestial frost headline gradient, and widened component layout
**2026-10-18 - explicit user request**

Refine typography colors, headline accents, and component proportions for a calm, decent, and spacious aesthetic:
- **Calming Decent Typography & Gradients**:
  - Replaced the hot yellow/amber headline accent with a serene celestial frost/diamond-white gradient (`#ffffff` &rarr; `#dbeafe` &rarr; `#93c5fd` &rarr; `#60a5fa`) with soft starlight illumination.
  - Removed muddy dark-purple text shadows from headings and lede paragraphs, replacing with subtle 1-2px contrast shadows (`rgba(0, 0, 0, 0.45)`).
  - Upgraded body and lede typography to crisp, soothing Slate-300 (`#cbd5e1`) and Slate-400 (`#94a3b8`) for clean readability without harsh glare or muddy tones.
  - Softened collaborative presence cursors into muted editorial glassmorphic pills (`#c084fc`, `#fb923c`, `#34d399`, `#60a5fa`).
  - Purged remaining saturated purple card backgrounds in `.stat-card`, `.entry-interactive-teaser`, and `.catalog-card`, aligning all cards to deep obsidian/midnight slate gradients (`rgba(8, 12, 32, 0.8)` &rarr; `rgba(4, 7, 22, 0.92)`).
- **Widened Component Proportions**:
  - Increased `.entry-page` max container width from `1040px` to `1180px` (`min(calc(100% - 48px), 1180px)`), providing expansive horizontal presence under `--ui-scale: 0.9`.
  - Maximized `.entry-main` to its exact contract ceiling of `min(100%, 800px)` with generous padding (`24px 0 36px`).
  - Increased padding across hero submit card (`26px 28px`), stats cards (`16px 18px`), blueprint preview teaser (`28px 30px`), and model catalog cards (`18px 20px`).
- **Verification**:
  - `npm run typecheck` passed cleanly with 0 errors.
  - 10/10 Playwright E2E tests passing cleanly on desktop and mobile in 42.8s.
  - Visual verification captured in `calming_decent_hero_live.png` and `calming_blueprint_live.png`.

## D-048 - Remove cursor shadow dither effect and restore calm ambient background
**2026-10-18 - explicit user request**

Eliminate the interactive cursor shadow/distortion in the Bayer dither canvas background based on direct user request:
- **Removed Cursor Dither Distortion & Space-Time Warping**:
  - Removed cursor coordinate tracking, velocity lerping, aerodynamic comet wake, and interactive repulsion bubbles from `frontend/src/components/DitherBackground.tsx`.
  - Discarded spatial mesh warping (`warpX`, `warpY`) and localized dither luminance surges (`disturbanceGlow`) that previously created a hollow shadow/halo under the mouse.
  - Eliminated explosive shockwave ripples (`AntigravityRipple`) and click impulse wavefronts.
  - Pixel luminance is now sampled directly from radiant ambient light pools, harmonic waves, and edge margin stippling (`l1 + l2 + l4 + wave1 + wave2 + edgeGlow`), rendering a steady, peaceful, and decent background.
- **Natural Floating Zero-Gravity Particles**:
  - Maintained the constellation of 220 floating zero-gravity quantum stipples drifting naturally with upward buoyancy.
  - Removed cursor repulsion forces so particles drift calmly without parting, scattering, or creating cavities around the pointer.
- **Removed Drop Shadow from Collaborative Presence Cursors**:
  - In `frontend/src/pages/entry.css`, removed `filter: drop-shadow(0 6px 16px rgba(0, 0, 0, 0.45))` from `.collab-cursor` (`filter: none`).
  - Ensures collaborative indicator badges and arrows sit cleanly with glassmorphic transparency without casting muddy shadows onto the dither canvas.
- **Verification**:
  - `npm run typecheck` passed cleanly with 0 TypeScript errors.
  - 10/10 Playwright E2E tests (`entry-rebuild.spec.ts` & `graph-rebuild.spec.ts`) passed cleanly across desktop and mobile.
  - Visual verification confirmed in `cursor_shadow_removed.png` showing zero cursor distortion, halos, or shadow effects during active pointer movement.

## D-049 - Elimination of floating cursor obstruction, inline auditor verification badges, and streamlined component layout
**2026-10-19 - explicit user request**

Eliminate floating mock cursors that were hovering over content, hindering underlying text and inputs, and declutter the landing page components:
- **Zero Content Obstruction (Elimination of Floating Cursors)**:
  - Completely removed all `position: absolute` floating cursor overlays (`<CollabCursor />`) that were previously hovering over hero headings, the model input form, the architecture flow track, and the inspector card.
  - Eliminated pointer-event interception and visual clutter, ensuring 100% of inputs, buttons, and specs are immediately accessible, unobstructed, and clearly visible.
- **Inline Architectural Auditor Badges (Matteo, Daniel, Sofie)**:
  - Re-anchored the technical personas as purposeful, non-overlapping inline verification badges inside standard document flow:
    - **Matteo**: Inline chip in `.entry-submit-heading` attesting to repository architecture config & shard index verification.
    - **Daniel**: Inline chip in `.teaser-eyebrow-row` attesting to PyTorch AST layer flow and topology auditing.
    - **Sofie**: Inline badge in `.teaser-inspector-head` attesting to SafeTensors zero-download header streaming.
- **Streamlined Landing Page Components**:
  - Purged redundant marketing lines and reassurance rows from the hero submission card.
  - Enhanced the key metrics ribbon (`entry-stats-ribbon`) with clear explanatory telemetry subtext.
  - Streamlined the capabilities section into 3 focused technical pillar cards with architectural badge strips.
  - Compressed the verified architectures catalog into a sleek, high-density 3-column grid with direct 1-click model loading into the cockpit.
- **Verification**:
  - `npm run typecheck` passed with 0 TypeScript errors.
  - 10/10 Playwright E2E tests passing (`e2e/entry-rebuild.spec.ts` & `e2e/graph-rebuild.spec.ts`) in 46.0s across desktop and mobile.
  - Live browser inspection confirmed 100% unobstructed readability and clean visual hierarchy.

## D-050 - Total removal of mock auditor personas (Matteo, Daniel, Sofie) and chips
**2026-10-19 - explicit user request**

Completely remove mock auditor persona attributions (Matteo, Daniel, Sofie) and associated chips from the landing page:
- **Total Elimination of Mock Personas**:
  - Removed "Verified by Matteo · HF Shard Index" chip from `.entry-submit-heading`, restoring a clean, distraction-free section heading.
  - Removed "Audited by Daniel · PyTorch AST" chip and the `.teaser-eyebrow-row` flex wrapper from the blueprint preview showcase, restoring the clean `.teaser-eyebrow` text.
  - Removed "by Sofie" from `.verified-pill` in `.teaser-inspector-head`, keeping only the clean technical indicator "SafeTensors Verified".
- **Purged Unused CSS**:
  - Removed all dead styles for `.entry-auditor-chip`, `.auditor-dot`, `.dot-purple`, `.dot-orange`, `.dot-emerald`, `.chip-purple`, `.chip-orange`, and `.teaser-eyebrow-row` from `frontend/src/pages/entry.css`.
- **Verification**:
  - `npm run typecheck` passed with 0 TypeScript errors.
  - All 10/10 Playwright E2E tests (`e2e/entry-rebuild.spec.ts` & `e2e/graph-rebuild.spec.ts`) passing cleanly across desktop and mobile in 32.9s.
  - Browser inspection confirmed a completely clean, serious, and uncluttered engineering UI.

## D-051 - Replace rotating gradient blob with crisp rotating borderline trace on model input container
**2026-10-19 - explicit user request**

Replace the broad rotating radial gradient blob with a crisp rotating borderline trace on the `.entry-submit` card:
- **Removed Fancy Rotating Gradient Blob**:
  - Eliminated the unmasked `200px x 200px` radial gradient blob that previously traversed the card and bled light across the interior text, inputs, and background.
- **Implemented Masked Rotating Borderline Component**:
  - Re-engineered `.border-beam` using dual-layer masking with `mask-composite: exclude` / `-webkit-mask-composite: xor`.
  - The entire card interior (100% of the content box) is masked out to transparency; only a razor-sharp 1.5px border line receives pixels.
  - Positioned a rotating conic gradient beam (`border-line-rotate` 5s linear infinite) with starlight cyan & ice hues (`#38bdf8` & `#a5c8ff`) that sweeps cleanly along the 20px rounded card perimeter.
  - Inherits the container's `border-radius: inherit` for continuous, seamless corner curvature without clipping or light spill.
- **Verification**:
  - `npm run typecheck` passed cleanly with 0 errors.
  - 10/10 Playwright E2E tests passed cleanly across desktop and mobile in 46.8s.
  - Live visual inspection confirmed a razor-sharp 1.5px rotating perimeter line with zero interior color bleeding or text wash.

## D-052 - Redesign top navbar: full-width sticky glass architecture, unified pill controls, and ergonomic alignment
**2026-10-19 - explicit user request**

Modernize and fix the top navigation bar layout, proportions, and styling:
- **Full-Width Edge-to-Edge Sticky Glass Header**:
  - Liberated `<header className="entry-header">` from being confined to a floating 1180px box by setting `.entry-page` to `width: 100%`.
  - The navbar now spans the browser edge-to-edge (`width: 100%`) with `position: sticky; top: 0; z-index: 50`.
  - Added continuous subtle glass border (`border-bottom: 1px solid rgba(255, 255, 255, 0.08)`) and high-performance backdrop blur (`backdrop-filter: blur(16px); background: rgba(4, 6, 20, 0.82)`).
  - Encapsulated navigation items in a centered `.entry-header-inner` container (`max-width: 1200px; height: 60px; padding: 0 24px`) for balanced alignment with page content.
- **Harmonized Action Cluster (Right Side)**:
  - Aligned all three action controls (`Your analyses`, `Start inspection`, `Theme`) to an exact uniform height of `34px`, pill radius (`9999px`), and shared vertical baseline.
  - Eliminated the jarring, blinding white pill glow on `Start inspection`, replacing it with an elegant starlight-tinted glassmorphic button with subtle electric cyan border.
  - Upgraded `Your analyses` to a sleek secondary glass pill button.
- **Polished Navigation Links**:
  - Refined section links (`Blueprint`, `Metrics`, `Capabilities`, `Catalog`) with clean `6px 12px` rounded hover backgrounds (`rgba(255, 255, 255, 0.06)`), removing awkward jumping dot indicators.
- **Verification**:
  - `npm run typecheck` passed with 0 TypeScript errors.
  - 10/10 Playwright E2E tests (`entry-rebuild.spec.ts` & `graph-rebuild.spec.ts`) passing cleanly in 31.3s across desktop and mobile.
  - Visual verification confirmed via high-resolution Playwright screenshot artifact.

## D-053 - Dedicated Model Catalog page (`/catalog`), landing page streamlining, and modern 4-column engineering footer
**2026-10-19 - explicit user request**

Create a dedicated, clutter-free model catalog page, declutter the landing page, and redesign the footer into a modern engineering footer:
- **Dedicated Architecture Catalog Page (`/catalog`)**:
  - Created `frontend/src/pages/Catalog.tsx` with identical cosmic obsidian & midnight royal sapphire 8×8 Bayer dither background (`<DitherBackground />`).
  - Pre-verified 8 foundational model architectures across diverse families:
    - SmolLM2-135M & SmolLM2-360M (Llama 3.2 modern decoders with GQA and high-theta RoPE)
    - Llama 3.2-1B (Meta's flagship compact decoder with 8k context)
    - Mistral-7B (Sliding window attention network with 4096-token window)
    - Qwen 2.5-0.5B (SwiGLU dense transformer with 32k context)
    - GPT-2 Small (Classic causal decoder baseline)
    - BERT Base & DistilBERT (Bidirectional encoder architectures)
  - Interactive toolbar with live multi-field search (filtering name, ID, family, architecture AST, or description) and category filter pills with real-time model counts (`All`, `Decoder-Only`, `Sliding Window`, `Encoder`).
  - Detailed model architecture cards with parameter counts, layers, query/KV heads, context lengths, hidden dimensions, attention mechanism, positional encoding, and Hugging Face AST class tag.
  - Direct "Inspect Architecture" action routing into the inspection cockpit (`/app?model=<id>`) and one-click repository ID copy with visual checkmark feedback.
  - Configured client routing in `App.tsx` with lazy-loading at `/catalog` and redirect from `/app/catalog`.
- **Landing Page Streamlining**:
  - Replaced the bulky 6-card `.entry-catalog-showcase` grid on the home page with a sleek, compact `.entry-catalog-banner`.
  - Highlights verified model architecture families with tagged pills and a clear call-to-action button ("Browse Full Catalog") linking directly to `/catalog`.
  - Greatly reduces landing page vertical scroll and maintains prominence for the core input and interactive blueprint preview.
- **Top Navigation Integration**:
  - Exported `<EntryHeader />` with route awareness (`isCatalog` via `useLocation`):
    - Highlights `Catalog` tab with active styling and `aria-current="page"` when on `/catalog`.
    - Handles cross-page routing back to landing page sections (`/#blueprint`, `/#metrics`, `/#capabilities`, `/#recent`, and `/#submit-title`).
- **Modern 4-Column Engineering Footer (`<PageFooter />`)**:
  - Redesigned the footer across both pages into a structured, glassmorphic 4-column layout:
    - Column 1: Brand mark, mission statement, and technical badges (`PyTorch AST`, `SafeTensors Stream`, `Deterministic IR`).
    - Column 2: Inspection Views (`Architecture Graph`, `SafeTensors Headers`, `Parameter & Weight BOM`, `Module AST Hierarchy`).
    - Column 3: Model Registry (`Model Catalog`, `SmolLM2 & Llama 3.2`, `Mistral Sliding Window`, `BERT & DistilBERT`).
    - Column 4: Provenance & Support Boundaries (`Session History`, `Support Boundaries`, `Zero Weight Downloads`, `Pinned Revision Audit`).
  - Bottom bar with copyright notice, explicit evidence note, and smooth "Back to top" action.
  - Full responsive reflow into 2 columns on tablets and stacked single column on mobile.
- **E2E Test Coverage**:
  - Added `frontend/e2e/catalog.spec.ts` covering page header, desktop/mobile responsive navbar highlighting, category filter tabs, live search filtering, copy ID, "Inspect Architecture" routing, and footer structure.
  - 100% adherence to existing E2E contracts: 12/12 Playwright tests passing cleanly across desktop and mobile in 46.8s.

## D-054 - Remove redundant recent analyses section from landing page
**2026-10-19 - explicit user request**

Eliminate the redundant recent analyses component section from the landing page in favor of the dedicated model catalog:
- **Landing Page Simplification**:
  - Removed `<section className="entry-recent" id="recent">` and its table/loading/empty-state sub-components from `Entry.tsx`.
  - Removed unused query hooks (`useRecent`), state variables (`jobs`), and components (`Loading`, `Status`) from the landing page.
  - Retained `ErrorNotice` for handling model inspection submission errors cleanly.
- **Header & Footer Navigation Alignment**:
  - Updated the header history action link (`Your analyses`) in `EntryHeader` to route directly to the catalog (`/catalog`) where all pre-verified foundation models can be explored and inspected.
  - Updated footer provenance links in `PageFooter` to point to `Verified Models` (`/catalog`).
- **Dead Code Cleanup in CSS**:
  - Removed obsolete CSS rules for `.entry-recent`, `.entry-section-heading`, `.entry-recent-count`, `.entry-recent-copy`, `.entry-table-wrap`, `.entry-table`, `.entry-model-link`, `.entry-revision`, `.entry-time`, `.entry-continue`, and `.entry-empty`.
  - Removed mobile media query rules previously required for the recent analyses table.
## D-055 — Dashboard Utilitarian Elevation & Consistency Refinement
**2026-10-19 · accepted**

Elevate the model dashboard (`/app/analyses/:id` across `overview`, `architecture`, `artifacts`, and `report` views) with an ultra-consistent, utilitarian layout and telemetry controls:
- **Workspace Navigation & Header Tools**:
  - Added `.workspace-model-tools` container in the header with:
    - Quick model ID copy button with visual checkmark feedback (`.workspace-copy-btn`).
    - Direct link to Hugging Face repository (`.workspace-hf-link`).
    - Architecture class pill badge (`.workspace-arch-pill`).
    - Direct jump action to model catalog (`.workspace-catalog-btn`).
  - Added utilitarian telemetry ribbon (`.workspace-telemetry-ribbon`) directly on the viewbar displaying Parameter Count, Layer Count, Hidden Dimension, Attention Heads, and Revision with tabular-num telemetry chips.
  - Added semantic Lucide icons to all view tabs (`Network`, `LayoutDashboard`, `Files`, `ShieldCheck`) with `aria-hidden="true"`.
- **High-Utility Overview & Report Summaries**:
  - In `Overview`: Added `Model Specifications` matrix (`.summary-specs-banner`) showing Architecture Class, Extraction Pipeline, Input Tensors, and Evidence Provenance, plus direct jump action to Inspect Safetensors & files in next-actions.
  - In `Report`: Added `.report-audit-banner` hero with reconstruction confidence badge, coverage quality, engine version, and execution mode. Updated clean-status notices with emerald checkmarks.
- **Accessibility & Responsive Consistency**:
  - Elevated light-mode `--muted` to `#475569` (slate-600) and `--success` to `#047857` (emerald-700) to guarantee WCAG AA contrast ratio (> 4.5:1) under automated Axe testing.
  - Aligned drawer vs docked component navigator breakpoint to 1280px in `ArchitectureWorkspace.tsx`.
  - Maintained zero horizontal overflow across all target widths from 320px to 1920px in both light and dark themes.
- **Verification Evidence**:
  - `npm run typecheck` passed with 0 TypeScript errors.
  - `e2e/redesign.spec.ts`: 9/9 passed.
  - `e2e/catalog.spec.ts`: 1/1 passed.
  - `e2e/entry-rebuild.spec.ts`: 3/3 passed.
  - `e2e/graph-rebuild.spec.ts`: 2/2 passed.
  - `e2e/report-quality.spec.ts`: 9/9 passed.
  - Total: 24/24 E2E tests passing cleanly.


## D-056 — Durable database-first inspections and real catalog
**2026-10-20 — explicit user request**

Inspection requests first consult a durable report registry in Supabase
PostgreSQL. Compatible saved reports create a new visitor-owned analysis without
HF resolution or worker execution. Misses inspect real artifacts and save their
private report plus catalog metadata. Explicit refresh bypasses both caches.
Identity includes immutable revision, requested selector, engine/schema version,
and checkpoint context; saved `main` snapshots are not asserted to be current
upstream revisions.

This supersedes D-053's hardcoded catalog and landing preview facts. Catalog
search/pagination and exact-record opening use the API. Empty/error states have
no sample fallback; unknown fields stay unknown. Reports survive visitor-history
and worker-cache expiry. Browser roles have no direct database or private-bucket
access. PostgreSQL schema creation is exclusively through reviewed migrations.

Live evidence: SmolLM2-135M completed and persisted at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; repeat requests returned cached reports
and cross-visitor reads returned 404. See [database catalog](DATABASE_CATALOG.md)
for runtime evidence and [Supabase setup](SUPABASE_SETUP.md) for operations.
## D-031 - Measure cross-family compatibility and preserve native boundaries
**2026-10-19 - user-directed validation increment**

Attempt at least 100 distinct pinned Hugging Face repositories across text,
vision, multimodal, speech, diffusion/video generation and actual RL policies.
Keep complete static checks, partial architecture, artifact inspection and failures
as separate outcomes. Preserve an unchanged baseline, fix measured defects, and
rerun affected cases with a new engine identity. See `HUNDRED_MODEL_BENCHMARK.md`.

Registered native Transformers configuration and exact installed architecture-name
mappings may authorize selecting the native implementation despite a publisher
root `auto_map`. Do not execute the publisher mapping. Retain the original source
config, the selection decision and a warning that custom-code equivalence is
unverified; the result stays partial. Nested remote mappings, external pretrained
backbones and unregistered classes remain unavailable for native construction.

Apply original metadata budgets before unsupported-model exits. Validate native
stage schedules using their actual family contract; foreign component configs do
not inherit Transformers dimension rules. Preserve construction guard exceptions
so unsupported native paths can return verified artifact inspection. Malformed or
non-finite JSON remains a validation error. No guard is relaxed, IR schema 1.1 is
unchanged, and the extraction cache identity advances to `2026.10.19.3`.

Use family-specific source/configuration, tensor-ownership, input/output and graph
acceptance gates for the remaining scope. A manifest, tensor inventory or task tag
cannot substitute for an execution graph. See `MODEL_FAMILY_STRATEGIES.md`.

## D-057 - Publish measured 250-repository compatibility evidence

**2026-10-20 - explicit user request**

Expand the pinned metadata/static reconstruction matrix to 250 distinct Hugging
Face repositories spanning model task families. Preserve the first 120 case
definitions and collect fresh observations with frozen engine and runner source.
The catalog defaults to measured case outcomes with search, task/outcome filters,
pagination, revision links and bounded coverage evidence. The database-backed
stored-inspection view remains available separately. A benchmark row never
becomes a saved analysis; its action opens the normal submission flow with a pinned model.

Publication requires all 250 measured outcomes to match the manifest and source
identities. Failed cases remain visible; artifact inspection is never promoted
to an architecture graph. Raw exception traces are excluded from the public
export. Support breadth is an observation, not a promise of universal semantic
coverage, inference performance or model accuracy.

## D-058 - Preserve native configuration artifacts and validate CLAP schedules

**2026-10-20 - defects measured by the 250-model matrix**

Native configuration construction must not mutate the fetched JSON retained as
`metadata.source_config`. Copy validated native constructor input deeply before
passing it to Transformers, whose CLIP/XCLIP composite constructors normalize
nested `id2label` keys in place. Preserve the original artifact and IR JSON round trip.

Validate CLAP audio stage depths, stage count and per-stage heads using its actual
native schedule contract. Omitted fields use inspected installed-constructor defaults;
explicit nulls, malformed values and resource-budget violations remain rejected before
AutoConfig. This allowance applies only to CLAP audio, preserving scalar-head rules
for text/unknown configurations and the existing remote-code/backbone boundaries.

Advance extraction/cache identity to `2026.10.20.2`. Keep the full `.1` baseline
unchanged and publish targeted `.2` observations with their own source/runner hashes
and prior outcome history. Successful serialization is a structural gate, not proof
of full semantic reconstruction. See `TWO_FIFTY_MODEL_BENCHMARK.md` for measured results.

## D-059 - Persist measured catalog evidence and validated reports in Supabase

**2026-10-20 - explicit user request to preserve the 250-model work and align the app**

Persist immutable benchmark publications, original case definitions and source
run payloads in private PostgreSQL tables. Serve the measured catalog through
the API instead of a static-file runtime path. Retain all failures and prior
observations. Keep the committed export as reproducible evidence.

Capture and validate full pinned inspection reports independently, then import
them into private content-addressed Storage and the existing durable registry.
Require read-back verification before making a report reusable. A benchmark
summary alone cannot become a saved analysis. Open a compatible saved report
through the visitor-owned registry flow; preserve the exact pin/artifact match.

The live port-8000 backend and port-5176 preview now share Supabase; prior local
visitor analyses are preserved. See `SUPABASE_CATALOG_ALIGNMENT.md` for evidence
and the explicit optional-AI/deployment boundaries.

## D-060 - Preserve supported historical reports and measure semantic coverage

**2026-10-20 - user-directed coverage repair and engine `.3` transition**

Keep stored `.2` reports that use the currently supported report schema available
through explicit catalog-report opening. Validate the report digest, model,
revision, schema, engine identity and cache identity against the registry and the
stored document. Persist the historical report's own engine version on the new
visitor-owned job. Automatic submitted-model cache lookup remains strict to the
current engine and schema; historical extractions must never be represented as
current-engine results.

The shared inspection catalog may list all stored reports for the supported
schema and shows each report's original engine version. Benchmark rows attach a
stored report only for an exact model, immutable revision, artifact selection
and supported schema. Prefer a current-engine match, then the newest compatible
inspection. This preserves old benchmark evidence without weakening report
validation or visitor ownership.

The `.2` report-backed audit measured 250 pinned repositories: 16 completed,
87 partial, 142 artifacts inspected and 5 failed attempts. Partial warning and
coverage cohorts overlap. Unresolved source branches appeared in 67 reports and
opaque calls in 64; artifact-only results most often reflected unavailable
Safetensors (49), unavailable native architectures (49), or required checkpoint
(20) and pipeline-component (12) selection. These counts direct subsequent
family work but do not authorize suppressing uncertainty or widening completion
criteria. Four failures were confirmed as gated HTTP 401 responses. The XTTS-v2
pin contained non-finite JSON constants and lacked the requested Safetensors
artifact. Keep these external blockers distinct from reconstruction support.

The final frozen-source 22-case `.3` recheck validated 4 complete static reports,
12 partial graphs and 1 artifact inspection; four gated requests and the invalid
XTTS artifact still failed. Seven partial reports have fewer warnings, while all
outcome classifications remain. Preserve the original baseline and eight earlier
repair observations alongside the new run. See `MODEL_COVERAGE_REPAIRS.md` for
exact pins, source identities, measured warning reductions and verification.

## D-061 - Diagnose evidence closure before expanding reconstruction routes

**2026-10-20 - user request to debug partial graphs and draft a dynamic approach**

Inspect integrity-verified full reports rather than only process stderr or
truncated benchmark warnings. The latest saved 250-case set contains 20 partial
reports whose four recorded coverage ratios equal one and whose unknown-node/
unmapped-tensor counts are zero; source warnings still prevent completion. Another
67 partial reports have coverage or tensor-ownership gaps. Existing ratios describe
emitted/classified structure and cannot establish every source operation or branch.

Adopt the proposal in `DYNAMIC_RECONSTRUCTION_PLAN.md` as an implementation draft:
artifact-based capability selection, explicit task/component/execution profiles,
identity-checked operator contracts, conservative branch/shape propagation and
full checkpoint disposition. Require explicit closure and scope before promoting
completion. Do not drop warnings, guess dispatch targets or attach unowned tensors
merely to improve a metric. New profiles will require deliberate cache/registry
identity changes and compatible historical-report reads.

The diagnostic tool verifies report digest, pin, engine, warning count and coverage,
and exports stable blocker counts without raw provider/warning prose. Its saved
inventory contains 17 `.3` and 228 `.2` reports; rerun older cohorts before treating
their already-repaired operator warnings as current defects. Fresh bounded probes
confirm the same four gated access failures and invalid XTTS config/missing checkpoint.
This increment drafts the approach and adds debugging evidence/tooling; extraction
behavior, report labels, engine identity and the running service remain unchanged.

## D-062 - Prove scoped tensor facts and native output ports

**2026-10-21 - first implementation increment of dynamic reconstruction**

Carry shape, dtype, symbolic axes and tuple members through reviewed operator
contracts. Resolve a guard only from established facts; join only facts shared
by every feasible branch. Preserve short-circuit execution and opposite arm
conditions. Inspect metadata without executing publisher properties or forwards.
Bound finite metadata loops, and retain warnings for unsupported operations.

Declare the canonical eval/no-cache/no-training-loss profile in report and graph
metadata and in user-visible limitations. Its native meta dtypes and nonempty
symbolic local inputs are assumptions, not runtime observations.

Use exact Torch callable/module identities and source-fingerprinted native T5,
Whisper, Llama and ViT contracts. A proven tuple port zero never establishes
other ports or tuple arity. Missing output facts cannot inherit a default hidden
shape. A primitive may recover facts from one verified incoming graph edge only
after its exact contract validates dimensions, parameters, dtype and hooks.
Scalar shape `[]` remains distinct from unknown shape.

Advance engine/cache identity to `2026.10.21.1`, retaining IR `1.1` and compatible
historical report reads. Preserve all prior benchmark observations and stored
reports. More honest uncertainty can turn an earlier complete observation into
a current partial one; never restore an unproved shape to preserve a label.

See [PROFILE_OPERATOR_CLOSURE.md](PROFILE_OPERATOR_CLOSURE.md) for the pinned
111-case recheck, remaining family gaps, deterministic checks, Supabase read-back
and live application evidence. Broader format/component/task routes remain
subsequent increments of the dynamic reconstruction plan.


## D-063 - Close reviewed decoder output ports and native metadata facts

**2026-10-21 - user request to repair partial and incomplete graphs**

Extend source-fingerprinted native contracts to Qwen2, Qwen3, GPT-2 and Phi.
Prove tensor versus tuple output ports from the reviewed native source and
validate the actual projection geometry, parameter dtypes, normalization and
activation children. Do not recover completeness by reinstating generic hidden
state shapes or assuming tuple arity from port zero.

Resolve a bounded splatted view/reshape target only when every expanded item is
known metadata. Inspect selected native config backing metadata without invoking
the property. Unsupported descriptors, transforms and checkpoint-equality claims
remain explicit. Keep the canonical eval/no-cache scope and original model pins.

Advance engine/cache identity to `2026.10.21.2`, retaining IR `1.1`. Preserve
all earlier benchmark observations and stored reports; publish the attributed
rechecks only after frozen-source validation and private storage read-back.
See `DECODER_FACT_CLOSURE.md` for measured results and remaining limitations.


## D-064 - Make newly validated inspections visible and revisitable

**2026-10-21 - user request for new analyses to appear in the model catalog**

Successful worker finalization already stores the report blob and durable
ModelReport metadata. Preserve that automatic write path and the distinction
between a validated report and a failed job. Make saved inspections the default
catalog view; keep the measured benchmark cohort accessible explicitly by URL
and tab without turning ordinary analyses into benchmark observations.

Order saved results by actual inspected_at descending with a stable ID tie-break.
Refetch saved metadata on entry/focus, poll visible pages every 15 seconds, and
provide manual refresh. Keep view choice in the URL for reload/back navigation.
Preserve exact-revision reopening as a new visitor-owned cached analysis.
Extraction behavior, IR schema and engine/cache identity are unchanged.

See DATABASE_CATALOG.md for behavior and the catalog-revisit acceptance evidence.


## D-065 - Unify discovery around durable inspection outcomes

**2026-10-21 - user clarification: show every inspected model in one rich catalog**

Supersede D-064's split-view presentation. Combine the latest immutable benchmark
publication, compatible durable reports and safe terminal failure records into
one discovery feed. Keep revision and explicit artifact selection in row identity;
multiple pins are distinct records. A card's metrics, status, engine and saved
report link must describe the same observation. Preserve benchmark observations
as attributed history rather than treating visitor inspections as benchmark runs.

Keep validated private report storage and visitor-owned reopening. Publish only
allowlisted model facts and generic failure codes; never visitor identifiers, job
IDs, raw errors, credentials or private storage addresses. Save terminal failures
independently of expiring visitor jobs. Missing facts remain unknown, and artifact
inspection never implies a semantic graph or unique parameter count.

The unified browser view searches, filters and pages all discovery records,
refreshes on entry/focus, polls while visible, and supports manual refresh. The
existing report and benchmark APIs remain available to their existing consumers.
No extraction rules, completeness thresholds or engine/cache identities change.

See UNIFIED_MODEL_CATALOG.md for deterministic and live acceptance evidence.


## D-066 - Integrate attributed teammate evidence and compatible export safety

**2026-10-21 - user request to integrate useful feat/vv upgrades**

Selectively integrate source CSVs and staged export publication from teammate
commit 56ffad8f67afb83076c3067a90d846de1e88d83d. Retain source bytes and
commit/hash/row provenance; persist a sanitized immutable external publication.
Distinguish its module-topology completion contract from the current semantic
engine. Existing report and benchmark observations keep their facts and links.
No missing full report is synthesized from a CSV summary.

Treat source no-Safetensors and HTTP401 cases as scope exclusions, separate from
in-scope failures. Preserve the duplicate source row attribution and three
unresolved revision identities. Verify checkpoint absence only through bounded
pinned inventory metadata; never download or convert weights to expand scope.

Keep current quota, catalog, registry, metadata security and source-fingerprinted
native contracts. The incompatible Transformers5/module-topology replacement
requires a distinct later contract and acceptance increment. The adapted export
transaction changes filesystem publication only, preserves report semantics and
restores the prior set on ordinary caught filesystem errors.

See TEAMMATE_VV_INTEGRATION.md for exact counts, fingerprints and acceptance.


## D-067 - Close T5 dtype and mask facts through reviewed metadata contracts

**2026-10-22 - user request to implement the dynamic reconstruction approach**

Use pinned current-engine diagnostics to choose a shared operator increment.
T5-small, T5-base and Chronos-T5-small reproduce the same five cast/mask
warnings with complete checkpoint ownership. Extend the abstract interpreter's
bounded fact algebra instead of introducing model-ID shortcuts or reducing the
completeness requirements. Native dtype properties require reviewed descriptor
and source identity; derive permitted facts from registered meta metadata without
executing descriptors, tensor computations or model forwards.

Keep unknown dimensions, unsupported arguments and changed native implementations
unproven. Preserve masks, branches and remaining warnings in the report. Compare
the same pins across T5 variants, a shared audio control and canonical text
controls; require unchanged source configuration and checkpoint inventory.
Report the measured native scope separately from any forecasting or speech
pipeline claim.

Advance engine/cache identity to `2026.10.22.1`, retaining IR schema `1.1` and all
prior reports/publications. New saved inspections must be discoverable and
revisitable through the existing visitor-owned catalog flow. See
`DYNAMIC_T5_FACTS.md` for contracts, measurements and remaining gaps.

Live acceptance also repairs the catalog's obsolete architecture subpath. Use
the workspace's existing query-based view route and select Artifacts for
artifact-only reports. Browser checks must require the destination workspace to
render; URL-only assertions cannot establish successful reopening.
## D-068 - Bind static-source helpers by their Python parameter contract
**2026-10-22 - backend reliability increment**

Static helpers keep their first argument; instance helpers bind their receiver
separately. Explicit keywords bind by parameter name and omitted literal defaults
remain available. Invalid bindings, class methods and dynamic argument expansions
retain dependencies and report uncertainty. Helpers are never executed.

This closes a concrete parser defect seen in T5's native causal-mask helper,
without assuming its symbolic sequence length or mask rank. Engine cache identity
advances to `2026.10.22.1`; saved reports remain immutable. Measured before/after
results, isolated-snapshot checks and remaining gates are recorded in
[BACKEND_RELIABILITY.md](BACKEND_RELIABILITY.md). The narrow increment excludes
the previously uncommitted Cursor changes.

## D-069 - Establish tensor-factory rank from native callable identity
**2026-10-22 - backend reliability increment**

Factory-derived tensor facts require native PyTorch callable identity and a
supported argument form. Rank follows the supported factory contract, without
assigning concrete values to symbolic dimensions. Opaque calls, unsupported
shape changes and nullable merges cannot silently preserve those facts.

The rank increment uses engine cache identity `2026.10.22.2`. It resolves the
pinned T5 mask-rank condition without forcing the remaining symbolic length
condition. Isolated-review and combined-working-tree evidence are recorded
separately because the earlier dtype/routing drafts remain unstaged. The next
teammate gate is value-specific dtype tracking; see
[TEAMMATE_HANDOFF.md](TEAMMATE_HANDOFF.md).

## D-070 - Attach dtype evidence to values instead of every model expression
**2026-10-23 - backend reliability increment**

Source dtype conditions use established input-profile, referenced-parameter,
trusted-factory and supported-cast facts. Agreeing conditional joins retain the
fact; opaque dependencies and unknown targets cannot inherit the model's first
parameter dtype. Generic inputs without an established profile remain unknown.

This corrects the demonstrated float16-cast and boolean-mask branch failures.
The engine cache identity advances to `2026.10.23.1`. It does not establish dtype
after arbitrary normalization/module operations or certify runtime graph-edge
casts. See [DTYPE_EVIDENCE.md](DTYPE_EVIDENCE.md) and the updated
[TEAMMATE_HANDOFF.md](TEAMMATE_HANDOFF.md) for evidence and the next bounded gate.

## D-071 - Keep input-value predicates unresolved without token evidence
**2026-10-23 - top integration**

The canonical input profile establishes shape and selected dtype facts, but it
does not include token values. A static membership check such as `0 in input_ids`
therefore remains unresolved. The earlier draft treated absence of token values
as proof of absence and selected one branch; that was unsound. The integrated
reader retains conditional dependencies and reports the uncertainty.

The combined branch preserves the measured catalog, native graph contracts,
value-specific dtype evidence, and the uncommitted frontend/backend work from
the previous checkout. The catalog reports outcomes separately: a record does
not imply a complete architecture graph. See
[top integration validation](TOP_INTEGRATION_2026-10-23.md).

## D-072 - Classify GPT-NeoX components from verified native topology

**2026-10-23 - dynamic graph coverage increment**

Classify fused QKV attention only when `query_key_value` and `dense` are native
linear projections with the 3H/H dimensions, and classify the GPT-NeoX MLP
only when expansion, activation and contraction agree. The existing layer
rule then identifies decoder blocks. Reject lookalikes with mismatched
dimensions; retain all unresolved flow warnings and partial outcomes.

Advance engine identity to `2026.10.23.3`. Two pinned Pythia variants reach
complete semantic/module classification but remain partial on tensor flow.
See [GPT-NeoX role evidence](DYNAMIC_GPT_NEOX_ROLES.md).

## D-073 - Preserve checkpoint-index declarations when tensor inspection is budgeted

**2026-10-23 - inspection transparency increment**

A validated, pinned Safetensors index may exceed the tensor-record budget before
any shard header can be read. Preserve only its source URL, canonical digest,
declared tensor-name and shard counts, and the configured limit in a distinct
index-declaration field. Keep verified tensor count at zero and make no graph,
parameter, ownership, or shard-header claim. Invalid shard paths still fail.
Show the separate native-architecture blocker and avoid completed-sounding
language on artifact-only reports. Advance engine/cache identity to
`2026.10.23.4`; earlier reports remain immutable. See
[inspection failure diagnosis](INSPECTION_FAILURE_DIAGNOSIS_2026-10-23.md).

## D-074 - Expose an exact-pin publisher reference topology as partial

**2026-10-24 - DeepSeek source-topology increment**

For `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, a reviewed config digest and
publisher source hashes gate a bounded text, MoE, speculative and vision module
graph. The code is read for review, never executed. Edges describe source order,
not runtime tensor flow. Keep the validated index declaration while leaving
shard-header verification, parameter count, tensor ownership and graph shapes
unknown. Other revisions and altered configs remain artifact-only. Advance
engine/cache identity to `2026.10.24.1`; existing saved reports remain immutable.
See [publisher topology evidence](PUBLISHER_SOURCE_TOPOLOGY.md).

## D-075 - Verify oversized indexed checkpoint headers as a separate audit

**2026-10-24 - DeepSeek checkpoint-evidence increment**

The normal reconstruction path retains its 20,000 tensor-record limit. An
explicit pinned-index audit may validate each Safetensors shard header and
compare every tensor name with the index without loading weight payloads or
retaining checkpoint-wide tensor records. A result is emitted only after every
shard passes descriptor, file-size, non-overlap, index-location and aggregate
byte-budget checks. Failed runs preserve a local error/request log and cannot
publish partial counts.

The exact DeepSeek V4.1 pin verified 48 headers and 96,085 indexed tensor names
with zero weight payload bytes read. Its 508,182,659,298 tensor elements include
scale tensors and do not establish model parameter count, ownership or runtime
dataflow. The audit is separate from the immutable visitor report and publisher
reference topology. See [publisher topology evidence](PUBLISHER_SOURCE_TOPOLOGY.md)
and [the per-shard aggregate](evidence/deepseek-v41-headers.json).
Recognizing `F8_E8M0` in normal metadata parsing advances engine/cache identity
to `2026.10.24.2`; existing visitor reports remain immutable.

## D-076 - Bind a prior exact-pin header audit to new partial reports

**2026-10-24 - DeepSeek report-evidence increment**

Package the completed 48-shard audit as a digest-pinned resource. A new
publisher-reference report may cite it only if its model/revision, validated
index digest and declared counts match, every audited shard is in the pinned
repository inventory, the bundled digest pins all recorded header hashes and
sizes, and the per-shard aggregates agree. The worker does not repeat the
roughly two-minute audit inside its 120-second deadline. Existing reports are
immutable.

Expose the prior audit's 96,085 tensor names, 48 headers, 10,685,312 header
bytes and 508,182,659,298 tensor elements (including scales) as a separate
evidence category. Keep report tensor records empty, parameter count and
ownership unknown, and graph links as source reference order. Engine/cache
identity advances to `2026.10.24.3`; see
[publisher topology evidence](PUBLISHER_SOURCE_TOPOLOGY.md).

## D-077 - Audit exact publisher state-key correspondence separately

**2026-10-24 - DeepSeek checkpoint-name increment**

At the exact DeepSeek V4.1 pin, compare the full 96,085-name checkpoint index
with a finite state-key inventory from the hash-verified publisher constructors.
Reject missing or unexpected names rather than treating a shared prefix as
ownership proof. Record counts by closest reference component and normalized
pattern in a compact, inspectable evidence document. Add graph nodes for the
three source-declared root image markers omitted from the earlier topology.

The pinned converter's name rewrites leave this index's names unchanged, but
its value transformations remain unverified. The audit is offline and does not
upgrade historical reports, parameter count, shapes, dataflow or runtime
coverage. The three new graph nodes advance engine/cache identity to
`2026.10.24.4`. See [publisher topology evidence](PUBLISHER_SOURCE_TOPOLOGY.md).

## D-078 - Bind prior state-key correspondence to new partial reports

**2026-10-24 - DeepSeek report-name increment**

Package the completed exact-pin name inventory as a digest-pinned resource. Bind
it only after the prior header audit is bound and its index digest, shard/name
counts, reviewed config and source hashes, converter hash, component paths and
aggregate counts all agree. Show the prior name match in the report and closest
graph components as a separate evidence category. Do not create individual
tensor records, parameter totals, shape coverage or runtime-flow claims. Older
reports remain immutable. Advance engine/cache identity to `2026.10.24.5`.

## D-079 - Prove Mistral hidden-state ports and identify native mask aliases

**2026-10-24 - decoder static-coverage increment**

For exact native Mistral modules, source-hashed output contracts establish the
hidden-state ports of attention, MLP, RMSNorm and decoder layer after checking
their configuration, projections, parameters and eager-attention structure. A
local `mask_function` alias is recognized only when it resolves to one of two
reviewed Transformers mask factories, its source digest and call signature match,
and no dynamic argument expansion is present. The report records that callable
identity as a diagnostic; it does not claim a mask-value or runtime trace.

Fresh pinned Mistral-7B-Instruct-v0.3 and Zephyr-7b-beta metadata runs each meet
the complete static-report gate: all four coverage ratios are 1, no extraction
warnings or unknown/unmapped records, 291 verified tensor headers, and zero
weight-payload bytes read. Other model families and older saved partial reports
are unchanged. Advance engine/cache identity to `2026.10.24.6` so a new
submission does not reuse a report from the previous extractor. See
[Mistral static closure](MISTRAL_STATIC_CLOSURE.md).

## D-080 - Audit the integrated submission candidate before public release

### 2026-10-25 - submission readiness and verification

Use Van's `feat/vv` at `7ce638e` as the audit baseline after PR #57.
Application PRs now target `feat/vv`, and quality CI runs for that branch as
well as `main`; promotion to `main` remains a reviewed PR.

Repair clean-checkout test reproducibility with an explicitly historical,
metadata-only GPT-2 fixture, preserving the distinction between regression
fixtures and actual visitor analyses. Keep the new registered-module topology
scope intact while repairing compatibility and validation regressions.

AI output retains citation/numeric validation and is explicitly labeled as
interpretation whose remaining prose is not independently verified. It cannot
establish or mutate reconstructed graph facts.

The dependency scan found five advisories in certifi, filelock and setuptools;
upgrade only those locked packages to patched versions and require patched
setuptools for source builds. Record scan limitations rather than equating a
clean advisory lookup with a complete security guarantee.

No public release or submission is certified by these code changes. The listed
Vercel URL returned 404 and GitHub reported the repository private. Public
source/license visibility, hosted judging access, actual Nebius/NVIDIA inference,
video and final release evidence remain explicit gates in
[submission readiness](SUBMISSION_READINESS.md).


The bounded repair also restores the configured guest/global quota rejection
inside serialized write transactions and preserves verified checkpoint header/
name evidence in analysis summaries. Missing checkpoints cannot establish
native compatibility. Native checkpoint renaming records exact rule patterns
and source digests. Engine/cache identity advances to
`2026.10.25.2-submission-audit` so fixed fresh reports do not silently reuse
earlier cached results. Current UI reports require explicitly successful
architecture checks before claiming complete architecture.

### 2026-10-25 - Review follow-up and teammate handover

Document the exact dependency installation and test runners in `AGENTS.md`.
The first-party package imports do not represent missing PyPI dependencies;
Playwright specs require Playwright, and Python service tests require API extras.
Add a permanent concurrent quota regression covering independent SQLite service
instances, both quota scopes and both actions at a nearly exhausted allowance.
Replace browser-test non-null assertions with explicit failure guards while
preserving the responsive geometry thresholds. Record the branch order and
remaining release requirements in [teammate handover](TEAMMATE_HANDOVER.md).
Run quality CI for PRs to any base branch so the stacked dependency follow-up
receives the same checks before its final integration into `feat/vv`.
