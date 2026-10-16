# Model Anatomy inspection workspace

The redesign serves two tasks using the same inspected data: learning a component
and quickly reaching its tensors and provenance. It adds no backend or IR contract.

## Navigation and information hierarchy

After the user's usability review, `/` and `/app` now open one tool entry screen. It accepts
a public HF ID, repository URL, or explicit checkpoint URL and lists only the
visitor's analyses. Recent analyses require an explicit continuation action.
`/app/analyses/:id?view=overview|architecture|artifacts|report` identifies the
selected analysis and primary view; old `/?analysis=...` links redirect.

Research and Learn both default to Architecture. Once a workspace
view is selected, changing mode preserves the view, graph path, selection,
inspector tab and viewport. Session memory is keyed by analysis and graph; theme
and mode alone persist in local storage. No reports are placed in local storage.

Overview presents identity, applicable dimensions and next actions. Architecture
provides an accessible component list, current-graph search, graph, connections
and inspector. Artifacts owns file and tensor pages, configuration and evidence.
Report separates coverage, incomplete checks, warnings, method scope and identity.

## Annotated wireframes

The original [desktop and mobile wireframes](design/workspace-wireframes.svg) record
the first design. The implemented tool-first revision is described in
[CANVAS_WORKSPACE.md](CANVAS_WORKSPACE.md); it removes the marketing/start split.

At 1280px the component navigator can open at 240px; it starts collapsed for new
analyses. The inspector opens at 360px.
Keyboard-operable separators resize the panels while preserving graph state.
At 1024–1279px a Radix dialog contains the navigator while the inspector remains
docked. Below 1024px selection opens a separate inspector view with Back to graph.
Panel scroll areas contain long details. Graph controls reserve their own space.
The full submission form appears on the unified entry screen. The October 16
[art-direction revision](FRONTEND_ART_DIRECTION.md) moves workspace views into
horizontal navigation and improves typography throughout the inspection screens.
The subsequent [white workspace revision](WHITE_WORKSPACE.md) replaces the stacked
toolbars with a single local graph bar and bottom dock.

## Design tokens and component states

| Role | Light | Dark |
| --- | --- | --- |
| Background | #FFFFFF | #141518 |
| Surface | #FFFFFF | #1B1D21 |
| Text | #17191C | #F1F2F4 |
| Secondary text | #5C626B | #AEB3BC |
| Accent | #222B35 | #DCE4F0 |
| Text on accent | #FFFFFF | #17191C |

Inter Variable and JetBrains Mono Variable are bundled locally through Fontsource.
Spacing follows a 4px scale. Surfaces use thin borders and restrained radii.
Light appearance is the default. Explicit Light/Dark/System choices are remembered.
Focus rings, selected labels, status text and dashed structural connections
supplement color. Reduced motion disables CSS animation and animated graph moves.

Buttons have normal, hover, focus, disabled and pending states. Radix handles
keyboard tab navigation, menu dismissal, dialog focus trapping and focus return.
Clipboard success follows the actual promise; rejection gives a manual fallback.
Exports show failures without replacing the report. Empty lists, absent facts,
unknown counts, request failures and expired analyses are explicit.

## Domain boundaries

Only completed/partial reports expose architecture. Artifact-only results open
Artifacts, omit Mermaid and architecture questions, and distinguish tensor
elements from unique model parameters. Failed, unsupported, cancelled and timed
out results have distinct recovery copy. New submissions require confirmed service
health; existing reports remain accessible during degradation.

Progress names and stages come from the server. There are no percentages or time
estimates. Nested graphs load on demand. Artifact and component tensor pages stay
bounded at 50 records. Graph connections preserve relation types and evidence;
repeated blocks open only their reported representative graph.

Learn guidance is authored general information for established component kinds.
Model-specific labels, shapes and evidence come from the API. Unknown kinds retain
an explicit unknown-purpose statement. Shape symbols are guidance, not inferred
numeric dimensions. Artifact-only guidance contains no architecture learning steps.

## Reference and dependency choices

Free Radix composition patterns from [shadcn/ui](https://ui.shadcn.com/docs/components/radix/dropdown-menu)
inform menus/dialogs/tabs. [React Router](https://reactrouter.com/start/declarative/installation)
provides route navigation; [react-resizable-panels](https://github.com/bvaughn/react-resizable-panels)
provides panel resizing. Tailwind v4 shares semantic tokens with dedicated graph
CSS. See `frontend/THIRD_PARTY.md` for attribution. The application imports no
paid component, shader, orb, particle effect, animation library or mock model data.

## Verification record

The primary checkout's `frontend/` now uses this routed implementation. The old
single-page entry and stylesheet have been replaced. Integration preserves the
newer model-routing work: declared repository profiles, explicit pipeline
checkpoint selection, generated API types and staged head dimensions. The UI
decision is D-028 after reconciling the independent routing decision D-027.
The integration's pipeline-selection and staged-dimension regressions pass on
desktop and mobile. Its first local full run passed 95 checks with one warning-view
deadline; that unchanged warning check then passed all six repeated runs. The
integration PR records the final CI result separately from the earlier isolated
redesign acceptance below.

Verified on Windows with Chromium, 2026-10-13:

- Production build (including TypeScript), formatting, and `git diff --check` pass.
- Regenerating the actual API's OpenAPI types produces no contract diff.
- The complete deterministic Playwright run passes **92 tests** in desktop and
  mobile projects. Its 14 live cases are deliberately skipped without
  `E2E_LIVE_API`; all **14 live cases also pass** in separate real-service runs.
  A final combined real-service run passes all **106 cases** with no skips.
- Architecture is checked at 320, 390, 768, 1024, 1440 and 1920px in both modes
  and themes, with 12 axe scans reporting zero violations. Entry/report layouts
  and long model IDs, tensor names and artifact URLs also pass every target width
  in both themes. A separate regression checks 200% CSS zoom at a 1440px viewport.
- Keyboard component/connection selection, drawer focus return, tabs, panel
  resizing, nested viewport restoration, reduced motion, late responses,
  cancellation, recovery, clipboard/export failures and bounded tensor pagination
  pass. Selecting a new component clears prior citation overrides and connections.
  The minimap renders the actual node count using the fixed card dimensions.
  Late submissions respect history changes during concurrent route transitions;
  affected navigation, graph and resizing checks also pass 30 repeated cases.
- Actual SmolLM2-360M jobs complete the Learn and Research submission → component
  → tensors → evidence → JSON journey on desktop and mobile. The pinned report
  records 361,821,120 unique parameters.
- Actual ViT jobs retain patch/CLS nesting, 86,567,656 parameters, tensor inspection
  and ZIP export. Kolibri and Qwen LoRA remain honest artifact-only results;
  selected Bonsai GGUF exposes 851 tensor descriptors without a graph. Exported
  artifact metadata records zero weight payload bytes read.

Deterministic regressions use real extraction artifacts only as test fixtures.
Live acceptance uses separate visitor-owned HTTP/worker jobs, with no route
interception or seeded application data. Backend behavior and API schemas remain
unchanged. The optional AI service was unavailable locally; its unavailable state
and citation interactions are covered, but no new live AI inference is claimed.
These checks cover Chromium and local services, not a hosted deployment,
cross-browser certification or a full assistive-technology audit.
