# White inspection workspace

The October 16 revision replaces the colored, stacked workbench layout with a
white, canvas-first tool. The entry is a single readable column containing the
actual model form, support disclosure and visitor-owned history. There is no
separate marketing page, fabricated graph or seeded analysis.

## Hierarchy

The workspace has two compact global rows: identity/actions (48px on desktop)
and plain view tabs (40px). Model
dimensions live in a keyboard-operable Model details disclosure rather than a
permanent metrics strip. Architecture adds one local bar for its path, component
list, presentation and search. Fit, Readable, zoom and advanced options belong to
a bottom control dock. The drawing occupies the remaining space; controls have
reserved layout space and do not cover the selected component.

The component navigator starts closed for a new analysis. Its list alternative
remains directly available, with desktop panel resizing and a drawer on smaller
screens. Existing session choices are preserved. The inspector opens on
selection; it avoids a repeated heading above its Details/Questions controls.
Component/Tensors/Evidence retain separate accessible tabs. General type guidance
starts collapsed in Research and expanded in Learn. Raw paths remain disclosures.
Port names, shapes and dtypes use simple aligned rows instead of nested boxes.

The control pass replaces the wall of individually outlined graph buttons with
quiet action groups, consistent rounded controls and visible focus/press states.
Menus, inputs and tabs share 180ms feedback, disabled states and reduced-motion
handling. Entry joins its model field and primary action and uses short example
labels while retaining complete model IDs in accessible names. The minimap uses
visible node outlines; its smaller frame keeps attention on the main graph.
Advanced options anchor to the dock and wrap within the canvas on phones.

The white scheme is the default for visitors without an appearance preference.
Explicit Light, Dark and System choices remain available and persistent. The
current user's requested Light preference is applied through the actual control.
Status colors continue to describe real outcomes; they are not a page-wide theme.

| Token | Light | Dark |
| --- | --- | --- |
| Background / surface | #F8F9FB / #FFFFFF | #141518 / #1B1D21 |
| Primary text | #17191C | #F1F2F4 |
| Secondary text | #5C626B | #AEB3BC |
| Primary action | #222B35 | #DCE4F0 |
| Border | #E7E9ED | #35383F |
| Secondary surface | #F6F7F8 | #24272D |

## References and skill application

The user's [Fanout](https://fanout.sh/), [Backdoor](https://textbackdoor.com/)
and [ClickUp](https://clickup.com/) references were revisited. White space, strong
text hierarchy and a clear distinction between navigation and local actions are
applied to the actual tool. No reference's claims, ratings, paid components,
photographs or decorative effects are copied.

The requested `ui-ux-pro-max` skill was read. Its design-system search was run
with the bundled Python 3.12 in UTF-8 mode after the native Python 3.11 failed on
a skill-script f-string. The generated Minimal Single Column pattern fits entry;
its unrelated vibrant/dark palette recommendation conflicts with the user's
explicit white direction and is not adopted. UX/React searches informed semantic
controls, composition, disclosure, keyboard focus, contrast and responsive layout.
Existing Inter/JetBrains Mono remain locally bundled; no dependency is added.

## Validation boundaries

New checks cover white appearance despite a dark OS preference, opt-in System
appearance, available drawing height, disclosure focus/escape behavior, and
Learn/Research guidance without losing selection. Graph layout checks cover
header/dock separation, readable scale, nested navigation and mobile operation.
The existing regression matrix covers both themes, long identifiers, six widths,
200%-equivalent reflow, accessibility, failure handling and bounded pagination.

Production requests still use the existing API and visitor ownership rules. Test
fixtures remain confined to Playwright interception. Real-report acceptance uses
the existing local service; the previously recorded cold 360M metadata timeout
is not treated as fixed by this layout revision. See the PR for final run evidence.

## Verification record — October 17

- Production typecheck/build and complete frontend formatting pass.
- The broad run executed 170 cases: 136 passed, 28 opt-in live cases skipped,
  and 6 failed. Three failures used earlier always-open navigation/settings or
  the former generic connection heading; the mobile connection test needed an
  explicit Fit before hit-testing its offscreen edge. Those user steps were
  updated without dropping graph-bounds, provenance or navigation assertions.
  Two other desktop cases timed out during page/report loading and passed in
  isolation. Targeted reruns verify every failed case: 142 distinct deterministic
  cases are verified across the broad and targeted runs, not a claimed clean
  single broad run.
- All 8 real desktop/mobile cases in `live.spec.ts` and `live-anatomy.spec.ts`
  pass in Learn and Research, covering entry, submission, actual graphs, lazy
  nested graphs, tensors, provenance and JSON download. These reuse real cached
  reports at pinned revisions. SmolLM2-135M has 134,515,008 parameters at revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; SmolLM2-360M has 361,821,120
  parameters at `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`.
- Responsive/theme/mode matrices cover 320, 390, 768, 1024, 1440 and 1920px;
  six-width accessibility scans, long labels, 200%-equivalent reflow, reduced
  motion, clipboard/export errors, stale requests and pagination are included.

API health confirms analysis, storage, database and worker availability. Optional
AI remains unavailable and is presented explicitly. The earlier cold 360M
metadata timeout remains outside this frontend change. No API, IR schema,
reconstruction, GGUF or ownership code changes are included.

## Compact scale and motion — October 17

The follow-up makes the interface dense at normal browser zoom, without CSS
zoom or viewport scaling. Desktop entry headings are at most 36px (previously
60px), ordinary controls use 32px heights (previously 44px), and the joined model
field/action is 46px overall. Section spacing is reduced so the form and real
recent analysis fit together in an 800px-high laptop viewport. Touch controls
retain 44px targets. Inputs retain 16px text on narrow mobile layouts.

Workspace identity/actions use 48px and view navigation 40px. Graph controls
respond to their canvas container, including when a docked inspector narrows it.
On phones, search and panel actions share a row; footer metadata shares the
Options row. This saves space without losing labels, navigation or touch targets.
Ports now align their semantic name, shape and dtype on one row, allowing long
values to wrap instead of overlap.

Fanout's home and AI curriculum workspace were inspected visually, including its
compact navigation, quiet surfaces and typography. Smooth UI's
[copy feedback](https://smoothui.dev/docs/components/button-copy),
[toggle](https://smoothui.dev/docs/components/morph-toggle) and
[motion guidance](https://smoothui.dev/docs/guides/animation-best-practices) informed independently
implemented CSS interactions: a moving selection indicator, icon crossfades,
short menu/drawer transitions and panel entry feedback. No component source was
copied and no dependency was added. All motion honors reduced-motion settings.

New regressions check that entry and history fit together at 100% zoom, keyboard
selection aligns the moving indicator, reduced motion removes transitions, and
laptop graph actions remain clear of search. Early development-server runs were
stopped after page/layout stalls under local memory pressure; final verification
uses the built production frontend with one browser worker. The initial
production pass also caught excessive mobile toolbar rows and a history row
slightly below its height budget; both layouts were tightened without relaxing
the assertions. The final production run passes all 38 focused checks in one
run with one worker. Typecheck/build, complete frontend formatting and diff
checks also pass. Earlier full-suite evidence above is historical; this
follow-up does not present it as a new full-suite run.

Real submissions on temporary production-preview port 5175 were rejected with
HTTP 403 because it is not an allowed submission origin in the existing local
profile. Live acceptance uses the configured application port 5173; no origin
or backend security settings were changed to accommodate the test server.
