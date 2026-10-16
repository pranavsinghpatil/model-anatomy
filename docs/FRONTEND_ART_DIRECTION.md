# Frontend art direction — October 16

## Intended use

Model Anatomy is a tool for inspecting actual model artifacts. The first screen
must accept a model immediately; the workspace must make a selected component,
its shapes, tensors and provenance readable. This revision builds on the routed
workspace and explanatory Anatomy presentation without changing their contracts.

The subject inventory is: a Hugging Face model identifier, a pinned repository
revision, a reconstructed component hierarchy, representative repeated blocks,
shallow child graphs loaded on demand, tensor names and dimensions, distinct
structural/dataflow relationships, source citations, visitor-owned history,
artifact-only outcomes, bounded metadata inspection, and honest unknowns.

## Argument and composition

1. “See how a model is built.” The actual inspection form is beside this promise,
   rather than behind a second introduction or a demonstration signup.
2. The model field accepts an ID or URL. Examples fill the field; they do not open
   prepared reports. Learn/Research explain the presentation choice.
3. Support boundaries explain static inspection and unavailable reconstruction.
   They are a disclosure, not a second screen competing with submission.
4. Recent analyses are evidence of this visitor's work, loaded from the existing
   owned endpoint. An empty visitor has an explicit empty state.
5. The selected report opens a technical workbench. Named horizontal views provide
   orientation; a component list, graph and inspector each have a distinct role.

The headline, form and real history carry the page. No customer logos, invented
metrics, screenshots of fictional reports, decorative plots or stock imagery are
used as proof.

## Reference interpretation

- [Backdoor](https://textbackdoor.com/) (redirects to Backdoor AI): decisive type
  hierarchy and a tangible product beside the introduction.
- [Fanout](https://fanout.sh/): generous scale and a clear relationship between
  technical subject matter and page composition.
- [ClickUp](https://clickup.com/): explicit workspace navigation and recognisable
  layers of product context, view choice and local controls.
- [EasyUI](https://easyui.site/components) and [React Bits](https://reactbits.dev/)
  were inspected as optional component references. This change adds no component
  dependency or copied animation; existing Radix controls and CSS suffice.

The references inform hierarchy rather than supplying product claims or a copied
visual identity.

## Skill direction and adaptations

The explicitly requested `design-taste-frontend` skill was read and applied. Its
scope says “Not for dashboards or app screens.” Its public-page direction is
therefore used for entry, with typography and visual critique adapted to the
technical workspace.

The script dealt `product-performs`, `one-motif`, and `quiet-instrument`.
`product-performs` was chosen because the real model submission and inspection
workflow can carry the design. Full-sheet seed: **1776876543**. Existing local
Inter/JetBrains Mono and the mineral-teal accent are retained. The palette is
adjusted toward paper and graphite-green surfaces, with fewer outlined boxes.

The dealt pinned reel/delayed form and logo-proof compositions are deliberately
adapted: delaying the working form would contradict direct tool use, and there
are no customer logos to substantiate a proof strip. Actual visitor history takes
that role. Marketing photo counts, video and animation quotas do not apply to an
inspection workspace. There is no claim that those marketing gates passed.

## Implementation

- Entry has an asymmetric two-column introduction and working form, stacked on
  smaller screens. Decorative step numbers, kickers and feature cards are removed.
- Workspace view links move from the vertical rail into the header. Model context
  remains distinct from local graph tools.
- Graph search and view controls have reserved space. Fit, Readable, orientation,
  density and neighborhood actions remain explicit and keyboard reachable.
- Node typography and dimensions increase together. Category labels use sentence
  case, repeated blocks remain identifiable, and literal shapes stay monospaced.
- Inspector purpose, ports and evidence use larger text and simpler surfaces.
  Report and Overview use larger headings and body text instead of repeated tiny
  labels. Existing tables and pagination remain bounded.

API, generated types, extraction, ownership, checkpoint selection, exports and
graph session memory are unchanged. No backend or GGUF implementation is modified.

## Visual review and verification

The first browser pass caught a real composition defect: the form was still
nested inside the introductory grid child, leaving the second column empty. The
DOM was corrected before further review. A mobile pass then caught a wrapped
wordmark and crowded history action; mobile header treatment was revised.

The baseline skill source lint found decorative labels, numbered steps, repeated
outlines and tiny text. Its initial browser inspector captured an unmounted SPA
with zero words, so that capture is not accepted as visual evidence. Browser
screenshots and isolated Playwright review use the mounted application instead.

### Critique rounds

**ROUND 1**

Seen in one second: the large model headline, followed by an accidental empty half.
Idea (one sentence): the real inspection form should perform the product promise.
Live thing: model input and visitor-owned analyses.
Layout words by section: introduction=split / support=disclosure / history=table.
Scores: design 5 · usability 6 · creativity 6 · content 8 → overall 5.8.
Caps applied: accidental empty ground; no marketing picture set.
Top three problems:
1. First desktop browser capture: nested form leaves an empty grid column; make
   the introduction and form siblings.
2. First 390px capture: the wordmark and history action wrap; retain the accessible
   history name while showing its icon on phones.
3. Baseline source review: tiny technical text and redundant decorative steps;
   enlarge the type scale and remove the step-card treatment.

**ROUND 2**

Seen in one second: model promise and actual form, with a clear next action.
Idea (one sentence): one direct entry feeds a readable inspection workbench.
Live thing: actual form state, history and selected component.
Layout words by section: introduction=split / support=disclosure / history=table.
Scores: design 6.5 · usability 8 · creativity 6 · content 8 → raw overall 7.0;
marketing-rubric cap 6.0.
Caps applied: the marketing inspector's picture floor remains unmet by design.
Top three problems:
1. `graph-1440-light.png`: CSS auto-placement creates an unnecessary second header
   row; assign the intended desktop row explicitly.
2. `graph-390-light.png`: the Report label is clipped; omit navigation icons on
   narrow screens so all four view names fit.
3. Mounted skill audit: prose size, long measures and decorative dividers;
   set entry prose to 16px, bound measures, and remove decorative rules.

**ROUND 3**

Seen in one second: a direct model-inspection tool and its selected component.
Idea (one sentence): work from a real artifact to structure, tensors and evidence.
Live thing: submission, visitor-owned history and contextual inspection.
Layout words by section: introduction=split / support=disclosure / history=table.
Scores: design 6.5 · usability 8 · creativity 6 · content 8 → raw overall 7.0;
marketing-rubric cap 6.0. This is below the skill's marketing-site target of 7.5;
no award-level score or universal skill-gate pass is claimed.
Caps applied: no photography/footage or staged entrance motion is introduced into
the tool to satisfy marketing quotas.
Remaining constraints and fixes:
1. The 320px regression exposed a long history identifier escaping its flex
   container; constrain the inner text width and rerun the full width matrix.
2. Fit mode necessarily reduces text on a large graph; Readable and selection
   provide complete details without shrinking the entire interface.
3. The real API is stopped; live report acceptance remains blocked separately
   from the isolated browser checks.

Final entry source lint: **0 FAIL, 0 WARN**. The explicit lint exceptions document
the boundary of a real history table and a visually hidden accessible name.
The final mounted inspector reports **1 FAIL, 4 WARN**: the unmet picture floor,
no footage, little entrance motion, an outlined control, and ten sizes across
prose, controls and metadata. The picture/footage/motion suggestions are marketing
criteria deliberately excluded from this direct-use tool. The outlined control
is interactive, and the smaller metadata sizes remain distinct from 16px prose.
Headline wrapping, body size, paragraph measure and decorative-divider findings
are resolved at 1440, 768 and 390px. Captures were inspected at fold and full-page
levels; source lint alone is not treated as visual acceptance.

TypeScript, formatting and the production build pass. The full Playwright run
completed with **132 passed, 2 failed, 28 skipped** in 3.7 minutes. Both failures
were the same long-identifier overflow in desktop/mobile configurations. The
correct child-span constraint fixes that actual intrinsic-width problem; the
two affected matrix tests then passed across 320, 390, 768, 1024, 1440 and 1920px
in both themes. Thus **134 distinct deterministic cases are verified across the
broad run and targeted rerun**, including axe accessibility, reduced motion,
200%-equivalent reflow, graph navigation, panel resizing, clipboard/export failure,
stale requests and tensor pagination. No unresolved deterministic failure remains.

The 28 opt-in live cases were skipped, not counted as passes. Workspace fixtures
are confined to Playwright routing and project an existing engine artifact into
the HTTP contract; they are never a production fallback. New screenshots cover
desktop/tablet/phone graph states and inspector, artifact and report layouts.

Fresh live model acceptance is currently blocked: automatic approval review
rejected resuming the stopped API and worker because of an earlier restriction
on backend restarts. The frontend preview is running independently and displays
the actual unavailable-service state. No process kill, database reset or alternative
backend was used to bypass that decision.
