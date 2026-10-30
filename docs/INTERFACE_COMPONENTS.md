# Sourced components and richer workspace surfaces

Subsequent user direction on October 17 removes Learn/Research and applies a 75%
page scale. See D-035 and the frontend README. The audit and validation below
record the preceding component-refinement increment, when both modes existed.

## Scope

The October 17 refinement keeps direct model submission and the compact graph
workspace. It addresses the user's request for visible use of free prebuilt
components, a patterned background, and clearer visual separation. It does not
change reconstruction, ownership, pagination, export contracts, or GGUF behavior.

## Audit findings and response

The entry screen was captured and inspected during this run at normal browser
zoom. Its primary form was usable, but the form, examples and history shared
nearly the same flat background. Example selection had no persistent visual or
accessible state. The existing graph controls and contextual inspector remain
the interaction foundation; their layout is not replaced with marketing blocks.

- A static Magic UI DotPattern establishes a quiet technical backdrop. It is
  decorative, hidden from assistive technology and cannot intercept input.
- A bounded, raised submission surface separates the task from the background.
  Service availability now sits beside the field label, reducing vertical space.
- Example buttons expose `aria-pressed`, reflecting the actual input value. Editing
  the input removes the selection when it no longer matches.
- SmoothUI Fluid Hover Highlight follows workspace navigation on pointer entry
  and keyboard focus. It does not replace active-route labels or the underline.
- SmoothUI Button Copy provides idle, loading, success and error feedback. It
  copies immediately, retains a visible manual fallback on failure, and resets
  after success. Changed values remount the control to discard stale feedback.
- Graph nodes use labeled category cues and a restrained stack treatment for
  reported repetition. Unknown kinds stay neutral. Reported shapes and graph
  geometry are preserved. Evidence disclosures receive compact bounded surfaces.

Pattern contrast is deliberately low; text and technical records remain on solid
surfaces. Both themes share the same information hierarchy. Reduced motion turns
off positional travel, icon blur and loading rotation. Motion's DOM animation
features load separately; no continuous animation or WebGL background is added.

## Component provenance

Actual source is adapted from the free registries rather than imitated solely
with independently written styles:

| Component | Upstream | Adaptation |
| --- | --- | --- |
| DotPattern | Magic UI | Static SVG pattern, unique IDs, theme color; optional glow and random measurement removed |
| Button Copy | SmoothUI | App value/label interface, visible clipboard failure, no artificial wait, lazy Motion features |
| Fluid Hover Highlight | SmoothUI | Rect API, route links and focus support, restrained spring, immediate reduced-motion state |

Pinned source revisions and full MIT notices are in
`frontend/THIRD_PARTY_NOTICES.md`. Notices are also shipped under
`frontend/public/licenses/`. No Pro component is included.

## Evidence limits

The in-app browser initially reopened on a connection-error document. Browser
security blocked control until the user reloaded and the app recovered. The
existing API, worker and Vite preview were resumed using the user's earlier
authorization. No security setting, origin rule, data profile or quota was changed.

After recovery, current-run screenshots were saved and inspected for six steps:
entry, architecture, selected component, tensor pagination, evidence, and export.
The live visitor-owned SmolLM2-135M report showed 30 repeated decoder blocks,
270 attributed tensors, a direct `named_modules` locator and separate reconstructed
evidence. Tensor pagination advanced to 51–100 of 270. Switching Learn to Research
preserved the component, evidence tab and 120% viewport. Both light and dark
evidence views were inspected. Report JSON downloaded and parsed successfully
(1,112,325 bytes). These findings apply to this recorded report, not every model.

The scoped audit follows continuation of an existing analysis. It does not
establish cold-model reliability, screen-reader compatibility or full WCAG
conformance. At fit scale, the whole graph is an orientation view; selecting a
component provides readable detail. The separate component-list alternative remains
available. Full responsive and error-state checks use deterministic fixtures;
real local acceptance is separate evidence.

## Validation

- TypeScript and production build pass. Motion features are a separate 46.70 kB
  chunk; the initial application chunk is 487.03 kB, below Vite's 500 kB warning.
- Formatting and diff checks pass.
- Eight real local acceptance checks pass (47.5 seconds): direct entry, explicit
  continuation, Research/Learn submission, component inspection, lazy nested graph,
  Anatomy/evidence and export on desktop and touch projects. These reuse real
  exact-revision cached reports; they do not resolve the earlier cold-360M timeout.
- 44 affected Playwright checks pass across desktop and touch projects, including
  320/390/768/1024/1440/1920 layouts, themes, modes, axe scans, clipboard failure,
  keyboard copy recovery, panel resizing, graph details, artifact pagination,
  GGUF selection, export failure, stale requests and artifact-only exports.
- The first 54-case run passed 51 checks and exposed a touch-layout height
  regression plus two instances of a missing fixture in the new clipboard test.
  The spacing was corrected without relaxing the density assertion; the clipboard
  test now navigates to the configured architecture fixture. All affected checks
  pass in the 44-case rerun. Fourteen Anatomy checks and two reflow checks also
  passed before the final density/menu corrections; they were not rerun afterward.
- The previous CI export-menu failure reproduced on both device projects. A
  closing animation retained Radix's focus scope during rapid export reuse.
  Closing menus now unmount immediately while opening retains its transition.
  All five artifact-only export formats now pass on both device projects.
