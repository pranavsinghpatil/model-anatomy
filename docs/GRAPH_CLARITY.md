# Graph clarity

This increment refines the architecture graph as an inspection tool. Reported
nodes, relationships, ports, tensor metadata and evidence remain authoritative.
It changes their presentation without changing the reconstruction engine or
stored architecture reports.

## Presentation contract

- Draw the layout engine's orthogonal routes, rather than recomputing unrelated
  curves after positioning nodes. Reported port names identify attachment points.
- Keep node cards and attachment points stationary during hover and selection.
  Borders, line weight and focus outlines communicate interaction.
- Use readable relationship and component names, retaining original identifiers
  and evidence in connection details. Unknown facts remain unknown.
- Reveal connection details through pointer or keyboard selection. Labels must
  remain readable without covering component cards in either orientation.
- Preserve the existing component navigator, nested graph navigation, saved
  viewport, compact/detailed cards, and explicit whole-graph/readable views.
- Keep disconnected inventory separate. A missing reported relationship stays
  missing; visual organization must never manufacture dataflow.

The visual direction is a restrained technical canvas: neutral surfaces, clear
typography, fine grid marks, visible connectors and one focus accent. Decorative
glows and card motion compete with tracing and are removed. Only the relevant
inspection and accessibility guidance from the requested design skills applies;
landing-page structures and cinematic scroll effects are outside this change.

## Verification

Validated on 2026-10-30 against code commit `940af12`. The canonical
[quality run](https://github.com/pranavsinghpatil/model-anatomy/actions/runs/41000436658)
passed all jobs:

- Frontend TypeScript/production build, formatting and generated API contract.
- Full Playwright suite with two workers: **270 passed, 36 explicitly skipped,
  zero failed** across desktop and mobile.
- Backend compile, Ruff and pytest: **855 passed, 23 skipped**.
- Container build, configuration checks and pinned-model container acceptance.

The browser checks cover exact named-port attachments, orthogonal routes and
label clearance in both orientations, keyboard selection, stale-node/active-edge
priority, endpoint visibility after relayout and Readable, real 390px layouts,
light/dark contrast, nested navigation and saved viewport restoration. The
existing long responsive sweep is isolated by viewport; its routes, themes,
assertions and per-case timeout are preserved.

Local Windows runs were not clean: the initial full run had 253 passes, 36 skips
and seven failures. The corrective five-spec run had 87 passes and 15 loading-time
failures. Presentation/contrast failures were corrected and passed in CI. Local
failure snapshots include `Opening workspace` before graph mounting and
`Arranging architecture`; their exact host or loading cause remains unconfirmed.
These local failures are recorded rather than described as a passing local suite.
The final local production build and diff checks passed as well.

Browser fixtures exercise UI rendering; skipped live specs do not establish broad
live-model acceptance. No populated runtime or database profile was replaced.

## Fixture previews

Reviewed light, dark horizontal and mobile views. These screenshots use the
self-contained contract fixture, not a live model inspection.

- [Light connection inspection](evidence/graph-clarity/light.png)
- [Dark horizontal connection](evidence/graph-clarity/horizontal-dark.png)
- [390px connection framing](evidence/graph-clarity/mobile.png)

## Limits

This is a static architecture visualization, not a runtime execution trace.
Readable labels do not establish additional semantic facts or repair missing
relationships in an inspected report. Test fixtures exercise drawing behavior;
they are not served as application analyses.
