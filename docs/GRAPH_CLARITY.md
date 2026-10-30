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

Validation is pending. Record completed deterministic checks and screenshot
review here before the PR is marked ready.

## Limits

This is a static architecture visualization, not a runtime execution trace.
Readable labels do not establish additional semantic facts or repair missing
relationships in an inspected report. Test fixtures exercise drawing behavior;
they are not served as application analyses.
