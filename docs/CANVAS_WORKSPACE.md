# Canvas-first inspection workspace

The user's usability review rejected a recolored version of the old layout and
the separate landing/start screens. The replacement entry accepts a model
immediately, shows explicit visitor-owned continuation, and explains the next
steps without a separate marketing route. Both experience modes reach the same
architecture. Backend extraction, reports, ownership and exports remain intact.

## Reading and operating the graph

The workspace uses a persistent tool rail, compact model context, and focused
Architecture, Overview, Artifacts and Report views. Input/output markers differ
from module cards. Repeated components show their real repetition count and an
explicit action for their fetched child graph. Native module paths distinguish
duplicate class names without asserting unverified semantic roles.

Compact and detailed node layouts use actual ELK dimensions, rather than shrinking
a large fixed card. Fit, readable focus, direction, zoom readout, current-graph
search and direct-neighbor focus provide distinct operations. Selection, graph
path, viewports, node density, direction, connection labels and panel sizing are
retained during the session. Missing edges remain missing: unconnected components
occupy a separate inventory shelf, and relationship inspection still exposes the
original ports, tensor shape, type and evidence.

Artifact-only reports lead with their verified result and next checkpoint or
component action. Full-width file/tensor tabs use independent bounded pages.
Configuration, identity and raw evidence are disclosed on demand. Failure recovery
prefills the submitted model for editing; progress uses actual server stages.

## Baseline verification, 2026-10-14

The restored baseline passed TypeScript and the production build. The complete
deterministic run passed 106 cases; its last two failures were ambiguous exact-text
pagination locators. After correcting that test locator, both affected desktop
and mobile cases passed. This accounts for all 108 deterministic cases, including
the responsive theme/mode matrix, keyboard panels, edge selection, graph context,
long identifiers, artifact-only behavior, cancellation and failure recovery.
The 26 opt-in live cases were not enabled in that deterministic run and are not
claimed as fresh runtime evidence. No fixture is served by the application.

The browser preview is port 5173, proxying the existing API on 8001. Port 5187 was
an obsolete isolated preview and should not be used for this increment.

The subsequent Anatomy increment completed a full deterministic run with 128
passing cases and 28 opt-in live cases disabled, plus two passing short-height
reflow checks. A separate fresh SmolLM2-135M run passed on desktop and mobile,
including lazy nested graphs, actual tensor/evidence inspection and JSON export.
See [ANATOMY_ILLUSTRATION.md](ANATOMY_ILLUSTRATION.md) for precise source boundaries
and validation scope. This supersedes the baseline's narrower verification record.
