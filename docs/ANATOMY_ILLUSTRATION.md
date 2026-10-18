# Anatomy presentation

The technical Graph remains the default inspection surface. Anatomy is an optional
isometric figure constructed locally from the same currently fetched shallow graph.
It introduces no extraction, inference or API contract. Graph and Anatomy share
the actual selected node, component/tensor/evidence inspector and child navigation.

## Source boundaries

Labels, parameter counts, shapes, repetition counts, component kinds and child IDs
come from the generated API response. The layout ranks only returned `dataflow`
edges. The drawing shows those links, without motion or simulated values. Other
relationships retain their original types and evidence in Graph. Disconnected or
cyclic components are placed in an inventory area; placement supplies no missing
edge. A graph with no dataflow is explicitly titled Component inventory.

There are at most 24 illustrated components. Selecting another reported component
includes it in that bounded set; further components remain available through the
Components list and technical graph. Repeated blocks display their reported count
but draw at most three representative inserts. Geometry never encodes vocabulary
size, individual neurons, head count, tensor values, activation or hardware layout.
Null output shapes and parameter counts remain Unknown; zero remains zero.
Artifact-only reports have no Architecture or Anatomy presentation.

## Original mechanism vocabulary

| Recorded kind | Illustration | Intended explanation |
| --- | --- | --- |
| Input / output | Receiving trays | Boundaries of the current graph |
| Embedding | Lookup cartridge | Maps identifiers to representations |
| Transformer / decoder block | Cassette rack | A representative repeated structure |
| Attention | Manifold and round hub | Relates representations; not an attention-weight plot |
| Feed-forward / gated MLP / projection / head | Finned banks | Transforms representations |
| Normalization | Calibrated drum | Normalizes a representation |
| Rotary position embedding | Indexed wheel | Applies positional structure |
| Add / residual | Two-inlet junction | Combines values; report links remain authoritative |
| Dropout | Fixed screen | General kind metaphor, never a sampled runtime mask |
| Other kinds | Sealed case | Internal operation has not been illustrated |

Mechanisms are chosen from exact kind allowlists. Model names, class-name guesses,
module paths and neighboring nodes never select a known mechanism. Source labels
remain visible in the shared readout and inspector. General purpose copy is the
existing authored Component kind guide, not a generated model-specific assertion.

## Interaction and rendering

Click or tap an exposed part to open its actual inspector. Arrow keys/Home/End
choose components; Enter/Space opens the inspector. Previous/Next and the Components
list provide alternatives to facet targeting. Explore inside requests the returned
child graph on demand. Evidence uses the existing evidence tab. Switching to Graph
restores its viewport, density, direction and selection; switching mode preserves
the same context. Each graph retains its own separation, rotation and zoom state.

Separate parts is a short construction transition, not execution progress. The
range control exposes the pose directly. Reassemble/Escape, rotate, zoom, fit and
reset have distinct actions. Reduced motion applies poses immediately. An
IntersectionObserver and page visibility cancel animation while hidden/offscreen;
there is no perpetual animation or added animation dependency.

Box and cylinder faces are projected in world coordinates with back-face culling
and depth ordering. The mounting bench paints behind all components to prevent a
large support plane's centroid from occluding distant parts. A fixed frame covers
both assembly poses across the allowed camera angles. Theme tokens define materials
and one selected component highlight. A scene error offers the technical graph.

Design reference: [Ryan's Anatomy skill](https://github.com/wheresryan22/anatomy/blob/main/SKILL.md).
The implementation and geometry are original. No external kit scripts, gallery
submission, assets, remote drawing call or private model upload are used.

## Verification

On 2026-10-14 the focused deterministic Anatomy suite passed 12 desktop/mobile
cases. It checks actual SVG rasterized paint, selection and inspector context,
keyboard/touch operation, separation/reassembly and stable framing, camera reset,
reduced motion/offscreen animation stability, lazy child loading, unknown shapes,
bounded repetition, graph-free artifact results and six widths in both themes
(320, 390, 768, 1024, 1440 and 1920px), including axe scans. The route fixture comes
from the repository's GPT-2 extraction artifact and is never served by the app.

The final complete deterministic run passed **128 cases**, with 28 opt-in live
cases disabled. Two additional desktop/mobile reflow checks passed at a 720×480
CSS viewport and 2× raster scale (the layout dimensions of a 1440×960 viewport
at 200% zoom). This tests equivalent reflow, not a browser-chrome zoom setting.
Both themes, the evidence inspector and the assembly actions remain reachable;
the assembly owns vertical scrolling when the available height is short.
TypeScript, Prettier and the production build also passed.

Separate fresh local acceptance passed **2/2 desktop/mobile SmolLM2-135M cases**
against the existing API/worker on 8001. Each submitted a visitor-owned analysis,
switched modes without changing the analysis, opened the reported representative
block lazily, selected a real part, inspected evidence and owned tensors, and
downloaded JSON after asserting its revision equals the report revision. The live
report identifies its feed-forward component as `gated_mlp`; tests use that actual
kind. On mobile, selecting a facet opens Inspector, whose Evidence and Tensors
tabs are the next action. These are separate from the fixture regression run;
other architecture families were not rerun for this increment.

A separate manual GPT-2 submission completed as analysis
`87b00994d86d452b87b08a53490f7029`. Its real graph, nested attention, shapes and
inspector were visually checked in the browser, with desktop dark, desktop light
and mobile screenshots saved outside the repository. No report or screenshot is
bundled as application data. Fixture evidence alone does not establish extraction
support. The live spec saves a named JSON evidence file within its own result
directory so future runs retain IDs/revisions independently of the shared HTML
report, which concurrent test invocations can replace.
