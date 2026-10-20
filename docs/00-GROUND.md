# Project Ground

## Promise

**Model Anatomy reconstructs a specific open-weight Hugging Face model into a semantic, tensor-aware architecture graph that a developer can inspect instead of reverse-engineering it manually.**

The input is a model ID or URL. The output is a model-specific technical representation: hierarchy, traceable relationships, components, tensors, dimensions, ownership, and evidence.

## Scope for the hackathon release

Build one reliable path for a decoder-only Transformer, beginning with [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) after revision pinning. A judge can paste the model, see a generated graph, select a component, and trace the displayed facts to artifacts.

The October 9 expansion direction is family-by-family coverage across text,
vision, multimodal, audio and video, with model size tested separately. Preserve
the working text path while closing documented gaps in small PRs. A native model
loading successfully does not establish complete semantic support. See
`SYSTEM_GAPS.md` for current evidence and the next acceptance gates.

The October 12 compatibility increment adds artifact-layout/native-task routing,
component selection and pinned partial reports across text, image, audio, video
and image-text. Their remaining semantic gaps are acceptance work, not grounds
for a universal support claim. See `DYNAMIC_ROUTING_ACCEPTANCE.md`.

The October 19 direction adds measured compatibility attempts across 120 pinned
repositories, followed by fixes and family-specific acceptance plans. Keep complete
static graphs, partial anatomy, graph-free artifact inspection and failures
separate. A broad benchmark does not establish universal model support. See
`HUNDRED_MODEL_BENCHMARK.md` and `MODEL_FAMILY_STRATEGIES.md`.

The [hackathon release](https://nebiusglobalaihackathon.devpost.com/) must run on Nebius Token Factory or Nebius AI Cloud and use an NVIDIA open-source model. The integration is an optional, evidence-bound explanation layer: it may summarize or answer questions about the completed reconstruction, but must never infer the graph or overwrite evidence.

## Non-goals for this release

- claiming universal support for every architecture or custom codebase without evidence
- a generic chatbot, model search tool, or hand-drawn diagram editor
- checkpoint forensics, comparison, or anomaly detection as a dependency for the first demo

## Product boundary

The application serves visitor-owned analyses and a shared catalog of metadata
from actual saved inspections. A request first checks the durable database
registry for a compatible report; a miss inspects real artifacts and stores the
result. Catalog access creates a new visitor-owned analysis for the selected
pinned report. See [Database catalog](DATABASE_CATALOG.md). There is no
prepared-example API, seeded graph, static report shortcut or demo fallback.
Exact-revision caching reuses real inspected reports; it is not mock data.
Historical CLI outputs may be retained as extraction evidence and test inputs,
but are never served as application analyses.

Safetensors is tensor evidence, not a computation graph. The reconstruction fuses configuration, executable structure, module hierarchy, graph tracing where possible, source, and tensor metadata. Unknown or untraceable facts stay explicit; the system never fills gaps with plausible architecture.

When native architecture reconstruction is unavailable, the web application may
return a separate **Artifacts inspected** result. This shows the pinned repository
file inventory and, where checkpoint selection is unambiguous, verified tensor
headers. It contains no architecture graph and makes no unique parameter count,
module ownership, execution-order or semantic-coverage claim. Malformed metadata
still fails validation. The CLI remains strict unless artifact inspection is
explicitly requested internally by the web worker.

## Product sequence

```text
reconstruct → visualize → inspect → analyze later
```

Model Autopsy is a future analysis layer built on a trustworthy representation, not a first-release feature.

The public benchmark catalog lists measured static compatibility outcomes for
pinned repositories. It publishes case evidence, not prepared architecture
reports or visitor analyses. Complete static reports, partial graphs, artifact
inspection and failed attempts stay distinct. Its inspection action opens
the normal submission flow with the exact pinned reference. Submitting creates a
visitor-owned analysis. Stored
inspections remain a separate view backed by the report registry. These
compatibility measurements do not measure inference speed or task accuracy.
