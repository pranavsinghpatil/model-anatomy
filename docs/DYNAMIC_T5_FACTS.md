# Dynamic T5 dtype and mask facts

This increment follows the [dynamic reconstruction plan](DYNAMIC_RECONSTRUCTION_PLAN.md):
choose a defect from actual pinned reports, prove bounded facts from native source
and metadata, then compare the same checkpoints. Repository names do not select
the new contracts. Their gates use native class, reviewed source, argument facts
and explicit unsupported cases.

## Observed baseline

Engine `2026.10.21.2` reproduces five common warnings for T5-small, T5-base and
Chronos-T5-small: two casts, mask expansion, mask cloning and masked fill. All
three already have complete checkpoint ownership, node classification and module
classification. Their recorded dataflow shape coverage is 0.903509. Flan-T5-small
has additional gated-feed-forward dtype branches; Whisper has separate convolution
and permutation gaps. These are distinct reconstruction defects.

## Contract boundary

The interpreter gains bounded mask fact algebra and a reviewed native dtype
metadata route. No property descriptor, model forward or tensor computation runs
to reconstruct a model. Dtype facts come from permitted registered meta parameter
metadata, with source/descriptor identity, traversal and environment guards.
Changed implementations, overrides and heterogeneous or real parameter metadata
cannot acquire the new dtype contract.

Mask facts require supported signatures, known rank/dimensions and valid
broadcast/dtype conditions. Unknown slice bounds remain unknown. Unsupported
calls retain their warnings and dependency evidence. New assumptions and native
metadata facts are attributed in the report diagnostics.

The verified chain includes explicit-dtype `full`, triangular masking,
integer comparisons, float/boolean mask multiplication, scalar-left subtraction,
slice/expand/clone, and masked fill. A proven full-slice assignment rebinds the
returned mask to the updated value, preserving dataflow into decoder execution.
Other slice assignments lose their facts and emit an unresolved-update warning.

The engine/cache identity advances to `2026.10.22.1`; IR schema remains `1.1`.
Historical reports and benchmark/external publications retain their identities.
New inspections use the existing durable save and visitor-owned revisit flow.

## Acceptance and remaining gaps

The [seven pinned cases](benchmarks/dynamic-t5-manifest.json) cover T5-small,
T5-base, Flan-T5-small, Chronos-T5-small, Whisper-tiny and the existing complete
GPT-2/SmolLM2 controls. Comparisons check full report digests, identical source
configuration and tensor inventory, service/graph validity, zero payload reads
and no forward execution. The final source capture has seven valid reports:
**two complete controls and five partial reports**, with no failed attempt.

| Pinned model | Warnings before → after | Dataflow shape coverage before → after |
| --- | ---: | ---: |
| T5-small | 5 → 1 | 0.903509 → 0.922865 |
| T5-base | 5 → 1 | 0.903509 → 0.922865 |
| Chronos-T5-small | 5 → 1 | 0.903509 → 0.922865 |
| Flan-T5-small | 13 → 9 | 0.853186 → 0.874346 |
| Whisper-tiny | 2 → 2 | 0.838983 → 0.838983 |
| GPT-2 and SmolLM2-135M controls | 0 → 0 each | 1.0 → 1.0 each |

All seven retain identical source configuration, checkpoint tensor records and
parameter counts. Each report passes graph/public API/storage validation and IR
round trip, with zero weight payload bytes read and no forward execution. Peak
sampled process-tree RSS was **327.8 MiB**; this is a measurement on the recorded
local Windows environment, not a universal capacity guarantee.

The offline backend/infra suite passes **882 tests**, with two platform skips
and 20 network integration cases deselected. Both pinned T5-small/T5-base
dependency tests pass separately. Changed Python passes Ruff, compilation and
diff checks. The first sandboxed full-suite attempt stalled without a result;
the rerun with process permissions and diagnostic output passed in 55.04 s.
Sandboxed Hugging Face probes also failed DNS; the permitted pinned runs passed.

See [before/after evidence](benchmarks/dynamic-t5-comparison.json). Final engine
source SHA256:
`67223d4d28f6dfe72a6cd0db481e28e18926c850171a9a8ebf3000d4f1a44319`.

## Persistence and live application

All seven validated reports were saved to Supabase with private storage read-back
verification. The registry now holds **408 reports**, with the identities and
content/summary digests of all **401 earlier reports** unchanged. Existing
benchmark publications/cases and the external CSV publication are unchanged.
Discovery still has **380 records**, all **250 benchmark models**, and all
**162 external observations**.

The API/worker on port **8000** run the verified engine; the frontend on **5176**
runs the catalog route correction. Database, storage and worker health pass.
A fresh pinned T5-small worker inspection reproduces the measured one-warning
partial result, saves automatically, opens through the catalog, renders the
interactive graph, and reopens from cache for independent visitors with isolated
analysis histories. The mobile catalog fits its viewport.

Visual acceptance caught a catalog link still using the obsolete
`/app/analyses/:id/architecture` path. Reopening now uses the actual workspace
route with `?view=architecture` or, for artifact-only reports, `?view=artifacts`.
The browser regressions require rendered workspaces rather than only a URL match.
All **48 desktop/mobile catalog cases** pass across the combined run and the
two-case fixture-ID correction rerun. Frontend formatting, TypeScript and the
production build pass; the existing Vite bundle-size advisory remains.

See [live acceptance and preservation](benchmarks/dynamic-t5-live-acceptance.json).

The remaining T5-small/base/Chronos warning is the symbolic
`sequence_length != 1` branch: both triangular-mask paths retain source
conditions, and the engine does not claim the branch has a single known outcome.
Flan-T5 additionally retains gated-feed-forward dtype branches. T5's auxiliary
position-bias outputs also need call-argument-aware contracts:
their dimensions can depend on the mask and cross-attention inputs. A successful
mask operator does not establish those outputs. Partial reports remain partial
while any recorded gate is unresolved. The Chronos case measures the selected
native T5 representation; it does not establish its forecasting preprocessing,
sampling or complete application pipeline. Static anatomy does not measure task
quality or inference performance.
