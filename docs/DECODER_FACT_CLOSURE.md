# Decoder tensor fact closure

This increment repairs missing output facts for reviewed native Qwen2, Qwen3,
GPT-2 and Phi modules. It extends the dynamic reconstruction approach without
changing the canonical eval/no-cache/no-training-loss scope or loading checkpoint
weight payloads.

## Changes

- Validate attention and feed-forward projection dimensions, parameter dtypes,
  normalization and activation children before proving a native output shape.
- Distinguish tensor outputs from tuple member zero using the reviewed native
  return implementation. A known tuple member does not prove tuple arity.
- Validate the native GPT-2 ModelOutput wrapper identity, ordered fields and
  indexing helpers before proving its primary output port.
- Fingerprint the native modeling source and imported GPT-2 Conv1D source. Source
  drift, replaced forwards, hooks and unsupported child implementations disable
  the associated contract.
- Carry native loop-member port facts only when every member of an exact,
  nonempty ModuleList proves the same map for its individually bound input.
  Direct container calls receive no such contract; no tuple arity or order is
  inferred from registration.
- Expand splatted view/reshape targets only from bounded known metadata items.
- Read the reviewed native attention-implementation property from its backing
  field without invoking the descriptor. Other properties remain unresolved.
- Inline a decoder backbone only through explicit child output-field mappings.
  Update endpoint ports and input facts together; preserve the nested boundary
  for unmapped positional outputs and retain unresolved downstream outputs.
- Advance engine/cache identity to `2026.10.21.2`; retain IR schema `1.1` and
  historical report identities.

## Measured results

The final frozen-source run rechecked **27 pinned repositories** in two isolated
capture processes: **16 complete static reports and 11 partial graphs**, with no
failed, invalid-report, timeout or memory-limit outcome. Every report passed IR
round-trip, graph validation and service/storage size and identity checks. All
27 observed zero checkpoint weight-payload bytes and no forward execution.
Maximum observed process-tree RSS was 331.3 MiB.

Nine previously partial observations now pass all four static coverage gates
with zero warnings:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `openai-community/gpt2`
- `microsoft/phi-2`
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-14B`
- `Qwen/Qwen2.5-7B-Instruct`

All seven previously complete regression controls remain complete. T5-small
warnings fall from 13 to 5 and shape coverage increases from 0.8863 to 0.9035;
its missing dtype and mask-transform contracts remain visible. Whisper-tiny
still has two unresolved transforms and shape coverage 0.8390. Qwen3-0.6B now
has full recorded coverage but remains partial because headers cannot prove
that separately stored tensors declared tied have equal values.

The 250-repository catalog retains every earlier observation and now records
**16 complete, 87 partial, 142 artifacts inspected and 5 failed attempts**.
The four gated access failures and invalid XTTS artifact remain historical
outcomes; they were not rerun or relabeled in this increment. All 27 new reports
were stored in Supabase and their bytes read back successfully. Publication
`947cbb5fb2274c8d2fd9b928c7bd1972c48f67d80220198199be8c8c32dbcf64`
contains all five attributed runs. Catalog import tests verify idempotency and
preservation of all 250 case definitions.

Evidence:

- [Pinned subset](benchmarks/model-matrix-250-decoder-manifest.json)
- [Portable observations](benchmarks/model-matrix-250-decoder-rechecks.json)
- [Latest 250-report diagnosis](benchmarks/decoder-fact-diagnosis.json)

Engine source SHA256:
`17c74c869a041a5ac8e42267074c4f4ff26109baa02bb618ee0874e894daa063`.
Static compatibility checks do not measure inference quality, accuracy or speed.

## Verification

The offline backend/infra run passed **808 tests**, with one Linux-only memory
gate skipped and 18 integration tests deselected. All **11** catalog publication
service tests passed separately after generation: **819 passed** in total.
Ruff, compilation and diff checks pass. Verification exposed a stale input/port
bug in backbone inlining; the final implementation repairs it with regression
coverage rather than weakening graph validation.


The API on port **8000**, Supabase worker and preview on **5176** run the frozen
source above. Database, storage and worker health checks pass. Live acceptance
verified all **250** unique catalog rows, all **245** stored-report links, all
**56** catalog categories, every one of the **27** current-engine stored reports,
historical engine reads and cross-visitor ownership isolation. A fresh pinned
SmolLM2 reconstruction completed, its next request reused the current-engine
cache, and all six exports passed. Optional AI remains explicitly unavailable.

Desktop/mobile browser validation passed **24** mocked catalog checks and **8**
checks against the live measured publication. The eight evidence checks require
`REQUIRE_BENCHMARK_EVIDENCE=1` and were explicitly enabled for their run. The
frontend type check and production build pass, retaining the existing large
bundle advisory.

A read-back-verified source/evidence archive remains local at
`D:/neb-hf/.local/decoder-fact-runtime/local-evidence.zip`. Automatic approval
review declined the optional source/log upload to Supabase because that payload
was not explicitly authorized. All 27 model reports were already stored and
read-back verified; the extra source/log archive was not uploaded.
