# Publisher reference topology — 2026-10-24

## Scope and evidence

The pinned `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint has a validated
index declaring 96,085 tensor names in 48 shards. The 20,000-record full
checkpoint limit stops shard-header extraction before native construction. Its
`deepseek_v41` type is absent from the installed Transformers 4.57 mapping.
Therefore the earlier visitor-owned reports, including
`c0a52a95f62c4e50ab4eedbfbb0c3f2e`, have no graph.

The publisher provides a reference `Transformer` in
[`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py),
with [`vision.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/vision.py),
[`engram.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py)
and [`inference/config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/config.json).
Their reviewed SHA-256 digests are stored in `publisher_topology.py`. The
published root config's canonical JSON SHA-256 is
`0110b6f9349cf6404bef91bfe8a491be248d3d5a044592d3c443dc9073fa0096`.
The route requires that exact model ID, immutable revision, digest and source
file inventory. It does not download or execute source during an inspection.

The graph shows 40 distinct decoder blocks, each with attention, normalization
and MoE; configured Engram attachments, compressed-KV sources and sparse
indexers; three separate speculative blocks; and a 32-block vision encoder
with aligner. The 384 routed experts per text block and 128 per speculative
block are compressed as repetitions. All links are explicitly labeled
`reference_order`. This reflects the reviewed reference's module structure,
not a native instantiated module tree or runtime dataflow.

## Trust boundary

The result is `partial` with mode `publisher_reference_topology`. Its detailed
warnings, node metadata and visible graph banner say that parameter counts,
tensor shapes, checkpoint ownership and dataflow are unverified. The checkpoint
index counts remain **declared**, not verified shard-header counts. The route
does not change strict CLI behavior or grant a graph to another revision,
changed config or missing source files. There is no model-weight download.

The first follow-up is bounded aggregate verification of all 48 Safetensors
headers against the index. That can establish checkpoint-wide counts without
retaining 96,085 tensor records. Mapping tensor names to reference modules
then needs separate source-backed conversion and ownership checks. A generic
custom-model route would need a reviewed, deterministic source extractor rather
than copying this model's module names.

## vv branch audit

PR [#48](https://github.com/pranavsinghpatil/model-anatomy/pull/48) merged
`feat/vv` into `b-mvp` at `d3cff4c` on 2026-10-23. It includes a new
`module_topology.py` parser, source-completeness logic and a Transformers
`5.20.2` pin. It also removes the existing `static_flow.py` path and several
tests, and changes the service/job flow. A read-only merge simulation against
this branch reports 15 conflicts, including the pipeline, hierarchy, registry,
jobs, worker and frontend proxy. The 5.20.2 mapping and the reported 48-model
outcomes have not yet passed this branch's pinned acceptance suite. The merged
work is retained on `b-mvp`; it should be reconciled by behavior and tests in
small increments so the current catalog and evidence rules are preserved.

## Validation

The focused backend suite passed 54 tests, including exact-pin topology,
altered-config/revision/file rejection, artifact fallback and service jobs.
Windows storage tests require unsandboxed hard-link creation; two sandboxed
attempts failed at `os.link` permissions, then the same suite passed outside
the sandbox. An unsandboxed 30-test native-routing suite and 11 focused API/
fallback controls passed. Ruff, frontend typecheck and production build passed.

The updated local Supabase-backed app ran on ports 8000/5176. A fresh
visitor-owned submission through the browser completed as `partial` at
`fd318b65ce144aa9a08e5c2a3b5834cd`. Its architecture view displayed the
source-topology notice, 49 root components and 45 `reference_order`
relationships; a decoder block's nested graph loaded. The Report view showed
all coverage values as unknown and stated that shard headers, parameter totals
and tensor ownership were unverified. The older report
`c0a52a95f62c4e50ab4eedbfbb0c3f2e` remains immutable and graph-free.
