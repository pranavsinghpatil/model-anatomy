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

## Follow-up: verified checkpoint headers

The separate `codex/deepseek-shard-header-evidence` increment adds an explicit
header-only audit. Run it with:

```text
python scripts/verify_checkpoint_headers.py deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e1088993b9b82de08f59de5be5a34e42170 --output .local/verification/deepseek-v41-headers.json
```

The live run verified all **48** indexed shard headers and exactly **96,085**
tensor names against the pinned index. Its index digest is
`abc09b0491e66b9f1c3d9dd635896523f2eaef3ca3904a91cfaa216390de7ba4`.
It read **10,685,312 header bytes** and **zero weight payload bytes**. The
validated descriptors sum to **508,182,659,298 tensor elements**, including
scales. This is not a parameter count or module ownership claim. Individual
records were discarded after each shard; the committed
[`evidence/deepseek-v41-headers.json`](evidence/deepseek-v41-headers.json)
contains only per-shard counts, sizes and header hashes.

The first attempt stopped honestly at `F8_E8M0`, which the existing dtype table
did not recognize. [Upstream Safetensors](https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs)
defines it as 8 bits per element; the
validator now supports it. The failed attempt is preserved locally at
`.local/verification/deepseek-v41-headers.failed.json`. The verifier saves a
failure log, with completed request audit, on any unsuccessful run. Its result
is a separate audit and was **not** attached to the immutable visitor
report or used to upgrade graph completeness. The following increment binds
this evidence to newly generated reports. Publisher checkpoint-name
conversion still needs independent review before assigning tensor ownership.

## Follow-up: binding the prior audit to a new report

Engine `2026.10.24.3` carries a digest-pinned copy of that completed audit in
the worker package. A newly generated report binds it only when the exact
model and revision, canonical index SHA-256, declared tensor/shard counts,
all 48 shard filenames, header hashes, file sizes and aggregate totals agree.
The bound evidence explicitly says **previously verified pinned header audit**:
the worker does not fetch those headers again within its 120-second budget.
The report exposes verified-header counts, an audit-document digest and the
per-shard hashes as evidence, while retaining zero individual tensor records,
unknown parameter count and ownership, and `reference_order` links. The saved
`fd318b65ce144aa9a08e5c2a3b5834cd` report remains immutable and unbound.
Only a fresh inspection at the exact pin can show the bound audit. Changed
index identity or missing shards leave it unbound; the source graph remains
partial. The next task is to test the reviewed `inference/convert.py` name
rules against checkpoint names and publisher module state keys before
attempting an ownership map.

A live Supabase-backed browser submission after restarting the API and worker
created visitor report `a1ceb64fb91445d3af65b5d5e222b78f`. The API health
checks showed database, storage and worker healthy. The Architecture view
rendered 49 root components and 45 reference-order relationships. The Report
view showed engine `2026.10.24.3`, the separate 48-header/96,085-name audit,
unknown coverage and no parameter total. The Artifacts view showed zero tensor
records in this job alongside the prior audit and the distinct index
declaration. The saved `fd318b65ce144aa9a08e5c2a3b5834cd` report remained
unmodified. The preview is on `http://127.0.0.1:5176/`; local API/worker and
browser logs are retained under `.local/app-v1/supabase/` and the pre-restart
copy under `.local/verification/preview-before-audit-binding/`.

## Publisher conversion review for the ownership gate

The pinned [`inference/convert.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/convert.py)
has SHA-256 `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76`.
It was read as source, not executed. It conditionally strips `model.`, renames
`self_attn` to `attn`, non-vision `mlp` to `ffn`, and scale/bias suffixes, then
partitions experts and selected tensor dimensions across model-parallel files.
It can skip tied MTP embeddings/heads, pad Engram embeddings, dequantize
`wo_a` weights using scale tensors, and recast expert data. Therefore its
output names and element totals cannot be assumed to equal raw HF checkpoint
parameters or ownership.

The pinned index itself contains 96,085 names: 93,415 under `layers`, 2,401
under `mtp`, 259 under `vision`, and four under `aligner`, plus root tensors.
47,675 names contain `scale`; 94,464 contain `.experts.`. It has 43
`wo_a.weight` names. The present index has no `model.`, `self_attn`,
`weight_scale_inv` or `e_score_correction_bias` spelling, and its `mlp`
spellings are confined to vision. That means conversion rules must be checked
against this checkpoint's actual naming and native module state keys before
building an ownership map. The conversion script reads tensor payloads; it is
not part of the app's metadata-only inspection route.

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
The PR's final quality, browser and container checks were failed when merged.
The quality log includes missing deleted `output/gpt2/model.json` fixtures,
quota behavior failures and a worker status mismatch. Its 48-model results do
not establish successful CI or compatibility with the current catalog branch.

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
