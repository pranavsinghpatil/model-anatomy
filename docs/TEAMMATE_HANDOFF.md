# Backend teammate handoff — October 24, 2026

## Current handoff — October 24, 2026

The current stacked header-verification work is on
`codex/deepseek-shard-header-evidence`, based on `codex/top-all-preserved` at
`e17f3b7` in draft [PR #51](https://github.com/pranavsinghpatil/model-anatomy/pull/51).
Engine `2026.10.24.1` gave the exact pinned DeepSeek V4.1 revision a
**partial publisher-reference topology**. The live visitor report is
`fd318b65ce144aa9a08e5c2a3b5834cd`: it displays 49 root components and
45 reference-order links, with nested blocks. It does not verify Safetensors
shard headers, checkpoint ownership, parameter totals or runtime dataflow.
Read [the evidence and vv audit](PUBLISHER_SOURCE_TOPOLOGY.md) before changing
that boundary. The sections below are historical handoffs, not the current
branch or task instructions.

The separate header audit now verifies all 48 shard headers and 96,085 names
against the pinned index, reading 10,685,312 header bytes and zero weight
payload bytes. The 508,182,659,298 tensor elements include scales and are not
model parameter count. It is stored at
[`evidence/deepseek-v41-headers.json`](evidence/deepseek-v41-headers.json);
the first failed dtype attempt remains in the local failure log. The saved
visitor report still has no attached verified-header evidence. The next branch
binds it to **new** reports with source/index/hash identity checks. Review
`inference/convert.py` before mapping names to reference modules. Do not
upgrade semantic completeness from header counts alone.
The independent audit and `F8_E8M0` metadata recognition advance engine
identity to `2026.10.24.2`; old visitor reports remain immutable.

The current branch `codex/deepseek-report-audit-binding` packages and binds that
audit for **new** exact-pin reports at engine `2026.10.24.3`. The old saved
report remains unchanged. The UI distinguishes the prior audit's verified
aggregate from tensor records inspected in this job. The next teammate gate is
a deterministic, metadata-only name audit over the pinned index: compare the
reviewed converter rules with publisher module state keys; count unambiguous,
aliased, scale and unmapped names; reject collisions and changed revision,
index digest or converter hash. Do not claim ownership until the name mapping
and source-defined module state keys are independently checked.
The pinned converter source hash and current index name cohorts are recorded in
[publisher topology evidence](PUBLISHER_SOURCE_TOPOLOGY.md). It can split,
pad, skip or recast tensors, so do not run it in the metadata-only worker or
equate its output with the raw indexed checkpoint.
The refreshed local app is open at the new browser report
`a1ceb64fb91445d3af65b5d5e222b78f`. Its graph remains partial, while
Report and Artifacts show the separately bound header audit. The old visitor
report remains unchanged.

Also reconcile vv's merged [PR #48](https://github.com/pranavsinghpatil/model-anatomy/pull/48)
on `b-mvp` in a separate tested slice. Its parser and Transformers 5.20.2 pin
are not in PR #51; a merge simulation found 15 conflicts. Preserve the catalog,
the current static-flow tests and the source/topology evidence rules while
measuring the new parser on pinned controls. Do not treat the 48-model claim or
the version pin as native DeepSeek support without a verified replay. Its
quality, browser and container checks failed; the quality log includes missing
deleted GPT-2 fixtures and service behavior failures. Repair and rerun those
checks before considering integration.

## Review and integration

The local checkout is now `codex/value-dtype-evidence`, created at the verified
rank commit `f6a1ffc`. PR
[#45](https://github.com/pranavsinghpatil/model-anatomy/pull/45) contains the
helper-binding fix (`8283627`). The tensor-rank increment is a separate review
on top of that commit, published as `codex/trusted-tensor-ranks` and targeting
`codex/static-helper-binding` so the review contains only this increment.

Earlier Cursor changes in the backend, frontend, deployment docs and local start
script remain unstaged. They are not all included in either backend increment.
Do not stage the whole working tree or overwrite those drafts. Review the small
PR diffs in order: helper binding, trusted ranks, then value-specific dtype.

Read `00-GROUND.md`, `DECISION_LOG.md` and `BACKEND_RELIABILITY.md` before the next
change. Artifact inventory is not an architecture graph, and mapped tensors alone
do not prove complete source semantics.

## Previous increment: trusted ranks

- `backend/model_anatomy/src/static_tensors.py` recognizes native PyTorch
  factories by callable identity, including imported aliases. It derives rank
  from supported explicit size forms and keeps unknown sizes explicit.
- `static_flow.py` carries those facts across justified paths, invalidates stale
  rank after unsupported shape changes, and separates opaque dependency records
  from evidence that a result cannot be `None`.
- Factory rules do not execute factories, model helpers or forward methods.
  Native meta-tensor calls exist only as deterministic test oracles.
- Engine version `2026.10.22.2` prevents reuse of earlier cache results. Existing
  stored reports are immutable; submit a new inspection to exercise this engine.

## Rank-increment T5 baseline (October 22)

Target: `google-t5/t5-small` at immutable revision
`df1b051c49625cf57a3d0d8d3863ed4d13564fe4`.

In the combined working tree, the mask-rank warning disappears: unresolved
conditions fall from two to one, retaining 41 graphs, 305 nodes, all 132 tensors
owned, 60,506,624 unique parameters and zero weight payload reads. The remaining
condition is `sequence_length != 1`. T5 remains partial. These figures include
the pre-existing Cursor changes.

The isolated rank PR independently produces the same graph/tensor counts and
has no mask-rank warning, but retains **22** other warnings: repeated dtype-clamp
and opaque-operation findings plus cache and symbolic-length conditions. The
single-warning combined result must not be claimed for a clean checkout of the
rank PR alone. Do not import the draft's global dtype shortcut just to remove
those warnings.

These are historical rank-stage counts, superseded by the value-specific dtype
findings below. The old single-warning combined result was not proof that every
dtype decision was correct.

Local diagnostic snapshots are `.local/verification/t5-ranks-before.json` and
`t5-ranks-after.json`; `t5-ranks-isolated.json` records the isolated review. They
are evidence, never application fallback data.

## Current increment: value-specific dtype

The unsafe `.dtype` fallback is replaced: casts and explicit factory dtypes now
control their own source conditions, parameters use their own metadata, and only
agreeing joins retain dtype facts. Arbitrary `x`, unknown targets and opaque
module/operator outputs remain unknown. The float16-cast and boolean-mask
reproductions now choose the correct branch instead of the wrong branch with
zero warnings. No model forward/helper is executed.

Engine version is `2026.10.23.1`. See [DTYPE_EVIDENCE.md](DTYPE_EVIDENCE.md) for
the supported transfer rules, meta-tensor oracles and static-profile limits.
Earlier broad dtype assertions in draft tests were corrected to require an
unresolved result where evidence is absent.

Current verification: **408 tests passed** in the combined tree and **404** in
the isolated PR, each with 17 opt-in/platform skips and the existing Starlette
deprecation warning. Both trees pass all **10 pinned engine/category cases**.
The focused source-rule suite has 107 passing cases, including 39 dtype checks.
Lint, scoped formatting, compilation and diff checks pass.

Pinned T5 still has 41 graphs, 305 nodes, all 132 checkpoint tensors owned and
zero payload reads. It now retains **25** actual warnings in the combined tree
and **26** in the isolated PR, exposing unproven normalization/operator dtype
and DenseReluDense cast conditions instead of assuming float32. The extra
isolated warning is the earlier cache-type gap. The graph is still partial;
its rank warnings remain closed. Current local snapshots are
`.local/verification/t5-dtype-after.json` and `t5-dtype-isolated.json`.

## Next teammate task: one proven module-output dtype path

1. Reproduce where pinned T5 loses dtype evidence after normalization/module
   calls. Select one concrete native path; do not treat every module output as
   the first parameter's dtype.
2. Establish its output dtype from a supported source rule or exact native
   operation contract. Reject opaque/custom overrides and unsupported mixed
   dtypes. Preserve unknown outputs and dependencies.
3. Verify against native meta-tensor oracles, then repeat pinned T5 plus the
   four canonical engine cases. Report the exact warning delta and zero payload
   reads. A reduced warning count alone is not sufficient acceptance.
4. Update cache identity, docs and the PR. Keep broad dtype propagation, runtime
   autocast and general operator inference outside the increment unless required
   by that measured path.

Do not force the remaining `sequence_length != 1` condition from config context
length: maximum capacity does not establish the actual input length. A later
input-profile increment must distinguish one-token and multi-token paths, retain
both when symbolic, and document how the graph represents that distinction.

## Reproduce

From the repository root with engine/API/dev dependencies installed:

```powershell
.venv/Scripts/python.exe -m pytest -q backend/tests/test_static_dtypes.py backend/tests/test_static_tensors.py backend/tests/test_static_helpers.py backend/tests/test_static_predicates.py
$env:RUN_HF_INTEGRATION = '1'
.venv/Scripts/python.exe -m pytest -q backend/tests/test_engine_integration.py backend/tests/test_category_integration.py
uv run --no-sync model-anatomy google-t5/t5-small --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --output .local/teammate/t5 --config-dir .local/teammate/t5-config --preview-dir .local/teammate/preview --inspect
```

The Windows API tests need local socket access. Restricted runs previously hung
inside Windows `socketpair`; this is an environment constraint, not a reason to
change application event-loop behavior.

## Historical rank verification and outstanding acceptance

- Isolated snapshot: **365 passed, 17 skipped**, and all **10** pinned native/
  category acceptance cases passed separately. One existing Starlette/httpx
  deprecation warning remains. Category acceptance permits documented partial
  graphs and artifact-only diffusion results; it is not complete semantics.
- Combined working tree: **369 passed, 17 skipped**; all **10** pinned cases
  passed. The final ViT/factory checks also passed in both trees.
- Forty factory/rank regressions include comparisons with real PyTorch meta
  tensor ranks, aliases, shadows, invalid bindings, dependency preservation,
  nullable merges and the installed T5 decoder source.
- Ruff lint, scoped formatting, compilation and scoped diff checks passed.

An earlier isolated run failed the ViT completion assertion because a
rank-preserving cast unnecessarily changed its static value identity. Preserving
the existing value for casts/contiguous fixed that regression. The corrected
isolated full run and real-model acceptance above passed; the failure was not
removed from the test expectations.

One combined-tree run also failed the existing HTTP-load benchmark's 0.3-second
wall-clock window while other verification jobs were running. With those jobs
finished, the full unmodified suite passed 369 tests in 45.37 seconds. This does
not establish benchmark reliability under contention or hosted load capacity.

The [October 23 validation](VALIDATION_2026-10-23.md) repeats both full suites and
pinned-model gates, and verifies real HTTP API/worker cold/cache flows on both
trees for SmolLM2-360M and ViT, each completed with zero warnings and zero payload
reads. Live browser handoff and hosted deployment remain separate gates. Swin,
Whisper, VideoMAE, SigLIP and the DeepSeek custom-architecture case still need
their own measured semantic increments.

## DeepSeek exact-name gate — October 24

The `codex/deepseek-name-coverage` branch is stacked on the exact-pin header
binding branch. Its offline audit checks the 96,085 checkpoint index names
against a finite publisher-constructor state-key inventory and the pinned
config, four source hashes and converter hash. All names match; the compact
[name evidence](evidence/deepseek-v41-name-coverage.json) lists 114 normalized
patterns and counts for 632 closest reference components. The source graph
gains the three previously missing root image-marker parameters. The
converter's name rewrites leave these pinned names unchanged, but its tensor
value transformations are not verified or executed.

The next gate is an exact-index and audit-digest-bound report field that exposes
these name counts per graph component. Keep that field distinct from verified
tensor records, element totals and parameter count. Historical visitor reports
stay immutable. Follow with per-name header shape/type association only after
the name field is accepted; do not infer runtime dataflow from source order.
