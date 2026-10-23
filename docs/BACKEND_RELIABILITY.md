# Backend reliability increments

This log tracks measured reconstruction gaps and the next bounded change. The
Cursor transcript is historical context; it is not fresh acceptance evidence.
The ground and decision log remain the product's source of truth.

## Working approach

1. Reproduce a concrete failing model at an immutable revision. Record the graph,
   tensor ownership, unresolved paths and metadata-read audit before changing code.
2. Fix the shared mechanism responsible for that failure. Never add an assumed
   architecture or suppress a warning to improve a completion badge.
3. Check valid and rejected cases, then repeat the real checkpoint inspection.
   Rerun established native-model acceptance to catch regressions.
4. Inspect the diff, document the remaining boundary, update cache identity when
   extraction changes, and send the increment through a PR.
5. Choose the next gap from the measured result. A family is not complete merely
   because its native model can be constructed or its tensors can be mapped.

## October 22: bind static-source helper calls correctly

Before this increment, a fresh inspection of `google-t5/t5-small` at revision
`df1b051c49625cf57a3d0d8d3863ed4d13564fe4` produced a
`T5ForConditionalGeneration` graph, 132 checkpoint tensors, 60,506,624 unique
parameters, full checkpoint ownership and zero weight payload reads. Three
decoder conditions remained unresolved:

- `sequence_length != 1`
- `attention_mask is not None`
- `attention_mask is not None and attention_mask.dim() == 4`

The static reader removed the first parameter from every helper, including
`@staticmethod` helpers. It also converted keyword arguments into an ordered
list, assigning them to positional slots. T5's native causal-mask helper is a
static method, so the reader lost the actual attention mask before inspecting
the source branch.

The increment preserves the first parameter of static helpers, distinguishes
instance receivers, binds named arguments to their declared parameters and
retains omitted literal defaults, including keyword-only defaults. Invalid or
unsupported binding must retain dependencies and issue a warning. Helpers and
forward methods are never executed. Class methods and dynamic argument
expansions are outside this increment's supported binding subset.

Engine version `2026.10.22.1` separates new inspections from previously cached
reports. Existing saved reports remain immutable.

### Evidence

The existing Cursor backend/frontend changes were present before this increment
and remain unstaged. The following before/after T5 measurements describe that
combined working tree, not a claim that this PR includes the earlier changes.

- T5 retains 41 nested graphs, 305 nodes, all 132 tensors owned, 60,506,624 unique
  parameters and zero weight payload bytes. It reads 15,536 Safetensors header
  bytes. Unresolved decoder conditions fall from three to two: the independent
  `attention_mask is not None` warning disappears; the symbolic sequence-length
  and combined rank-four-mask checks remain. The report is still partial.
- Final working-tree suite: **329 passed, 17 skipped**. The skips are opt-in
  network/platform checks. One existing Starlette/httpx deprecation warning
  remains.
- Isolated staged snapshot, excluding prior Cursor edits: **325 passed,
  17 skipped**, with the same deprecation warning. Eleven helper regressions
  cover binding, rejected cases, dependency preservation and duplicate-event
  prevention.
- Both the combined working tree and isolated snapshot pass all **10 pinned
  engine/category acceptance cases**: GPT-2, SmolLM2, Qwen3, ViT, T5, Swin,
  Whisper, VideoMAE, SigLIP and selected diffusion artifacts. Category acceptance
  permits documented partial results; it does not certify complete semantics.
- Ruff lint, scoped formatting, Python compilation and scoped diff checks pass.

The full report snapshots are local diagnostics at
`.local/verification/t5-step-baseline.json` and `t5-step-after.json`; they are not
seeded application results. These measurements verify engine extraction, not a
fresh running web-worker/browser session.

### Remaining acceptance work

- Establish T5 decoder sequence-length/rank facts without inventing a concrete
  input size. Preserve unresolved numerical/mask paths until that evidence exists.
- Audit existing dtype, tensor-factory and token-membership assumptions before
  treating the corresponding branch decisions as universally valid. This helper
  change does not establish those assumptions.
- Measure Swin, Whisper, VideoMAE and SigLIP separately. Existing category tests
  establish graphs and ownership, not complete operation or modality semantics.
- Reproduce the handoff's DeepSeek custom-architecture failure at a pinned
  revision before designing a separate source-reader increment. Unknown native
  architectures still have no justified module graph; artifact inventory is not
  architecture support.
- Live HTTP/browser handoff and hosted deployment are separate acceptance gates;
  engine checks alone do not establish them.

The following increment closes the rank gate. Broader category/custom support
continues through measured acceptance, one family and one mechanism at a time.

## October 22: trusted tensor-factory rank facts

The reader now recognizes native PyTorch tensor factories by callable identity,
including imported aliases, rather than a final name such as `ones`. Supported
size forms establish rank without inventing symbolic dimension values. Opaque,
shadowed, dynamic and invalid forms retain dependency evidence and uncertainty.
Shape-changing operations invalidate stale rank; nullable branch merges do not
become non-null merely because they have a tensor dependency. Rules record a
`rule_sha256` identity and explain their basis. No production factory/helper or
forward method is called.

Engine version is `2026.10.22.2`. In the combined working tree, the pinned T5
mask-rank warning is gone: warnings fall from two to one, with unchanged 41
graphs, 305 nodes, 132 tensors, 60,506,624 unique parameters, 15,536 header bytes
and zero payload reads. The symbolic sequence-length condition remains.

The isolated rank PR has the same graph/tensor counts and no mask-rank warning,
but retains 22 other dtype/opaque-operation/cache/length warnings because the
earlier Cursor drafts are excluded. It remains partial. This distinction is
part of the acceptance evidence, not a reason to import the draft's global dtype
assumptions.

Verification: 40 new factory/rank checks include real meta-tensor rank oracles;
the combined suite passed 369 tests with 17 skips, while the isolated suite
passed 365 with 17 skips. Both passed all 10 pinned engine/category cases. Ruff,
scoped formatting, compilation and scoped diff checks pass. An initial isolated
ViT completion regression was fixed by preserving value identity through
rank-preserving casts; the test expectation was retained and the full isolated
rerun passed. One existing Starlette/httpx deprecation warning remains.

A combined-tree rerun hit the existing HTTP-load benchmark's 0.3-second timing
window while other verification jobs ran. The final full rerun without those
jobs passed 369 tests; no benchmark expectation or application behavior changed.
This is not evidence of hosted load capacity or timing stability under contention.

The next bounded task is **value-specific dtype evidence**, followed by explicit
one-token/multi-token input-profile handling for the remaining symbolic length
condition. The [teammate handoff](TEAMMATE_HANDOFF.md) gives the review order,
reproduction commands and acceptance criteria. Live web and deployment gates
remain unverified by this engine increment.

## October 23: value-specific dtype evidence

Work continues on the user-requested new branch `codex/value-dtype-evidence`,
based on rank commit `f6a1ffc`. The draft's unconditional model-dtype shortcut
was demonstrably wrong after a half cast and for a boolean factory mask: both
selected the wrong source branch with no warning. Both now select the correct
branch, matching native meta-tensor dtype oracles.

Dtype facts are attached to values from explicit input profiles, actual parameter
metadata, supported factory arguments and cast contracts. Unknown generic inputs,
opaque outputs, mixed joins and unresolved `.to` arguments stay unknown. Engine
cache identity is `2026.10.23.1`. This increment does not establish all module or
operator output dtypes, runtime autocast or every IR edge's runtime cast.

Verification: **107 focused checks passed**, including 39 dtype cases. Full
combined tree: **408 passed, 17 skipped**; isolated PR: **404 passed, 17 skipped**.
Each tree also passed all **10 pinned engine/category cases**. Lint, scoped
formatting, compilation and diff checks pass; the existing Starlette/httpx
deprecation warning remains. Network/platform skips were not silently counted
as passing tests.

T5 preserves its graph and full checkpoint ownership with zero payload reads,
but now exposes 25 combined-tree/26 isolated warnings instead of using the global
dtype shortcut to hide unknown normalization and module outputs. The four
canonical completed-profile models retain their strict checks. T5's remaining
dtype, cache and symbolic-length conditions are explicit acceptance work.

The [dtype evidence](DTYPE_EVIDENCE.md) and updated
[teammate handoff](TEAMMATE_HANDOFF.md) define the next increment: prove one
concrete native normalization/module-output dtype path, without broad assumptions
about arbitrary outputs. Live web/browser and deployment checks remain separate.
