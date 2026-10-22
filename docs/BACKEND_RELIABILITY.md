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

The immediate next gate is trustworthy tensor facts for the remaining T5 mask
path. Broader category/custom support follows measured acceptance, one family
and one mechanism at a time.
