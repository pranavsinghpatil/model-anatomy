# Backend teammate handoff — October 22, 2026

## Review and integration

The local checkout remains on `pranav/frontend-revamp`. PR
[#45](https://github.com/pranavsinghpatil/model-anatomy/pull/45) contains the
helper-binding fix (`8283627`). The tensor-rank increment is a separate review
on top of that commit, published as `codex/trusted-tensor-ranks` and targeting
`codex/static-helper-binding` so the review contains only this increment.

Earlier Cursor changes in the backend, frontend, deployment docs and local start
script remain unstaged. They are not all included in either backend increment.
Do not stage the whole working tree or overwrite those drafts. Review the small
PR diffs and integrate the helper fix before the rank increment.

Read `00-GROUND.md`, `DECISION_LOG.md` and `BACKEND_RELIABILITY.md` before the next
change. Artifact inventory is not an architecture graph, and mapped tensors alone
do not prove complete source semantics.

## What this increment changes

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

## Measured T5 boundary

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

Local diagnostic snapshots are `.local/verification/t5-ranks-before.json` and
`t5-ranks-after.json`; `t5-ranks-isolated.json` records the isolated review. They
are evidence, never application fallback data.

## Next teammate task: establish value-specific dtype facts

The existing unstaged `profile_dtype()` draft gives every `.dtype` access the
first model parameter's dtype, including values after a cast. Do not use that as
proof of a branch decision. It is deliberately outside the rank PR.

1. Reproduce a wrong branch with an explicit native `.half()`, `.float()` or
   `.to(dtype=...)` cast and with an integer/bool input or constructed mask.
2. Track dtype per static value only when the native operation establishes it.
   Keep an unknown dtype unknown; do not substitute the model's parameter dtype.
3. Preserve both dependencies and a warning for an undecidable dtype condition.
   Add deterministic tests against native meta-tensor dtype behavior, plus opaque
   and shadowed-call regressions that prove no helper/forward execution.
4. Repeat pinned T5 and the established canonical engine cases. Report any newly
   exposed warnings honestly; do not delete warnings to recover a completion badge.
5. Update engine cache identity, documentation and the PR with exact evidence.

Do not force the remaining `sequence_length != 1` condition from config context
length: maximum capacity does not establish the actual input length. A later
input-profile increment must distinguish one-token and multi-token paths, retain
both when symbolic, and document how the graph represents that distinction.

## Reproduce

From the repository root with engine/API/dev dependencies installed:

```powershell
.venv/Scripts/python.exe -m pytest -q backend/tests/test_static_tensors.py backend/tests/test_static_helpers.py backend/tests/test_static_predicates.py
$env:RUN_HF_INTEGRATION = '1'
.venv/Scripts/python.exe -m pytest -q backend/tests/test_engine_integration.py backend/tests/test_category_integration.py
uv run --no-sync model-anatomy google-t5/t5-small --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --output .local/teammate/t5 --config-dir .local/teammate/t5-config --preview-dir .local/teammate/preview --inspect
```

The Windows API tests need local socket access. Restricted runs previously hung
inside Windows `socketpair`; this is an environment constraint, not a reason to
change application event-loop behavior.

## Verification and outstanding acceptance

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

Live web-worker/browser handoff and hosted deployment remain separate gates;
engine tests do not establish them. Swin, Whisper, VideoMAE, SigLIP and the
DeepSeek custom-architecture case still need their own measured semantic
increments.
