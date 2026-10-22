# Value-specific dtype evidence

## Reproduced failure

The unstaged dtype draft returned the first model parameter's dtype for every
`.dtype` expression. A float32 model therefore selected the wrong source branch
after `x.half()` and after `torch.ones(..., dtype=torch.bool)`, with no warning.
The native meta-tensor oracles were float16 and bool respectively. The new reader
selects the correct branch in both reproductions without executing model forward
methods or source helpers.

## Rules and boundaries

`StaticValue` now carries a dtype fact. Reading `.dtype` uses that value's fact,
not an unconditional model dtype. The evidence can come from:

- the canonical input contract: token/position IDs are int64; named floating
  activation/image/audio roles use the declared native meta-model profile;
- the referenced native parameter's own dtype, including mixed parameter dtypes;
- a recognized native tensor factory's explicit dtype, prototype, scalar fill
  inference or current PyTorch default-dtype profile;
- supported tensor casts: `.float()`, `.half()`, `.double()`, `.bfloat16()`,
  `.to(...)` and `.type_as(...)` with established arguments;
- agreeing dtype facts on both sides of a conditional join.

Arbitrary `x`, ambiguous input names and unspecified parameterless inputs do not
inherit float32. Dependency collections, opaque calls, mixed joins, unknown cast
targets and dynamic cast argument expansion cannot establish dtype. A subsequent
undecidable condition retains its branches and warning. A default input profile
is an assumption about the inspected path, not a runtime observation.

Factory rules distinguish float defaults from bool/int `full` fill inference,
and distinguish an omitted/`None` dtype from an unknown dtype expression. Cast
rules distinguish a dtype enum, a device move and another tensor as a target.
An unknown second positional dtype in `.to(device, dtype)` stays unknown.

The bounded rules do not establish dtype after arbitrary normalization, module
calls, functional operators or runtime autocast. General IR floating-edge dtype
annotations still describe the native meta-model profile; this increment improves
source-condition decisions and does not certify every runtime edge or cast.

Production rules only inspect source and metadata. Native meta-tensor operations
are used in tests as independent oracles. No model forward/helper call or weight
payload download is introduced.

## Cache and review

Engine cache identity is `2026.10.23.1`; stored reports remain immutable. A new
inspection is needed to exercise this engine. The branch is
`codex/value-dtype-evidence`, stacked on the helper/rank increments in PRs
[#45](https://github.com/pranavsinghpatil/model-anatomy/pull/45) and
[#46](https://github.com/pranavsinghpatil/model-anatomy/pull/46).

Earlier routing, cache-type, token-membership and frontend Cursor drafts remain
outside this dtype increment. The combined working tree and isolated review
snapshot are verified separately. Existing draft tests that relied on the global
dtype shortcut now require an unresolved result when no value evidence exists.

## Verification

- Focused verification: **107 passed**, including 39 dtype cases and native
  meta-tensor oracles. Both wrong-branch reproductions now select the correct
  branch with the expected float16/bool dtype.
- Combined working tree: **408 passed, 17 skipped**. Exact isolated PR snapshot:
  **404 passed, 17 skipped**. The skips are opt-in network/platform checks; one
  existing Starlette/httpx deprecation warning remains.
- Both trees separately pass all **10 pinned engine/category acceptance cases**.
  GPT-2, SmolLM2, Qwen3 and ViT retain their strict completed-profile checks.
  Category checks allow documented partial/native or artifact-only results.
- Pinned T5 retains 41 graphs, 305 nodes, 132 checkpoint tensors, 60,506,624
  unique parameters, full tensor ownership, 15,536 header bytes and zero weight
  payload reads. It is still partial: **25 warnings** in the combined tree and
  **26** in the isolated PR, including unproven normalization/operator output
  dtype, DenseReluDense cast guards and symbolic sequence length; the isolated
  tree also retains the earlier cache-type gap.
- These warnings replace the draft's false certainty; the earlier combined
  single-warning result relied on the global dtype shortcut. Source paths with
  no dtype proof remain visible, rather than forcing them to float32.
- Ruff lint, scoped formatting, compilation and scoped diff checks pass.

Local diagnostics are `.local/verification/t5-dtype-after.json` and
`t5-dtype-isolated.json`. They are not application fallback data. A subsequent
[fresh validation](VALIDATION_2026-10-23.md) repeats both full suites and pinned
model gates, and adds real HTTP API/worker acceptance on both trees. SmolLM2-360M
and ViT complete with zero warnings, real cold/cache results and zero payload
reads. Browser and hosted deployment acceptance remain separate gates.

## Next gate

Establish dtype transfer for one concrete native module/operator path where T5
loses the fact after normalization. Preserve unsupported outputs as unknown.
Measure before/after on the pinned checkpoint, use native meta oracles where
possible, and retain the symbolic sequence-length condition until an explicit
input profile establishes the one-token or multi-token path. Broader category
and custom-architecture support remains a separate measured sequence.
