# Inspection failure diagnosis — 2026-10-23

## Observed DeepSeek report

The visitor-owned report `a63acf9030994eda97dd90a6fd21088b` for
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` showed 83 repository files,
zero inspected tensor headers and no graph. Its saved reason is
`metadata_budget_exceeded`. The visible profile separately identifies
`model_type=deepseek_v41` and “No native Transformers config for
model_type=deepseek_v41”. The user-facing report incorrectly used generic
reconstruction wording and a “Checkpoint selected: Yes” label that meant only
checkpoint artifacts existed. API and worker stderr at the time held startup
messages; model-stage outcomes live in saved reports, not necessarily stderr.

The pinned Safetensors index declares 96,085 tensor names across 48 shard files.
`MAX_CHECKPOINT_TENSORS=20_000` stops full inspection before shard headers are
fetched. A separately selected shard yielded 263 verified headers in an earlier
inspection, but that is a single-file result. Installed Transformers 4.57.0 and
4.57.6 lack the `deepseek_v41` native config mapping. Raising the tensor budget
alone cannot create a trustworthy architecture graph; header-byte, native-record
and graph budgets would still need review.

## Wider evidence and limits

The live catalog returned 412 records at the time of this audit. Only two saved
records were from engine `2026.10.23.3`: this DeepSeek artifact inspection and
a partial Pythia-70m report. A local pinned Pythia-160m replay is also partial.
Both Pythia cases have semantic, module and checkpoint coverage 1.0 but
tensor-shape coverage 0.5641 and seven unresolved warnings concerning tensor
transforms, QKV splitting, rotary/attention calls, and legacy `inv_freq`
ownership. The historical 250-model audit is mixed-engine evidence (142
artifact-only and 87 partial), not a current-engine success rate. Its largest
artifact-only cohorts are 49 missing native routes and 49 unavailable
Safetensors. See `docs/benchmarks/model-coverage-audit-250.json` and
`docs/DYNAMIC_GPT_NEOX_ROLES.md` for pinned cases and warning details.

An exact pinned metadata-only replay on engine `2026.10.23.4` confirmed the
same reason and native blocker, with zero verified tensors, graph nodes, and
weight-payload bytes. The index declaration had canonical JSON SHA256
`abc09b0491e66b9f1c3d9dd635896523f2eaef3ca3904a91cfaa216390de7ba4`.
The bounded machine-readable observation is
`docs/benchmarks/inspection-diagnosis-2026-10-23.json`.

After restarting the local Supabase-backed API and worker on engine
`2026.10.23.4`, a visitor-owned submission through the app completed as
`2643aca98c5b44219e7798710a50f669`. The live Artifacts view displayed
96,085 index-declared names, 48 shard files and zero inspected tensor headers.
The Report view displayed the 20,000-record limit, “Not measured” coverage,
and the independent missing-native-config blocker. The graph remained absent.

## This increment and acceptance boundary

The resolver now retains an index-only declaration summary after validating
the pinned index and shard paths but before rejecting it for the tensor budget.
The summary contains the index URL, canonical JSON digest, declared name/shard
counts and configured limit. It does not count as a verified tensor header.
The saved report, API and UI expose the summary and native blocker distinctly;
artifact-only UI no longer implies native reconstruction or a selected
checkpoint. Engine identity is `2026.10.23.4`, so existing reports remain
immutable and a new submission is needed to see the new summary.

Positive and negative tests cover declaration retention, zero verified tensors,
no shard fetch, invalid shard rejection, API projection and UI wording. The next
separate gates are a bounded source-reviewed DeepSeek native route and a current
engine pinned cohort large enough to measure completion by blocker stage. For
partial GPT-NeoX, the next shared increment is a source-backed tensor-operation
contract, followed by checkpoint ownership reconciliation. No universal
completion claim follows from this change.
