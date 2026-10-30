# Profile and operator evidence closure

This increment implements the first part of [the dynamic reconstruction plan](DYNAMIC_RECONSTRUCTION_PLAN.md): conservative tensor facts, bounded metadata control flow, reviewed native output ports, and graph contracts. It does not implement all proposed family routes.

## Implemented behavior

- Shape, dtype, symbolic dimension and tuple-member facts propagate only through supported contracts. Unknown calls cannot become shape-preserving operations by default.
- Canonical local inputs declare nonempty symbolic-axis and native meta dtype assumptions. Reports identify eval mode, disabled cache, excluded training/loss paths and the absence of runtime evidence.
- Exact Torch identities authorize bounded range/zip metadata, integer position ranges, split/chunk, casts, reductions, reshape/view and selected unary operators. Literal strings are not symbolic dimensions.
- Source-hashed native T5, Whisper, Llama and ViT contracts establish only reviewed output ports. A known port zero does not establish tuple length or other ports.
- An exact native ModuleList can use a sequential child graph when an unconditional source loop proves output carry, including singleton containers; registration order alone is insufficient.
- Branch joins retain facts true on every feasible path. Boolean expressions short-circuit, conditional arms retain opposite conditions, and unverified properties are not executed.
- Graphs retain per-port output specs, scalar shapes and uncertainty. Primitive contracts can be reconciled from one verified incoming graph edge; missing facts cannot be filled by a default hidden-state shape.
- Engine identity advances to `2026.10.21.1`; IR remains `1.1`. Historical reports keep their original extraction identities.

## Validation scope

Full validation results and persisted report counts are recorded below only after capture and read-back verification. Static compatibility checks do not measure model inference quality or speed.

## Measured recheck and persistence

The frozen-source run rechecked **111 pinned repositories across 38 categories**:
all 87 previously partial graphs, all 16 previously complete observations and
eight artifact/access/failure controls. Two isolated capture processes retained
the same per-case 120-second/4-GiB limits. Maximum observed process-tree RSS was
361.8 MiB. The manifest preserves the original case definitions and revisions.

| Recheck outcome | Cases |
| --- | ---: |
| Complete static report | 7 |
| Partial graph | 96 |
| Artifacts inspected | 3 |
| Gated HTTP access failure | 4 |
| Invalid XTTS artifact | 1 |

All **106 successful reports** passed IR round-trip, graph validation, service
size/identity checks and zero-weight-payload/no-forward observations. The five
failed attempts have no validated report and no invented measurement. No new
crash, timeout, memory-limit or invalid-report outcome occurred. These checks
measure static compatibility, not inference quality, latency or task accuracy.

Supabase import read-back verified 105 new reports and reused one verified current
report, with zero failures. The fourth benchmark publication retains all 250
repositories and every previous observation. Its latest totals are **7 complete,
96 partial, 142 artifacts inspected and 5 failed attempts**. All **245 stored
report links** remain available. Historical reports retain their original engine
identities; automatic submitted-model cache lookup uses the current engine.

Evidence:

- [Pinned subset](benchmarks/model-matrix-250-profile-manifest.json)
- [Portable run observations and source identities](benchmarks/model-matrix-250-profile-rechecks.json)
- [Integrity-verified latest 250-report diagnosis](benchmarks/profile-operator-diagnosis.json)

Engine source SHA256: `8cbdbd67a7b17a82c075f4071a3b9f67b265c77d58d322e48e1a5d6f9aa1dc7a`.
The capture runner is the unchanged `scripts/store_benchmark_reports.py`; the
private archive preserves both disjoint capture manifests, records, merge
validation, source snapshot and storage receipts.

## What changed in the findings

T5-small warnings decreased from 25 to 13; Whisper-tiny decreased from four to
two. Both remain partial. ModernBERT now has actual chunk output ports `0` and
`1`, each `[B, S, 1152] / F32`, for its native MLP projection. Its unsupported
attention/transform paths remain visible.

Across the rechecks, 13 reports have fewer warnings and 50 have more. Nine prior
complete observations (GPT-2, Phi-2 and seven Qwen cases) are now partial because
missing output facts can no longer inherit a default hidden-state shape or
unproved tuple port. Original measurements remain in the catalog history.
Warnings and coverage fractions can worsen when previously hidden operations or
unknown facts are exposed; the new graph also changes the coverage denominator.
These counts are not evidence of universal support or improved task accuracy.

All 96 partial graphs now have a recorded coverage or ownership gap. Overlapping
blocker cohorts include 80 models with unproved tensor transforms, 67 with
unresolved branches, 62 with opaque calls, 10 with unresolved loops and 11 with
unowned checkpoint tensors. The cohort counts must not be added together.

Next contracts should address Qwen/GPT-2/Phi attention outputs, T5 native dtype
metadata and mask dispatch, spatial/window/padding transforms, task/component
input profiles and explicit disposition of legacy/extra checkpoint tensors.
The four gated repositories still need owner-approved Hugging Face access. XTTS
still needs a supported, valid format route; deleting non-finite config values
or inventing a checkpoint would change its evidence.

## Verification and live alignment

- Final offline backend/infra suite: **728 passed, 1 skipped, 18 integration tests
  deselected**. The skipped production meta-import memory gate requires Linux.
- Ruff, compilation and diff checks passed; frontend type checking and production
  build passed. Vite retains its existing large-bundle advisory.
- Live acceptance exercised all 56 catalog categories, all 250 rows/245 links,
  historical and current report reads, ownership isolation, a fresh pinned
  reconstruction, current-engine cache reuse and all six export formats.
- API `8000`, worker and preview `5176` run this branch against Supabase. Database,
  storage and worker health checks pass. Optional AI is unconfigured and returns
  its explicit unavailable response.
- The worktree preview now permits the two resolved font-package directories.
  Both WOFF2 assets return 200; unrelated files outside the workspace remain 403.
  Filesystem serving stays restricted.

- Final desktop/mobile browser suite: **32 passed** after the font fix. Live
  tensor inspection returns direct checkpoint evidence; both ModernBERT MLP
  template graphs expose the verified numeric chunk ports and shapes through
  the API.

The [private evidence archive receipt](benchmarks/profile-operator-archive.json)
records 19 read-back-verified objects, including the source snapshot, capture
inputs/records, merge validation, store receipts and release acceptance logs.
