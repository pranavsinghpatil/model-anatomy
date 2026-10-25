# Teammate handover — October 25, 2026

## Branch and PR order

The integrated application is on `feat/vv`, including Van's merged PR #57.
`main` is an older baseline. Do not start new application work from `main`.

1. Review [PR #58](https://github.com/pranavsinghpatil/model-anatomy/pull/58),
   `codex/submission-readiness` into `feat/vv`.
2. [PR #59](https://github.com/pranavsinghpatil/model-anatomy/pull/59),
   `autter/fix-updates` into `codex/submission-readiness`, is a dependency
   follow-up to #58. Its original proposal against old `main` admitted versions
   affected by its own listed advisories. It now raises the engine Torch floor
   to `2.10.0`, preserves `transformers==5.20.2`, and checks both the manifest
   and every platform lock entry. After #58 merges, target #59 at `feat/vv`.
3. Promote the reviewed application to `main` through a separate PR when the
   team is ready. No direct commits to `main`.

## Checks and review evidence

Use the exact setup and commands in [AGENTS.md](../AGENTS.md). Python tests
require the API extras; browser specs use Playwright from `frontend`.
The GitHub `quality`, `browser`, and `containers` jobs passed for #58 at
`53e5e2b`; see [the CI run](https://github.com/pranavsinghpatil/model-anatomy/actions/runs/40195549006).
Check the latest commit's jobs before merging further updates.

The dependency guard fails against the previous `torch>=2.8` constraint and
passes after raising the floor. `uv lock --check --offline` passes; the resolved
versions remain Torch `2.14.1` / `2.14.1+cpu` and Transformers `5.20.2`. The floor
addresses the reviewed advisories, including
[PYSEC-2026-2286](https://osv.dev/vulnerability/PYSEC-2026-2286), and is not a
claim of universal security or a replacement for future advisory scans.

The review follow-up adds a concurrent SQLite quota regression: four service
instances, twenty simultaneous requests, both guest/global limits and
analysis/question actions. With two of three credits already spent, exactly
one request succeeds and nineteen receive 429; persisted counters remain three.
PostgreSQL uses the shared `Mutex` row lock; it was source-reviewed, not tested
against a live PostgreSQL service in this audit.

Autter's import warnings refer to the first-party `model_anatomy` package.
The flagged numbers in the pinned DeepSeek header manifests are shard byte
sizes. Preserve those manifests' exact bytes and `.gitattributes`: their digests
are part of the reconstruction evidence contract.
Autter's reported test failures used missing Python extras and a Node runner
for Playwright files; correctly configured GitHub CI passed. The owner disabled
the repository in Autter during handover. On October 25 GitHub still required
`autter/review-gate` on `feat/vv`; disabling the integration does not remove that
branch-protection requirement. The owner must choose the replacement review
policy before this can be considered merge-ready.

## Submission gates still open

The code audit does not certify a submitted or publicly deployed release.
Complete the evidence in [Submission Readiness](SUBMISSION_READINESS.md):

- Working public frontend, API, and worker with free judge access.
- Public source with the license on the intended default branch.
- A real Nebius/NVIDIA inference request through the deployed app.
- Public English YouTube demonstration shorter than three minutes.

The recorded deadline is October 30, 2026, 17:00 UTC / 22:30 IST; recheck the
official rules. Judge access must last through the judging period.
Do not commit credentials, weight payloads, or private service logs.

## Local workspace caution

`D:\neb-hf` contains unrelated uncommitted work. Preserve it. The audit work is
in the `submission-readiness` managed worktree. Pull the intended branch and
follow the locked setup before continuing development.
