# Quality and delivery gates

## Branch and PR policy

`main` is PR-only. The baseline GitHub protection requires a pull request, the `quality` check, resolved conversations, no force pushes or branch deletion, and no administrator bypass. It uses zero required approvals so a solo maintainer is not locked out; raise the threshold to one when an active collaborator can review every PR.

Every PR states the change, its reason, deterministic verification, limitations, documentation impact, and next task. Topic branches use the `codex/` prefix by default.

## Verification

Use deterministic evidence before judgment:

- tests for behavior;
- compilation/type/lint checks when they apply;
- `git diff` to verify scope;
- a pinned benchmark before milestone and release completion.

The `quality` workflow runs for pull requests and pushes to `main` and the active
integration branch, `feat/vv`. It checks locked installation, compilation, backend
and infrastructure tests, Ruff, generated API types, frontend build/formatting,
desktop/mobile browser tests, and the bounded container smoke flow. Opt-in live
provider tests still require separate evidence; green CI alone does not certify
public deployment or NVIDIA inference.

## Benchmark

Create a pinned manifest under `benchmarks/` before M5. For each model/revision, state expected direct facts, difficult cases, resource limits, and the expected handling (`success`, `partial`, or `unsupported`). Record model, revision, tool commit, environment, results, failures, and next action. Never silently revise expected answers or invent scores.

## Demo and submission

The primary demo is: paste primary model → inventory → generated graph → select node → show module, tensor metadata, parameter count, and evidence → optional evidence-bound AI explanation. Rehearse a local/saved fallback that does not misrepresent stale data as live.

Before submission, verify the deadline in Devpost, public repository and license, setup steps, working demo URL, public video of three minutes or less, NVIDIA model + Nebius service usage, and all credits/licenses. The release candidate must be tested from a clean checkout.
