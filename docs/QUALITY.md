# Quality and delivery gates

## Branch and PR policy

`main` is PR-only. Enable a GitHub ruleset or branch-protection rule for `main` that requires a pull request, the `quality` check, resolved conversations, and one approval when a second collaborator is available. Block force pushes, branch deletion, and normal bypasses.

Every PR states the change, its reason, deterministic verification, limitations, documentation impact, and next task. Topic branches use the `codex/` prefix by default.

## Verification

Use deterministic evidence before judgment:

- tests for behavior;
- compilation/type/lint checks when they apply;
- `git diff` to verify scope;
- a pinned benchmark before milestone and release completion.

The current `quality` workflow runs installation, bytecode compilation, and `pytest` on pull requests to `main`. Keep it passing and expand it with lint/type checks when those tools are added.

## Benchmark

Create a pinned manifest under `benchmarks/` before M5. For each model/revision, state expected direct facts, difficult cases, resource limits, and the expected handling (`success`, `partial`, or `unsupported`). Record model, revision, tool commit, environment, results, failures, and next action. Never silently revise expected answers or invent scores.

## Demo and submission

The primary demo is: paste primary model → inventory → generated graph → select node → show module, tensor metadata, parameter count, and evidence → optional evidence-bound AI explanation. Rehearse a local/saved fallback that does not misrepresent stale data as live.

Before submission, verify the deadline in Devpost, public repository and license, setup steps, working demo URL, public video of three minutes or less, NVIDIA model + Nebius service usage, and all credits/licenses. The release candidate must be tested from a clean checkout.
