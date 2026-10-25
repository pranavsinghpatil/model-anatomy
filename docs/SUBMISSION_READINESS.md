# Submission Readiness

**Status: not ready to submit as of October 25, 2026.** The current evidence does
not establish a working public demo or a live Nebius/NVIDIA inference call through
the application. The release must close the gates below before anyone submits.
This document records evidence and actions; it does not authorize publishing,
provisioning, or changing repository visibility.

The official [Devpost rules](https://nebiusglobalaihackathon.devpost.com/rules)
and event API were checked October 25. Submissions were open with a deadline of
**October 30, 2026 at 17:00 UTC (22:30 IST)**. Judging ends **December 15,
2026 at 20:00 UTC**. Recheck the event page before submission.

The sanitized audit evidence record tracks the exact baseline and verification
results. See the [JSON record](evidence/submission-readiness-2026-10-25.json).

## Current Blockers

| Priority | Gate | Evidence now | Exit evidence |
| --- | --- | --- | --- |
| P0 | Judge-accessible demo | The known Vercel URL `https://model-anatomy-eight.vercel.app/` returned HTTP 404 on October 25; its API health paths returned 404. No alternate working public demo URL is established by this audit. | Integrated release is reachable over HTTPS from an external network; new visitor completes primary model flow and can inspect graph/tensors/evidence; judge access remains available through Dec 15. Record URL, UTC check time, and sanitized result. |
| P0 | Public open-source repository | GitHub reported `pranavsinghpatil/model-anatomy` private and default `main` had no detected license on October 25. The baseline checkout has an MIT `LICENSE`, but that does not establish public release source. | Intended release source is public without authentication; default branch contains license and GitHub detects it; setup and platform/model usage are documented. Record repository URL and check date. |
| P0 | Runtime platform and NVIDIA model | No live Nebius inference through the integrated app has been verified. A model-catalog/preflight request is not inference evidence. | Complete an explanation request in the deployed application using an exact NVIDIA open-source model on Nebius Token Factory, or document qualifying Nebius AI Cloud compute use. Keep sanitized request/model/service/response evidence, citations checked against IR, before/after IR digest, and a useful engineering outcome. |
| P0 | Public demo video | No public video link is recorded. | English YouTube video publicly accessible and shorter than 3 minutes; record link and duration. Footage matches the release that judges can access. |
| P1 | Integrated release and CI | Release identifier and reviewed integration are pending. Local changes and test status are tracked by the implementation workstreams. | Merged reviewed release commit/tag; required CI passes for that revision; record PR and check links. Do not infer a pass from workflow configuration. |
| P1 | Quotas, ownership, and recovery | These are required by the deployment checklist; no public production verification is recorded here. | Exercise guest ownership/isolation, approved quotas (2 jobs/hour and 10 questions/hour per guest; 20 jobs/day and 100 questions/day global), 10 waiting jobs, timeout/failure recovery, and worker restart on deployed services. Record results without personal data. |
| P1 | Budget and judging availability | No hosting budget or end-of-judging availability commitment is recorded here. | Confirm service credits/billing, set operator budget alerts, and arrange free judge access that remains usable through Dec 15. Monitor uptime and quota headroom. |
| P2 | Submission claims and evidence | Draft language and local benchmarks exist, but deployment and inference evidence do not. | Reconcile form/video wording with measured deployed behavior, limitations, licenses, and attributions. Provide platform/tool feedback based on observed use. Submit under Best Apps and Agents and retain receipt. |

## Release Sequence

1. Integrate and review the release candidate; verify required CI and record its commit.
2. Publish the intended source repository with its license on the public default branch; verify anonymous access and license detection.
3. Deploy the integrated release, then test HTTPS and the complete primary model flow externally. Verify the actual runtime platform and model service.
4. Run one useful evidence-bound Nemotron explanation through the application. Preserve sanitized evidence, check citations and numeric claims against the IR, and record that prose is not independently verified. Include an insufficient-evidence question and ensure the IR digest is unchanged.
5. Verify judge access, quotas, isolation, migration, failure recovery, private storage, browser credential boundaries, and service health. Establish a budget and availability monitoring through judging.
6. Record a public YouTube demo under three minutes from the deployed release. Ensure the video does not imply unverified features or treat cached results as new inspections.
7. Recheck Devpost deadlines and requirements; complete the form with actual evidence, submit by October 30 at 17:00 UTC, and retain the receipt.

## Claim Boundaries

The current local evidence supports model reconstruction and local application
behavior as documented in [benchmarks](BENCHMARKS.md) and [live backend
acceptance](LIVE_BACKEND_ACCEPTANCE.md). It does not establish production hosting,
public judge access, live Nebius inference, or submission completion.

The application validates AI citations and numeric claims against supplied
reconstruction evidence; explanatory prose is not independently verified. The
AI layer cannot establish or alter graph facts. State these limits wherever the
integration is described. A successful catalog lookup or preflight is not a
runtime inference call. A local benchmark is not a deployed performance result.

Keep credentials, signed URLs, billing details, and unrelated visitor data out of
public evidence. Do not change any status from pending until the exit evidence
has been collected and reviewed.

## Code Audit and Repair Evidence

The branch advances engine/cache identity to
`2026.10.25.2-submission-audit`; older saved reports remain immutable.
The audit restored guest/global quota rejection with an atomic write recheck
and a read-only availability precheck, restored prior header/name audits in
analysis summaries, retained provenance for native checkpoint renaming, and
made missing-checkpoint compatibility explicit.

Report UI completeness now requires four explicitly successful architecture
checks. Historical reports with missing checks, engine version or mode display
those unknowns instead of implying a current complete architecture. AI answers
carry their interpretation warning, and pinned tensor links stay within the
reported Hugging Face repository/revision.

The clean checkout now includes digest-documented regression fixtures, separate
from runtime analyses. The historical validation CSV is kept separately from
Van's later run. Dormant activation/source contracts remain fail-closed when
the pinned library source differs; tests for the retired expression graph were
replaced with current module/topology/checkpoint/completion coverage. No source
fingerprints were reapproved just to satisfy tests.

Final deterministic checks so far: **838 backend/infrastructure tests passed
(25 opt-in/platform skips)** and **214 browser tests passed (34 environment-gated
skips)**. Ruff, compilation, lock consistency, API type regeneration, frontend
typecheck/build/formatting and diff checks passed. The Python environment scan
and npm audit report no known advisories within their documented scan scope.
The wheel includes both runtime audit manifests and excludes tests/frozen reports.
Vite reports a nonblocking chunk-size advisory; deployed network performance
and local Docker execution are unverified.

Three real isolated HTTP acceptance cases passed, including the pinned
SmolLM2-135M cold/cache flow, all 30 registered layer instances, tensor evidence,
exports, visitor isolation and quotas. Fresh pinned GPT-2 and SmolLM2-135M engine
checks also passed with zero weight payload bytes read. The final wheel was
rebuilt with the current engine identity and its package contents inspected.
Real desktop and mobile browser submissions also passed against the current
API/worker: SmolLM2-360M resolved to `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`,
reported 361,821,120 parameters, exposed owned tensors and evidence, and exported
without browser errors. One cold and two cached runs were observed, including a
desktop screenshot capture. Temporary test services were stopped. PR CI remains
separate from these local results. These checks do not close the external
submission gates above.

The first Linux PR run exposed three audit-binding failures that Windows did
not: Git had normalized the pinned audit manifests from CRLF to LF. The repair
preserves the original manifest bytes and matching documentation copies through
Git attributes. JSON content and both approved digest constants are unchanged.
Final-head Linux CI is tracked in PR #58 separately from the local results.

The first PR container job passed: all three Linux images built and the real
pinned primary-model acceptance completed. The new Vercel preview deployment
also reports success, but anonymous requests redirect to Vercel login/SSO.
A green frontend deployment therefore does not establish judge access or a
working public API/worker. The original listed production URL still needs a
working replacement or repair.

The Linux browser run initially passed 213 cases and failed one responsive
geometry check: a transient zero-height measurement produced NaN. That check
now waits for positive finite dimensions and uses the same captured snapshot;
the original position and drawing-area thresholds remain unchanged. The final
desktop/mobile responsive cases passed locally.

Autter also flagged valid first-party imports as absent external packages and
alleged a quota race without accounting for the database write mutex. Import
resolution/package inspection confirms the local modules. Four real concurrent
20-request SQLite bursts, across four service instances, each admitted exactly
3 and rejected 17 with accurate guest/global counters. PostgreSQL locking was
source-reviewed separately. Evidence is included in the PR for reviewer
classification; the automated review gate has not been disabled.
