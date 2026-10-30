# Hackathon Submission

Status: **not ready to submit**. This requirements and evidence record does not
claim that the application is deployed or that a submission has been published.
Use [submission readiness](SUBMISSION_READINESS.md) as the prioritized release
gate. Update pending rows only when the corresponding evidence exists.

## Official Requirements

The official [Devpost rules](https://nebiusglobalaihackathon.devpost.com/rules)
and [submission API](https://nebiusglobalaihackathon.devpost.com/) were checked
on **October 25, 2026**. The API reported `submissions_open` and a deadline of
**October 30, 2026 at 17:00 UTC (10:00 PDT / 22:30 IST)**. Judging runs
**December 1–15, 2026**, ending **December 15 at 20:00 UTC (12:00 PST)**.
Recheck the official page before submitting in case the event changes its dates.

The official rules state:

> Entrants must create a working software application that runs on either Nebius Token Factory or Nebius AI Cloud and uses at least one NVIDIA open source model
>
> The repository must be public and open source by including an open source license file (such as Apache 2.0, MIT, or MPL 2.0).
>
> should be less than three (3) minutes. Judges are not required to watch beyond three minutes

The runtime definition accepts a Token Factory inference API call or qualifying
Nebius AI Cloud compute. Our release gate uses the integrated Nemotron
explanation path, a public YouTube video under three minutes, and a working
demo with free judging access through the judging period.

The selected track is **Best Apps and Agents**. Judging criteria include
technological implementation and meaningful Nebius/NVIDIA use, coherent design,
a specific audience and impact, and a creative or non-obvious idea. The final
submission must identify the track, explain functionality and platform/tool
feedback, and describe significant new work if the project predates the event.

The repository contains an MIT `LICENSE`, but the connected GitHub check on
October 25 reports the repository private and its default `main` branch without
a detected license. A license file on a non-public branch does not satisfy the
public repository requirement. Public access and the default-branch license
must be verified after publication; do not claim this requirement is met yet.

## Description Draft

Model Anatomy helps engineers inspect a specific open-weight Hugging Face model.
It reconstructs a semantic, tensor-aware architecture from model configuration,
module structure, source analysis, and Safetensors metadata. The generated graph
lets an engineer inspect components and trace facts to their evidence. Unknown
or unsupported behavior is explicit.

The application has an optional NVIDIA Nemotron explanation path configured for
Nebius Token Factory. It is intended to answer questions about an already
reconstructed model using selected IR facts and provenance. The service validates
citations and numeric claims against supplied evidence, but explanatory prose is
not independently verified. The model cannot create or modify the graph. No
live inference through the integrated application has been verified; do not
present the integration as demonstrated until runtime evidence is recorded.

## Links and Evidence to Complete

| Deliverable | Current evidence | Required action |
| --- | --- | --- |
| Working HTTPS demo and free judge access | The listed `https://model-anatomy-eight.vercel.app/` returned HTTP 404 on October 25; API health paths also returned 404. No alternate working demo URL is established by this audit. | Deploy the integrated application, verify it from an external network, and keep a usable judge path available through December 15. |
| Public repository with detectable license | `https://github.com/pranavsinghpatil/model-anatomy` reported private on October 25; default `main` had no detected license. The checkout contains an MIT `LICENSE`. | Make the intended release source public with the license on the default branch, then verify unauthenticated access and GitHub license detection. |
| Public YouTube demo under 3 minutes | No video URL or demo asset is recorded. | Record an English demo under 3:00 against the integrated release and publish it publicly. |
| Integrated release identifier | Pending reviewed integration and release PR. | Record the merged commit/tag and the PR links after review. |
| Deployment and benchmark evidence | Local benchmark evidence is recorded in [benchmarks](BENCHMARKS.md); deployed checks are pending. | Record production smoke, primary-model flow, quotas/recovery, and deployed performance results. |
| Nebius/NVIDIA usage and feedback | No live inference call through the integrated explanation path is verified. | Capture sanitized runtime evidence and actual user-facing value; record observed platform/tool feedback. |
| Track and final submission | Best Apps and Agents selected; no submission receipt. | Complete the form with measured claims, submit before deadline, and retain its receipt. |

## Meaningful NVIDIA Evidence

Keep a small sanitized evidence bundle for one end-to-end question: pinned model
revision and IR digest, selected component/evidence IDs, exact configured NVIDIA
model ID, Token Factory service, UTC time, response model/request ID if provided,
latency and token usage, returned explanation, and a citation-to-IR comparison.
Record how the answer improves an engineer's inspection task. Compare the IR
digest before and after the answer to show reconstruction was not modified.
Include a question with insufficient context and show the honest failure path.
Verify factual citations and numeric claims against the supplied IR; state that
explanatory prose is not independently verified.

Catalog availability alone is insufficient evidence of inference or product
value. CPU VM hosting alone is insufficient evidence of NVIDIA model use. Only
record actual calls made through the integrated application's explanation path;
the read-only preflight performs no inference. Do not include keys, connection
URLs, signed object URLs, raw user prompts unrelated to the demo, or billing
account details in published evidence.

## Video Plan (2 Minutes 45 Seconds)

| Time | Show |
| --- | --- |
| 0:00-0:20 | The model inspection problem and working HTTPS application |
| 0:20-1:05 | Submit a pinned supported model, show progress and generated graph |
| 1:05-1:40 | Inspect attention/MLP tensors and follow provenance to artifacts |
| 1:40-2:20 | Ask an evidence-bound question; show NVIDIA model identity, explanation, and citations |
| 2:20-2:45 | Show an honest limitation, repository/license, and concrete user benefit |

Use footage from the integrated release and label real cache hits accurately.
The runtime has no prepared reports. Record a submitted primary model analysis
with its actual pinned revision, graph, tensors and exports. Historical CLI
outputs are regression evidence, not images or substitute application results.

## Submission Gate

Complete the criteria in [submission readiness](SUBMISSION_READINESS.md), then
confirm a new visitor can open the public URL, run the advertised flow, inspect
evidence, and obtain the demonstrated NVIDIA answer. Verify ownership and quota
checks, failure recovery, migrations, private Storage, no credentials in browser
assets, public source/license visibility, setup instructions, and third-party
attributions. Verify the approved limits: 10 waiting jobs, guest jobs 2/hour and
questions 10/hour, and global jobs 20/day and questions 100/day. Global quotas
must use daily windows; the running job does not occupy a waiting queue slot.
Record an operating budget and monitor availability through the end of judging.
Confirm the final text and video match measured functionality. Submit through
Devpost before the deadline and retain the submission receipt.
