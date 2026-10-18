# Hackathon Submission

Status: preparation checklist. No public deployment or NVIDIA inference has been
verified by this deployment workstream. Fill links and evidence only after the
integrated application passes [deployment checks](DEPLOYMENT.md).

## Official Requirements

Checked against the [official rules](https://nebiusglobalaihackathon.devpost.com/rules)
on October 12, 2026. Submission closes **October 30, 2026 at 10:00 PDT**
(17:00 UTC / 22:30 IST). Judging runs **December 1-15, 2026**, ending at
12:00 Pacific Time on December 15. Recheck the rules before submission.

- Supply a working public demo URL and keep free judging access available through judging.
- Provide a public licensed repository containing code/assets, setup instructions, and clear platform/model usage.
- Publish an English demo video on YouTube lasting **less than three minutes**.
- Use Nebius at runtime and at least one NVIDIA open-source model meaningfully.
- Identify the track, describe functionality, and supply platform/tool feedback; describe significant new work if the project predates the event.

Approved track: **Best Apps and Agents**. Publication and the submission form
remain release tasks. The root MIT license covers repository code; third-party
software and model artifacts retain their own licenses.

## Description Draft

Model Anatomy helps engineers inspect a specific open-weight Hugging Face model.
It reconstructs a semantic, tensor-aware architecture from model configuration,
module structure, source analysis, and Safetensors metadata. The generated graph
lets an engineer inspect components and trace facts to their evidence. Unknown
or unsupported behavior is explicit.

The implemented NVIDIA Nemotron explanation path uses Nebius Token Factory to
answer questions about an already reconstructed model. Its context is restricted
to selected IR facts and provenance. It must cite existing evidence and report
insufficient evidence when a question exceeds that context. It cannot create or
modify the graph. Live NVIDIA calls have not been verified in this workspace;
record actual runtime evidence before presenting the integration as demonstrated.

## Links to Complete

| Deliverable | Status |
| --- | --- |
| Working HTTPS demo | Pending deployment and external-network test |
| Public repository | [model-anatomy](https://github.com/pranavsinghpatil/model-anatomy) is currently private (verified October 12); publication and detectable release license pending |
| Public YouTube demo, under 3 minutes | Pending recording |
| Integrated release identifier | Pending reviewed PR integration into `b-mvp` and separate release PR |
| Deployment/benchmark evidence | [Local benchmark](BENCHMARKS.md) recorded; deployed evidence pending |
| Nebius/NVIDIA feedback | Pending observed integration results |

## Meaningful NVIDIA Evidence

Keep a small sanitized evidence bundle for one end-to-end question: pinned model
revision and IR digest, selected component/evidence IDs, exact configured NVIDIA
model ID, Token Factory service, UTC time, response model/request ID if provided,
latency and token usage, returned explanation, and a citation-to-IR comparison.
Record how the answer improves an engineer's inspection task. Compare the IR
digest before and after the answer to show reconstruction was not modified.
Include a question with insufficient context and show the honest failure path.

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

Confirm a new visitor can open the public URL, run the advertised flow, inspect
evidence, and obtain the demonstrated NVIDIA answer. Verify ownership and quota
tests, failure recovery, migrations, private Storage, no credentials in browser
assets, open-source license visibility, setup instructions, and third-party
attributions. Verify the approved limits: 10 waiting jobs, guest jobs 2/hour and
questions 10/hour, and global jobs 20/day and questions 100/day. Global quotas
must use daily windows; the running job does not occupy a waiting queue slot.
Record an operating budget and monitor availability through the
end of judging. Confirm the final text and video match measured functionality.
Submit through Devpost before the deadline; retain the submission receipt.
