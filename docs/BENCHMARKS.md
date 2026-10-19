# Verification and Benchmarks

The [cross-family matrix](HUNDRED_MODEL_BENCHMARK.md) adds 120 distinct pinned
repository attempts. It measures reconstruction compatibility and report contracts;
the HTTP measurements below remain a separate service-performance experiment.

## Current Real-Job Run: October 12

Engine `2026.10.12.2`, Python 3.11.4, Windows, SQLite/local storage and a
separate worker. An isolated service on port 8002 used a fresh data directory,
not the user's analyses. No runtime/provider/report mocks or prepared examples
were used. All tested reports had four coverage ratios of 1, no extraction
warnings, no forward execution and zero weight payload bytes read.

| Measurement | Observed |
| --- | ---: |
| GPT-2 fresh-process CLI inspection | 8.54 seconds |
| GPT-2 peak sampled process-tree RSS | 314.49 MiB |
| SmolLM2-135M fresh-process CLI inspection | 10.44 seconds |
| SmolLM2 peak sampled process-tree RSS | 316.11 MiB |
| First SmolLM2 HTTP job, queue/network/storage included | 8.66 seconds |
| Same-revision cached HTTP job, queue/resolution included | 1.90 seconds |
| Independent visitor sessions / measured reads | 10 / 200 |
| Read median / p95 / maximum | 89.1 / 335.6 / 1,572.0 ms |
| Checkpoint weight payload downloaded | 0 bytes |

The script admitted twelve visitor-owned jobs: two primary-model admissions
for the first visitor, one for each of nine more visitors, and a separate fresh
GPT-2 job. Each visitor read its own analysis, root graph and component, plus
health. All 200 measurements fell wholly between the first and last status
polls reporting GPT-2 as Running, and all four routes were measured for every
visitor. The final GPT-2 result was completed and not cached. The last-running
to completed polling bracket was 108 ms; polling is not an exact measurement
of the worker's state-transition instant.

[Machine-readable current results](benchmarks/local-windows-2026-10-12.json)
record model SHAs, admissions, coverage, timings and status polls. The p95
target of less than one second passed locally. It is not a production SLA.
Fresh process does not mean cold provider or OS caches; neither was flushed.
RSS was sampled every 25 ms and may miss brief peaks. The development host was
not reserved exclusively for benchmarking; regression work also ran that day.
Windows does not enforce the Linux worker address-space limit.

To reproduce, start an isolated API and worker with the same **new** data
directory and enough quota. Never reset a user's data to make a benchmark cold.

```powershell
uv run --no-sync python scripts/benchmark.py --base-url http://127.0.0.1:8002 --output .local/benchmark-new-run --inspect --submit --visitors 10 --rounds 20
```

HTTP admissions require `--submit` and consume actual quotas. A fresh GPT-2
cache miss is required; the script fails rather than claiming work was running
if that revision is already cached. It also fails for insufficient in-window
visitor/route coverage or an unmet latency target. Use another fresh isolated
data directory for a repeat, leaving prior evidence intact.

## Historical Local Run: October 8

On October 8, 2026, Python 3.11.4 on Windows, the application used a real API,
SQLite/local storage, a separate worker and Hugging Face network metadata. No
provider responses or extraction results were mocked in this benchmark. No
Nebius inference or Supabase connection was available.

```powershell
uv run --no-sync python scripts/benchmark.py --inspect --submit --rounds 20
```

The script starts fresh CLI processes, samples their process-tree RSS every
25 ms, submits two real SmolLM2 jobs through one visitor, verifies the second
uses the exact-revision cache, and measures ordinary HTTP reads after observing
a running analysis. Submission is opt-in because it consumes real job quotas.

| Measurement | Observed |
| --- | ---: |
| GPT-2 inspection | 14.31 seconds |
| GPT-2 peak sampled process-tree RSS | 313.54 MiB |
| SmolLM2-135M inspection | 16.00 seconds |
| SmolLM2 peak sampled process-tree RSS | 316.30 MiB |
| New SmolLM2 analysis, including queue/network/storage | 13.56 seconds |
| Reused SmolLM2 analysis, including queue/revision resolution | 4.75 seconds |
| HTTP visitors / requests | 10 / 200 |
| HTTP median / p95 / maximum | 29.5 / 88.8 / 123.5 ms |
| Checkpoint weight payload downloaded | 0 bytes |

The HTTP mix includes health, capabilities, analysis summary, root graph and
embedding component details. The benchmark setup warmed prepared views in the
then-current implementation; those example views were removed by D-023.
Cold process does not mean cold provider or OS caches; those were not flushed.
Sampling can miss brief RSS spikes. These are development-machine results, not
Nebius VM capacity or a production SLA. The test suite was also running during
the recorded run, so the machine was not idle.

[Machine-readable results](benchmarks/local-windows-2026-10-08.json) include exact
model SHAs and conditions. SmolLM2's valid report is `partial`, not falsely
complete, because some intermediate shapes remain unknown.

## Historical Performance Fix

These measurements describe the pre-D-023 implementation and are historical;
they are not evidence for current example-free runtime performance.

An initial development run missed the target: ten visitors, 200 requests and
p95 **3.87 seconds**. Prepared views reparsed full IR and deep-serialized it just
to calculate a summary on every request. The historical fix kept at most two
read-only prepared snapshots, invalidated them by file stat identity, and
calculated summaries from existing metadata. D-023 later removed prepared views
from the product, so these optimization results do not describe current runtime
behavior. A regression test checked invalidation and that graph/component reads
did not mutate cached IR.

[Before-cache diagnostic](benchmarks/before-prepared-cache.json) used a smaller
three-endpoint mix. It is retained as diagnostic evidence, not a directly
equivalent production baseline. The optimized run includes five endpoints.

## Current Verification

Final engine run: 222 backend/infrastructure tests passed, one Linux-only
memory test skipped on Windows and one existing Starlette deprecation warning.
Four pinned model engine cases and fresh HTTP-worker cases used real metadata.
The frontend build/types, formatting and generated OpenAPI drift check passed;
62 deterministic browser checks and six final real desktop/mobile checks passed.
Container images cannot be built on this Windows host because Docker is absent.
CI builds all three images and runs the pinned-model Linux-container smoke; record
that CI result separately before claiming the container path is verified.

## Historical Test Evidence

- Windows: 98 tests passed, including both fresh pinned Hugging Face targets;
  one actual Linux memory-enforcement test skipped.
- Service PR Linux CI: 87 tests passed, including real Torch/Transformers imports
  and a tiny meta GPT-2 under the 4 GiB virtual-address limit; two optional
  Hugging Face tests skipped. These counts predate the current infra tests.
- Ruff, Python compilation and whitespace checks passed.
- Browser: an independent full run passed 38 checks (34 scenario checks and four
  real desktop/mobile flows), including live jobs, nested graphs, evidence,
  downloads, full initial framing and no viewport overlap. No live AI was used.
- Production browser build, formatting and generated OpenAPI drift checks pass;
  npm audit reports zero known advisories across runtime/build dependencies.
- Container build and pinned-model smoke checks run in CI, since Docker is absent
  on the local host. They deliberately use SQLite/local storage without secrets;
  production PostgreSQL, Supabase and NVIDIA inference remain separate gates.

## Remaining Release Gates

Before public release, repeat the benchmark against the Nebius CPU VM, run both
primary models under its worker/container budget, verify PostgreSQL locking and
private Supabase object operations, and record a real NVIDIA Nemotron answer.
Test external HTTPS access, recovery and judging availability. Local numbers
alone do not complete these gates.
