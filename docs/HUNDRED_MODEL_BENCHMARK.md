# Cross-family model anatomy matrix

This matrix attempts real pinned Hugging Face repositories across language,
vision, multimodal understanding, speech, generative image/video and RL tasks.
The manifest contains 120 distinct repository IDs. Repeated configurations or
multiple checkpoints in one repository are not counted as additional models.

## Reproduction

Install the project's engine, API and development dependencies. From the
repository root:

```powershell
python scripts/model_matrix.py --manifest benchmarks/model-matrix-100.json --output .local/model-matrix/results.ndjson --summary .local/model-matrix/summary.json --resume
```

The public Hugging Face metadata API supplies repository revisions and file
inventories. `scripts/discover_model_matrix.py` can refresh the candidate list;
review the resulting manifest before a new benchmark. Refreshing revisions is a
new experiment. Explicit file URLs select verified Diffusers components or GGUF
variants, and must match the case's repository and immutable revision.

Each case runs in a fresh spawned process, serially, with a 120-second wall
deadline and a sampled 4 GiB process-tree RSS guard. On Windows this is not an
OS-enforced address-space cap; 50 ms RSS sampling can miss brief peaks. Process
startup, imports and teardown are included in wall time. The host is a development
machine, not a reserved performance environment. This is reconstruction
compatibility testing, not inference throughput or output-quality benchmarking.

NDJSON rows are flushed and synced as each case finishes. Resume identity includes
the pinned case, engine version, runner source hash, backend source-tree hash,
dependency versions, Python/platform and resource budgets. Do not edit either
source tree during a run. The original baseline runner recorded dependencies and
budgets but did not include them in resume identity; neither changed during it.

## What the checks establish

Successful reports undergo IR JSON round-trip equality, graph integrity checks,
public GraphResponse validation for every graph, one representative owned-tensor
NodeInspection sample and the artifact-response schema. Serialized reports must
fit the service's default 32 MiB storage contract. This does not exercise every
HTTP admission, quota, pagination, export or concurrent storage path; the service
test suite and existing HTTP benchmarks cover those separately.

Weight-payload and forward-execution fields are engine assertions checked by the
runner, supported by source review of bounded metadata/range reads, native meta
construction and static source analysis. The runner does not independently
instrument every network read or Python call. Access failures and killed children
do not produce a validated IR, so they have no successful-report evidence.

| Outcome | Meaning |
| --- | --- |
| `completed` | All four recorded static coverage ratios are one, no extraction warnings/unknown nodes/unmapped tensors, and report contracts pass. This is bounded static completeness, not proof of every possible execution path. |
| `partial` | A validated architecture report exists, with explicit semantic, shape, mapping or other uncertainty. |
| `inspected` | A validated graph-free artifact report exists. Depending on selection/support, this may be file inventory only or include verified tensor headers. |
| `http_access` / `network` | Upstream access or transport prevented reconstruction. Public repository metadata can be readable while gated config/checkpoints return 401. |
| `unsupported` | No supported reconstruction/inspection path completed. |
| `invalid_report` | A report contract or validation failed; retained as failure evidence. |
| `artifact_error` | Metadata/configuration/checkpoint validation failed. This alone does not establish that publisher artifacts are corrupt: an engine compatibility defect can raise the same exception. |
| `timeout` / `memorylimit` / `failure` | The case exceeded a resource guard or failed. Inspect the recorded exception; the baseline classifier uses broad exception categories. |

There is no combined "models supported" percentage: artifact inventory,
partial anatomy and complete static checks answer different questions.

## October 19 baseline

The baseline attempted all **120 distinct repositories** using engine
`2026.10.19.2` and Transformers `4.57.6`. The runner and engine source hashes
remained unchanged throughout the run.

| Outcome | Repositories |
| --- | ---: |
| Complete bounded static checks | 10 |
| Valid partial architecture | 43 |
| Valid artifact-only inspection | 52 |
| Validation/construction rejection | 11 |
| Late JSON serialization failure | 1 |
| Upstream access blocked | 3 |

[Baseline results](benchmarks/model-matrix-2026-10-19-baseline.json) retain every
repository/revision, category, status, measured coverage, bounded warning sample
and performance measurements. The eleven rejection rows use the earlier runner's
`invalid_report` label even though reconstruction failed before emitting an IR.
They consist of three SegFormer schedule rejections, five blocked audio
constructors and three foreign Diffusers configuration rejections. The JSON
failure was XTTS configuration containing a non-finite number. Llama, FLUX and
DeepFloyd component metadata returned 401; public inventory access does not grant
access to their gated files.

## Fixes and targeted rechecks

Engine `2026.10.19.3` addresses the measured compatibility defects:

- SegFormer stage head/depth schedules follow the native configuration contract;
  scalar text heads remain strict, and total stage depth receives the layer cap.
- Configurations outside registered Transformers types receive structural budgets
  without being judged against Transformers dimension fields. Selected diffusion
  components can reach verified header inspection; no pipeline graph is invented.
- Construction guards remain `UnsupportedModelError`, allowing the existing
  artifact-inspection fallback. No forward or meta-device guard is relaxed.
- A publisher root `auto_map` can be ignored only for explicitly declared classes
  matching the installed native mapping. Original config and an explicit warning
  remain: equivalence to publisher custom code is unverified. Remote-only and
  nested remote mappings still cannot execute.
- Non-finite JSON constants and numeric overflow fail at the metadata boundary
  instead of causing a late report serialization exception. Invalid upstream
  metadata remains an error.

The recheck manifest includes all fifteen baseline failure/access cases, three
root remote-map blockers and four text/vision/audio/multimodal controls. Reproduce:

```powershell
python scripts/model_matrix.py --manifest benchmarks/model-matrix-rechecks.json --allow-small --output .local/model-matrix/rechecks.ndjson --summary .local/model-matrix/rechecks-summary.json
```

[The 22 rechecks](benchmarks/model-matrix-2026-10-19-rechecks.json) returned two
complete, four partial and twelve artifact-only reports, all eighteen passing
their report contracts. Three gated cases still returned 401; XTTS now fails
explicitly on its `Infinity` JSON constant. All eleven prior configuration or
construction rejections now return validated reports. Phi-3 advances from
artifact-only inspection to a native partial graph.

Replacing the 22 baseline observations with those rechecks gives:

| Latest observation across 120 repositories | Count |
| --- | ---: |
| Complete bounded static checks | 10 |
| Valid partial architecture | 45 |
| Valid artifact-only inspection | 61 |
| Gated upstream files | 3 |
| Invalid upstream JSON | 1 |

This combines the unchanged baseline with targeted reruns. The other 98 cases
were not rerun on `.3`; it is not a new full-matrix run. No unexpected exception,
report-schema failure, storage overflow, timeout or sampled memory violation
remains in the targeted rechecks. Artifact-only results do not establish native
architecture support.

A separate official `no_init_weights()` experiment did not satisfy the existing
meta-device guard on the pinned Wav2Vec2 model. It was not adopted. Constructor
support for these audio families needs a reviewed native approach; converting an
unsupported constructor into artifact inspection is not complete audio anatomy.

## Verification and running API

Final backend/infrastructure run: **427 passed, one Linux-only resource test
skipped on Windows**, with all eighteen real Hugging Face/HTTP integrations
enabled. Ruff, Python compilation and diff whitespace checks passed. The existing
Starlette/httpx deprecation warning remains.

The first Linux CI run subsequently found a benchmark cleanup defect: using
`psutil.wait_procs` on the multiprocessing-owned child could consume its POSIX
exit status before `Process.join()`. Cleanup now lets multiprocessing terminate
and reap its own child, using psutil only for descendants. Nine runner unit tests
pass, including an ownership regression, and a fresh real SmolLM2 control passes
all static/report checks after the fix. See the
[post-fix runner smoke](benchmarks/model-matrix-2026-10-19-runner-smoke.json) and
[PR CI status](https://github.com/pranavsinghpatil/model-anatomy/pull/32).

The 120 baseline and 22 targeted observations retain their original runner hashes
(source represented by commit `365e7f4`). None of those cases required resource
termination. The subsequent runner-only cleanup change does not alter extraction
engine `.3` or rewrite the recorded outcomes.

The owned API and worker were restarted on port **8000**, preserving existing
analyses and backing up SQLite first. Fresh uncached HTTP jobs verified Phi-3
(3,821,079,552 parameters, 195 tensors) and SegFormer B0 (3,752,694 parameters,
208 tensors), persisted partial reports, root graph and owned-tensor inspector
schemas, JSON exports, exact revisions and `.3` engine identity. Both have full
checkpoint parameter mapping. SegFormer summary heads remain `[1,2,5,8]`, with
eight layers. See [live HTTP evidence](benchmarks/model-matrix-2026-10-19-live-http.json).

The baseline's 105 emitted reports all passed their contracts. Median case wall
time was 14.336 seconds; maximum was 35.786 seconds. Maximum sampled process-tree
RSS was 379,473,920 bytes, and the largest serialized report was 3,974,494 bytes.
These are local observations under the stated policy, not a production SLA.

## Results and follow-up

The committed result snapshot records measured counts, environment and individual
outcomes. Keep the original baseline distinct from targeted reruns after fixes.
See [family-specific reconstruction strategies](MODEL_FAMILY_STRATEGIES.md) for
the acceptance work needed beyond the general native Transformers route.
