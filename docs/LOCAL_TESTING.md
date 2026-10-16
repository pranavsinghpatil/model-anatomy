# Test the Local Application

This is the current first-release checkpoint, not a claim that every model or
runtime variant is supported. Cloud accounts are not required for these checks.
Internet access to Hugging Face is required. Use the same browser for a job: its
visitor cookie owns the report. Another browser cannot open that job's URL.

## Open and Analyze

1. Open [the app](http://127.0.0.1:5173/). A new visitor starts empty.
2. Enter `HuggingFaceTB/SmolLM2-135M` and select **Analyze Model**.
3. Progress should leave Queued, move through inspection stages and finish with
   a graph. If it fails, the app must show an error, not an example report.
4. Open a repeated Transformer block, then attention and its projections.
5. Select a component and inspect its dimensions, parameter count, tensors and
   evidence. Download JSON or the complete ZIP from the export control.
6. Reload the page. Your report should remain accessible. Submit the same model
   again: an unchanged revision should reuse the real report and show Cached.

The default allowance is two submissions per visitor per hour, twenty globally
per day. Cached submissions also consume an admission. Visitor jobs expire after
24 hours; real cached reports are retained by default and reused when their
immutable model/engine/schema/artifact identity is unchanged. An expired link is
not a broken engine and must not display another visitor's or a replacement report.

For manual exploration beyond those defaults, stop the existing API, worker and frontend
and run `powershell -ExecutionPolicy Bypass -File scripts/start-local.ps1` from
the repository root. The script starts hidden local processes, preserves the
existing `.local/app-v1` database and reports, and applies bounded local limits:
100 jobs per visitor/hour, 500 jobs globally/day, 100 AI questions per
visitor/hour, and 500 AI questions globally/day. Parameters may set each limit
from 1 to 1,000. It refuses to attach to listeners already using ports 8001/5173 or a
running worker lease, because those processes cannot receive new settings. Existing
quota records are preserved and continue to count until their normal windows
expire. The regular settings defaults remain 2/20 jobs and 10/100 AI questions.
Launcher logs are written to `.local/app-v1/local-testing/`.

## Debugging a Failed Analysis

For a readable local terminal view, use the API environment's Python:

```powershell
python -m model_anatomy.service.debug --data-dir .local/app-v1 --follow
```

It reads API and worker JSON events and follows new entries. Routine polling
timings are hidden by default; `--requests` includes them. Ctrl+C stops the
viewer, not the API/worker. To inspect one saved analysis and its event history:

```powershell
python -m model_anatomy.service.debug --data-dir .local/app-v1 --analysis-id ANALYSIS_ID
```

This local operator command reads SQLite in read-only mode, verifies the private
report's content hash, and prints the same recorded partial checks and warnings
as the report, plus root relationship counts and isolated components. It does
not contact Hugging Face, execute a model, clear a cache or change saved reports.
It is not a public endpoint and requires local filesystem access. It supports
the local SQLite/private-files profile, not hosted Supabase diagnostics.
`--log-dir .local/app-v1/local-testing` selects logs from the regular launcher.
An immutable cached report retains its existing gaps; a fast cache hit is not
evidence that extraction issues were fixed.

Text following is best-effort, not an exactly-once audit stream. File identity,
prefix/boundary checkpoints and size changes detect normal rotation/truncation;
concurrent rotation can repeat records. Persisted jobs and immutable reports
remain the source of truth. Malformed oversized log records stop the viewer
explicitly instead of consuming unbounded memory.

The October 13 operator terminal uses the isolated Python executable at
`.local/native-runtime-env/Scripts/python.exe` and the existing `.local/app-v1`
data. That environment points to the backend worktree while the UI chat owns
the primary frontend checkout.

Start the local profile with `-LogLevel DEBUG` to include progress and HTTP timing.
For manually started services, set `MODEL_ANATOMY_LOG_LEVEL=DEBUG` in **both**
API and worker environments before startup. INFO is the regular default; DEBUG
does not enable remote model code, weight loading or extra public endpoints.

Application events are JSON lines on stderr. The launcher saves them in
`api-error.log` and `worker-error.log` beneath its log directory. The October 13
isolated-runtime activation uses `.local/app-v1/native-runtime/` instead.

Copy the analysis ID from the result URL and filter both logs in PowerShell:

```powershell
$analysisId = 'paste-analysis-id-here'
Get-Content .local/app-v1/native-runtime/api-error.log, .local/app-v1/native-runtime/worker-error.log |
  Select-String -SimpleMatch $analysisId
```

Follow the live worker log with:

```powershell
Get-Content .local/app-v1/native-runtime/worker-error.log -Tail 40 -Wait
```

`analysis_submitted` links the API request ID to the analysis ID. The worker logs
`analysis_started`, `analysis_resolved`, DEBUG `analysis_progress`,
`analysis_finished` (including cache reuse), and `analysis_outcome`. Failed jobs
have `analysis_failed` with stage, code, exception type and bounded stack-frame
locations; parent-enforced timeout/crash events identify interrupted children.
Graph-free outcomes retain their inspection reason code, not an invented graph.
Use these events to distinguish repository access, metadata parsing, native
construction, reconstruction, storage, resource limits and unresolved support.

No request bodies, question text, cookies, credentials, raw model URLs, exception
messages, source lines or local variables are serialized by these events. Keep
logs private: model IDs and activity timestamps still describe user activity.
Third-party server/library notices can also appear in stderr; they are not this
JSON event schema. Hosted retention/collection and alerting remain deployment
work, not a new public logs API.

## Expected Facts

These counts are verified at the pinned revisions in the committed engine tests.
A repository's current revision can change; compare the revision displayed by
your result before comparing counts.

| Model | Unique parameters | Layers | Hidden size |
| --- | ---: | ---: | ---: |
| `openai-community/gpt2` | 124,439,808 | 12 | 768 |
| `HuggingFaceTB/SmolLM2-135M` | 134,515,008 | 30 | 576 |
| `Qwen/Qwen3-4B-Thinking-2507` | 4,022,468,096 | 36 | 2,560 |
| `google/vit-base-patch16-224` | 86,567,656 | 12 | 768 |

Fresh engine `2026.10.12.2` checks require all four recorded coverage checks to
reach 100% and no extraction warnings for these canonical pinned cases. ViT
shows pixel input, 196 patches plus a CLS token, and a 1,000-class output; it
does not show an irrelevant Vocabulary field. Text reports show vocabulary
when text input facts establish applicability. Missing relevant facts stay Unknown.

Completed means the scoped static report is complete, not that the model was
run. The app reads configuration, native meta structure, source and tensor
headers. It does not download weights, execute forward, verify predictions or
execute remote model code. Analysis scope remains visible for that reason.
Other models can correctly return Partial or Unsupported with an explanation.

## Service and Remaining Work

[API docs](http://127.0.0.1:8001/docs) expose real job, graph, component, export and
question operations. [Health](http://127.0.0.1:8001/health) checks database, storage
and worker; all three must be true to accept inspections. `ai_available: false`
is expected without Nebius credentials. It must not disable graph inspection.

Start commands are in the [README](../README.md). API and worker must use the
same data directory. The currently launched local workspace uses `.local/app-v1`;
the documented default for a fresh checkout is `.local/model-anatomy`.
Neither directory or its visitor data belongs in Git.

Hosted release still requires cloud account/credit setup, Supabase PostgreSQL
and private storage, real Nebius Nemotron inference, Linux container checks,
HTTPS deployment and external load/recovery checks. No credentials, VM or paid
resources are created by the local application. Model Autopsy, comparison and
additional model families are not part of this checkpoint.
