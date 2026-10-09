# Service configuration

Run from the repository root with API dependencies installed. API startup and
stored report reads never import Torch or Transformers. The worker needs the
separate engine extra and starts a fresh spawn child for each job.

```powershell
$env:MODEL_ANATOMY_PORT = '8001'
python -m model_anatomy.api
```

Start the worker in a separate terminal/process:

```powershell
python -m model_anatomy.service.worker
```

Local defaults: SQLite and immutable JSON objects in `.local/model-anatomy`;
root `alembic.ini` and `migrations/` match the infrastructure preflight. Local
startup creates missing tables for convenience. Production never creates tables;
run `python -m alembic -c alembic.ini upgrade head` before API/worker startup.
Early local databases receive the two retention fields without deleting jobs;
using a fresh local data directory is also supported during development.

## Production

Required settings:

| Variable | Value |
| --- | --- |
| `MODEL_ANATOMY_ENVIRONMENT` | `production` |
| `DATABASE_URL` or `MODEL_ANATOMY_DATABASE_URL` | PostgreSQL DSN, direct or session pooler, with `sslmode=require` |
| `MODEL_ANATOMY_STORAGE_BACKEND` | `supabase` |
| `MODEL_ANATOMY_SUPABASE_URL` | `https://PROJECT.supabase.co` |
| `MODEL_ANATOMY_SUPABASE_KEY` | Server-only service-role key |
| `MODEL_ANATOMY_SUPABASE_BUCKET` | Existing **private** bucket; default `model-anatomy-private` |
| `MODEL_ANATOMY_ALLOWED_HOSTS` | JSON array of explicit public hosts |
| `MODEL_ANATOMY_CORS_ORIGINS` | JSON array of explicit HTTPS browser origins |

PostgreSQL transaction poolers are incompatible with the lifetime advisory
worker lock. SQLite uses an OS file lock plus transaction mutex and conditional
claims. PostgreSQL uses the mutex, advisory lock, and `FOR UPDATE SKIP LOCKED`.
The worker lease defaults to 30 seconds and renews while the child runs. Leases
fence every result write. Dead workers recover interrupted jobs within bounded
attempts. Linux children receive parent-death termination, a 4 GiB virtual-address
limit (`RLIMIT_AS`), CPU limit, and independent 120-second wall-clock watchdog.
Production workers fail closed on other platforms. Windows local execution
enforces timeouts but does not enforce the memory limit. `RLIMIT_AS` limits virtual
address space, not RSS. The conditional Linux test imports the real engine and
instantiates a tiny GPT-2 on meta under that limit; it is skipped on Windows.
Production readiness has not been demonstrated on this Windows host: Linux CI
must pass that test, and the deployment must also enforce its 4 GiB container RSS
budget. Large import reservations may require a different process memory policy.

Readiness requires initialized database, current migration, private storage and a
live worker heartbeat. Health and owned stored reports remain independent of AI.
There is no prepared-report directory setting, example endpoint or public sample
analysis ID. Every report read requires the submitting visitor's cookie.

## Explanation provider

Set `NEBIUS_API_KEY` and `NEBIUS_MODEL` to an exact available `nvidia/...` Nemotron
model ID from the provider. Also accepted: `MODEL_ANATOMY_NEBIUS_API_KEY` and
`MODEL_ANATOMY_NEBIUS_MODEL` (legacy `MODEL_ANATOMY_NEBIUS_MODEL_ID`). No model ID
or key is guessed. `MODEL_ANATOMY_NEBIUS_BASE_URL` defaults to
`https://api.tokenfactory.nebius.com/v1`. Missing credentials produce explicit
503 responses; graph/inspection/export remain available. Provider mock tests use
`httpx.MockTransport`; no live inference has been verified in this workspace.

Answers must be actual JSON, cite only supplied IDs, and pass exact numeric-value
validation. Context is bounded to 24,000 bytes and retains static-analysis
limitations. These checks cannot prove every prose claim is entailed; answers are
marked AI-generated and never modify graph facts.

## Limits and retention

Defaults: ten **waiting** jobs; two inspections/guest/hour; ten questions/guest/hour;
twenty inspections globally/day; one hundred questions globally/day. Persisted
fixed UTC windows count successful enqueues/provider attempts, not reads.
Override with `MODEL_ANATOMY_QUEUE_CAPACITY`, `MODEL_ANATOMY_GUEST_JOBS_PER_HOUR`,
`MODEL_ANATOMY_GUEST_QUESTIONS_PER_HOUR`, `MODEL_ANATOMY_GLOBAL_JOBS_PER_DAY`, and
`MODEL_ANATOMY_GLOBAL_QUESTIONS_PER_DAY`.

Visitor jobs expire after 24 hours and are deleted by worker cleanup. Hash-only
tombstones retain owner-only 410 responses for at most another 24 hours, then
disappear; other visitors receive 404. Inactive unused guests and old quota
buckets are removed. Cache entries expire seven days after extraction; hits do
not extend retention. Blob deletion runs outside database transactions and uses
a durable deletion marker to fence new references. Upload reservations collect
objects left behind by interrupted completion. No reports are seeded at startup.

Cache identity includes canonical model ID, immutable 40-character revision,
engine version (from `src.version`), IR schema, artifact filename, and explicit
artifact selection. A resolved revision is reused on retry. Cache hits prevent
engine imports/extraction in the worker child.
