# CPU VM Deployment

This stack is a deployment candidate, not a verified live deployment. Docker,
the Nebius CLI, cloud credentials, and a public domain were unavailable during
implementation. Complete the checks below before publishing a demo URL.

## Start Here Without Accounts

First complete the [local testing guide](LOCAL_TESTING.md); no cloud account or
API key is needed for model inspection. Then create the cloud dependencies:

1. Join the event and obtain Nebius Token Factory access through the
   [Builder Program](https://dev.nebius.com/). Verify credits and expiry in your
   account before choosing an exact NVIDIA Nemotron model. The
   [event FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs) says
   promotional credits are for Token Factory, not a separate VM grant.
2. Create a Supabase project and its private storage bucket as described below.
   The database URL and server-only key belong in the private `infra/.env`, not
   in chat, Git or frontend settings.
3. Choose an approved hosting budget and HTTPS domain. The CPU VM is the current
   deployment target, but its credit/billing coverage must be checked separately.
4. Run the build, migration, preflight and end-to-end checks below. A configured
   account or successful model-list request alone does not complete this gate.

## Topology and Integration Contract

Internet -> Caddy HTTPS -> Vite static assets or FastAPI -> Supabase PostgreSQL
and private Storage. A separate Python worker consumes durable jobs. NVIDIA
Nemotron runs remotely on Nebius Token Factory; this VM needs no GPU.

| Target | Runtime | Production dependencies |
| --- | --- | --- |
| `api` | `uvicorn model_anatomy.api.app:create_app --factory --host 0.0.0.0 --port 8000` | `uv sync --frozen --extra api --no-dev --no-default-groups --no-editable` |
| `worker` | `python -m model_anatomy.service.worker` | Same sync with `--extra engine`; CPU Torch from `uv.lock` |
| `frontend` | Caddy serving `frontend/dist` | Node build stage runs `npm ci` and `npm run build`; Node/dev packages are absent from runtime |

Python and frontend dependencies are locked. Root
`alembic.ini` resolves root `migrations/` through `%(here)s`; runtime images copy
both alongside `backend/` and contain no prepared reports or snapshots. Compose
runs `alembic -c alembic.ini upgrade head` from `/app`, and preflight checks that
same configuration. Migrations use `MODEL_ANATOMY_DATABASE_URL`.

Environment contract: `MODEL_ANATOMY_DATABASE_URL`, `DATA_DIR`, `ENVIRONMENT`,
`ALLOWED_HOSTS`, `CORS_ORIGINS`, `SUPABASE_URL`, `SUPABASE_KEY`,
`SUPABASE_BUCKET`, `NEBIUS_API_KEY`, `NEBIUS_MODEL` (each has the prefix).
The backend accepts the canonical prefixed `NEBIUS_MODEL` and `SUPABASE_KEY`.
Hosts/origins are JSON arrays; Compose supplies explicit production values.
Production sets Secure cookies through the environment mode, not a separate
cookie override. All browser requests use the same HTTPS origin.

The backend must enforce durable anonymous per-guest and global admission limits,
question/token budgets, queue capacity, ownership, timeouts, and retry bounds.
Approved defaults in `infra/.env.example` are **10 waiting jobs**, **2 jobs/hour
and 10 questions/hour per guest**, and **20 jobs/day and 100 questions/day
globally**. Queue capacity counts waiting jobs only, excluding the running job.
Global environment fields are `MODEL_ANATOMY_GLOBAL_JOBS_PER_DAY` and
`MODEL_ANATOMY_GLOBAL_QUESTIONS_PER_DAY`; the backend must enforce daily windows
before public deployment. Caddy's 64 KB request cap is only
a transport bound. Test quota exhaustion and foreign guest access before exposing
the service; do not count the proxy or a cookie alone as quota enforcement.

Only Caddy publishes ports. API/worker have no host ports. Root filesystems are
read-only, every runtime uses UID/GID 10001, capabilities are dropped, and `/tmp`
is bounded tmpfs. `/data` is the shared writable app volume; certificates live in
separate persistent Caddy volumes. The worker is capped at 4 GiB total memory,
no extra swap, and two CPUs. Its 512 MiB tmpfs counts against that memory cap.
Use one worker until concurrency and recovery are measured. Logs rotate locally.

## Provisioning and Budget

Use the Nebius console to select an available **CPU-only**, x86-64 Linux VM:
start with 4 vCPUs, 8 GiB RAM, and a 40 GiB persistent boot disk. Build on a
separate machine or increase build-time RAM if needed; runtime limits do not
limit Docker builds. Region/platform capacity and prices must be checked live.
No resources are provisioned by these files.

Before creating the VM, record the current VM, disk, public IPv4, bandwidth,
Supabase plan, backups, domain, and Token Factory costs. Estimate the budget as
hourly VM cost times uptime hours plus persistent services and bounded token
usage; set billing alerts and an operator-owned spending limit. Verify credit
expiry and service availability through **December 15, 2026**, not just submission
day. Token Factory promotional credits are not a VM funding guarantee; verify
grant coverage in the [event FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs).

Install Docker Engine and Compose v2 using the
[official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
Use an SSH key, restrict TCP 22 to operator addresses, and permit public TCP
80/443 (UDP 443 optional for HTTP/3). Do not open 8000 or database ports.
Point the domain's A record at the VM; publish AAAA only if IPv6 routing works.
Caddy maps external 80/443 to unprivileged 8080/8443 and obtains certificates
using [automatic HTTPS](https://caddyserver.com/docs/automatic-https).

## Private Services

Create a Supabase project and a **private** `model-anatomy-private` bucket in the
console. Obtain the server-only service-role key, project HTTPS URL, and direct
or session-pooler PostgreSQL connection URL. Use the session pooler when the VM
cannot reach direct IPv6; avoid the transaction pooler for migrations. Include
`sslmode=require` at minimum; prefer `verify-full` with the provider's trusted CA.
See [Supabase connections](https://supabase.com/docs/guides/database/connecting-to-postgres)
and [private buckets](https://supabase.com/docs/guides/storage/buckets/fundamentals).
Never expose the service-role key in Vite environment variables or browser code.
Do not grant anonymous public bucket access. The backend mediates object access
after checking guest ownership. Configure provider backups and retention.

Obtain a Nebius Token Factory key separately. Select an exact, available
`nvidia/...` Nemotron model ID from the model catalog. The preflight below makes
only `GET /v1/models`; it neither performs inference nor proves meaningful model
usage. See the [Token Factory API](https://api.tokenfactory.nebius.com/docs).

## Build and Start

On the Linux VM, obtain the reviewed integrated release checkout containing the
Vite npm lock, migrations, and worker. Run from the repo root:

```bash
cp infra/.env.example infra/.env
chmod 600 infra/.env
nano infra/.env
docker compose version
docker compose --env-file infra/.env config --quiet
docker compose --env-file infra/.env build api worker frontend
docker compose --env-file infra/.env run --rm --no-deps api python infra/preflight.py --skip-schema
docker compose --env-file infra/.env up -d --wait --wait-timeout 180
docker compose --env-file infra/.env run --rm --no-deps api python infra/preflight.py
docker compose --env-file infra/.env ps
```

Stop on any nonzero exit. Do not use `--skip-schema` for the post-migration check.
The one-shot Alembic service must exit successfully before API/worker start;
Caddy waits for API readiness. `--env-file` is needed for Compose interpolation
as well as the service `env_file`; use it on every command. `config --quiet`
validates without printing interpolated credential URLs. Never paste full
`docker compose config`, environment dumps, or provider exception bodies into
submission evidence. These commands create/run resources only when the operator
executes them; the preflight itself does not write database/storage state.

For a static runtime-image contract check and deployment preflight regression
tests:

```bash
docker compose --env-file infra/.env run --rm --no-deps api python infra/preflight.py --static
python -m unittest discover -s infra/tests -v
```

Before release, record exact base-image digests and resulting image IDs for all
targets. Tags in the Dockerfile are versioned but mutable. Retain the previous
release images and a database backup for rollback.

## Verify and Keep Available

Replace the example domain in these commands with the configured domain:

```bash
curl --fail --silent --show-error https://model-anatomy.example.com/health
docker compose --env-file infra/.env logs --tail 100 migrate api worker frontend
docker stats --no-stream
```

Open the HTTPS URL from another network. Check that HTTP redirects to HTTPS,
the Vite application renders, `/api/*` keeps its prefix, unknown API routes return
API errors, and no provider credentials appear in browser traffic. Submit the
primary pinned model, inspect tensor evidence, verify a new visitor starts empty,
and check cancellation/restart recovery. An operator-approved Nemotron
explanation is a separate, potentially billed acceptance test; save sanitized
evidence and verify its citations resolve to existing IR facts. Preflight never
runs it.

API health checks require `/health` to return 200. Caddy's local probe verifies
only its listener. Worker health checks verify the worker process and a fresh
global lease using a read-only database query; they do not prove job progress.
Monitor stale jobs/leases and worker progress externally. Compose marks unhealthy
containers but **does not automatically restart them**; `unless-stopped` restarts
exited processes and after host reboot. Investigate an unhealthy service before
running `docker compose --env-file infra/.env restart api worker frontend`.

Enable Docker at boot (`sudo systemctl enable --now docker`). Set an external
HTTPS readiness monitor and a billing alert; test alerts and certificate renewal.
Check the actual model catalog, Supabase project/bucket availability, credits,
disk usage, and domain expiry before submission and during December judging.
Arrange free judge access that can complete the demo within normal quotas; if
judges need expanded access, implement a reviewed operator-managed
allowance. This does not authorize disabling global spend caps.

Before upgrades, back up Supabase and record image IDs. Stop writes with
`docker compose --env-file infra/.env stop api worker frontend`, build the new
release, run migrations/preflight, then `up -d --wait`. Roll back only to an image
compatible with the migrated schema; do not blindly downgrade migrations. Restore
from a tested backup when necessary. Preserve Caddy and app volumes; `down -v`
deletes persistent state and must not be used for routine maintenance.

## Release Evidence

Record release/image IDs, static runtime checks, clean image builds, CPU/no-Torch
build assertions, migration head, sanitized preflight results, HTTPS health responses,
end-to-end demo results, quota/recovery checks, and NVIDIA runtime evidence.
Missing credentials, failed builds, failed migrations or an untested public URL
remain release blockers. Repeat the [recorded benchmark](BENCHMARKS.md) against
the deployed VM and retain sanitized results. Nothing here claims provisioning,
successful cloud inference or a completed hackathon submission.
