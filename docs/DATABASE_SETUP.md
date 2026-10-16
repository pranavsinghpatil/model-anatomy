# Database Setup and Report Reuse

## What This Changes

An analysis has two separate lifetimes:

| Stored information | Purpose | Default retention |
| --- | --- | --- |
| Visitor job | Your progress, recent history and private analysis link | 24 hours |
| Inspected report | Reusable real architecture, tensors and evidence | No automatic expiry |

For every new submission the worker checks the requested Hugging Face reference:

1. Resolve the reference to its exact commit SHA using repository metadata.
2. Look up model ID + commit + engine version + report schema + artifact selection.
3. If the report exists, verify its SHA-256 and attach it to your new private job.
   No extraction, model import, weight download or storage upload is repeated.
4. A different commit or processing identity requires a new inspection. Old
   reports remain independent; a saved link never changes to another revision.

Concurrent requests queue behind the single globally locked worker. After the
first succeeds, identical queued requests reuse its persisted report. They still
require admission/quota and a small metadata lookup; this is not a zero-network
or zero-latency path. An upstream outage cannot prove a branch is unchanged.

Partial reports retain their warnings. Fixing extraction requires an engine
version change so improved code does not reuse an older engine's result.
Missing/corrupt storage fails explicitly; restore it from backup rather than
silently running the engine again. Failed jobs are not cached as reports.

## Create Supabase

The code supports Supabase, but no hosted project is provisioned or verified by
this change. There are no credentials to enter into the frontend.

1. Create a Supabase project in your account. Keep its database password private.
2. In **Connect**, get the direct PostgreSQL connection string. If the backend
   network cannot reach IPv6, use **Session pooler** instead. Do not use the
   transaction pooler: the worker uses a session-level advisory lock.
3. Create a **private** Storage bucket named `model-anatomy-private`. Never make
   it public or add anonymous read/list/write policies.
4. Get the project HTTPS URL and server-only service-role key. Only the backend
   receives these. Never put them in `VITE_*` variables, chat, logs or Git.
5. This is a backend-only database. In the Dashboard's **Data API** integration,
   turn **Enable Data API** off before serving real jobs, or have an operator
   establish and verify equivalent
   restrictive database access/RLS. The FastAPI visitor cookie is not a Supabase
   login; public database access would bypass its ownership checks.
6. Configure backups for PostgreSQL **and the private report objects**. Verify an
   actual restore. Monitor stored object count/bytes as new revisions accumulate.

Connection methods and credential locations are described in
[Supabase PostgreSQL connections](https://supabase.com/docs/guides/database/connecting-to-postgres).
See [private bucket behavior](https://supabase.com/docs/guides/storage/buckets/fundamentals)
and [Storage access control](https://supabase.com/docs/guides/storage/security/access-control).
Disabling the Data API for direct database connections follows
[Supabase's backend-only security guidance](https://supabase.com/docs/guides/api/securing-your-api).

## Configure the Backend

Use the placeholders in `infra/.env.example` to create the ignored `infra/.env`
on the deployment machine. Its database/storage fields are:

```dotenv
MODEL_ANATOMY_DATABASE_URL=postgresql+psycopg://YOUR_CONNECTION_STRING?sslmode=require
MODEL_ANATOMY_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
MODEL_ANATOMY_SUPABASE_KEY=YOUR_SERVER_ONLY_SERVICE_ROLE_KEY
MODEL_ANATOMY_SUPABASE_BUCKET=model-anatomy-private
MODEL_ANATOMY_CACHE_TTL_SECONDS=0
```

Use the actual host/username from Connect, percent-encode reserved characters in
the password and append TLS parameters correctly if the URL already has a query.
These placeholders are not usable credentials. Prefer certificate-verified TLS
where the provider CA is available. Restrict the environment file's permissions.

`0` means keep reusable revision reports. A positive cache TTL explicitly opts
into deletion by age and may cause reinspection. Switching to a positive TTL can
retire old entries immediately. API and worker must use the same settings.
Keep the remaining production settings from the deployment template, including
private storage, explicit HTTPS origins and the real explanation credentials.

On the deployment checkout, follow [Deployment](DEPLOYMENT.md) to build the
locked images, run `alembic upgrade head`, run preflight, and start the API/worker.
The compose stack loads `infra/.env`; Python launched directly does **not**
automatically load that file. Direct launches need the same environment exported
into their processes. Do not paste credential-bearing commands into shared logs.

Changing a database URL does not transfer the existing SQLite database or local
objects. Before switching a populated installation, stop writers, back up the
database and objects together, and plan a verified import. Do not delete local
data or claim it is migrated. Fresh hosted installations start with an empty cache.

## Verify It Yourself

1. Submit a real accessible model. Record its `revision`, `engine_version` and
   analysis ID. The first uncached result should show `cached: false`.
2. Submit the same reference again. If its commit has not changed, expect
   `cached: true`, the same revision and the same downloaded report bytes.
3. In worker logs, find `analysis_cache_hit` for the second ID; it should have no
   extraction progress events. The first miss has `analysis_cache_miss`.
4. Restart API/worker with the same database and bucket. Repeat step 2: it must
   still be cached. A different visitor must not access either of your job URLs.
5. Try another **existing** immutable revision or an artifact selection. A new
   identity should miss once, then hit on repetition. Never invent a commit SHA.
6. After visitor links expire, submit the same unchanged reference again. The
   old URL remains expired, but the new owned job should reuse the report.

Automated tests cover restart/history expiry beyond 90 days, preserved legacy
deadlines, concurrent submissions, changed commits, identity isolation, skipped
uploads, integrity errors and upstream outages. Hosted PostgreSQL/private-bucket
restart, privacy and restore verification still require your project credentials.
