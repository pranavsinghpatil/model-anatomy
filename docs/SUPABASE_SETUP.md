# Supabase setup for Model Anatomy

This guide connects a fresh Supabase project to the API and worker. The app uses
PostgreSQL directly and private Supabase Storage; it does not use Supabase Auth,
the Supabase JavaScript client, or PostgREST.

## Provisioned project status (October 20, 2026)

The `model-anatomy` project is created and reports **ACTIVE_HEALTHY** in the
`pranavsinghpatil's Org` organization, Mumbai (`ap-south-1`), project ref
`kkvpwozxbkimwinffclg`. The selected dashboard estimate was $0/month; confirm
current billing and limits in the account. [Open the project](https://supabase.com/dashboard/project/kkvpwozxbkimwinffclg).

The project and database schema are provisioned. The canonical Alembic schema
head is `0001_durable_jobs`; all seven service tables and `alembic_version`
have RLS enabled, with no browser-role policies or grants. The bucket,
credentials, and app connectivity are still pending, so no app connection has
been claimed. Use these links to finish configuration:

- [Storage buckets](https://supabase.com/dashboard/project/kkvpwozxbkimwinffclg/storage/buckets)
- [API keys](https://supabase.com/dashboard/project/kkvpwozxbkimwinffclg/settings/api-keys)
- [Database connection settings](https://supabase.com/dashboard/project/kkvpwozxbkimwinffclg/database/settings)

## Create the project

1. The project is already created as described above. Set or retrieve its
   database password and save it in your password manager. Do not
   put it in chat, source control, or frontend settings.
2. In **Storage**, create `model-anatomy-private` and leave it **private**. Do
   not add public access or anonymous storage policies. The backend checks
   visitor ownership before serving objects.
3. In project settings, copy the project URL and the server-side legacy
   `service_role` key. Keep the key private. The app does not need an anon or
   publishable key.
4. In **Connect**, choose the direct database connection or **Session pooler**
   and copy its URI. Use port 5432 and SSL. Do not use the transaction pooler
   (usually port 6543): the worker holds a PostgreSQL advisory lock for its
   lifetime, which requires a stable connection. If using the pooler, follow
   Supabase's username format shown in its connection details.

See Supabase's [database connection guide](https://supabase.com/docs/guides/database/connecting-to-postgres)
and [private bucket guide](https://supabase.com/docs/guides/storage/buckets/fundamentals).

## Prepare local environment

The ignored `infra/.env` holds local credentials. Copy `infra/.env.example` if
needed. For database connection checks and migrations from Windows, keep
`MODEL_ANATOMY_ENVIRONMENT=local` and set values like these, URL-encoding special
characters in the database password:

```dotenv
MODEL_ANATOMY_DATABASE_URL=postgresql+psycopg://postgres.PROJECT:URL_ENCODED_PASSWORD@DB_HOST:5432/postgres?sslmode=require
MODEL_ANATOMY_SUPABASE_URL=https://PROJECT.supabase.co
MODEL_ANATOMY_SUPABASE_KEY=YOUR_SERVER_ONLY_SERVICE_ROLE_KEY
MODEL_ANATOMY_SUPABASE_BUCKET=model-anatomy-private
MODEL_ANATOMY_STORAGE_BACKEND=supabase
MODEL_ANATOMY_DATA_DIR=.local/supabase
MODEL_ANATOMY_ENVIRONMENT=local
MODEL_ANATOMY_PORT=8001
MODEL_ANATOMY_ALLOWED_HOSTS=["localhost","127.0.0.1","[::1]"]
MODEL_ANATOMY_CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
```

Do not share the file or paste its values into logs or screenshots. Settings do
not load `.env` automatically; prefix each migration or check command with the
repository's dotenv wrapper, as shown below. This local environment profile is
only for read-only checks and migrations against Supabase. Do not run the API or
worker with `ENVIRONMENT=local` and a remote PostgreSQL URL: local mode may call
`create_all`. For app runtime, use the Linux production profile described in
[Deployment](DEPLOYMENT.md), with `ENVIRONMENT=production`, explicit HTTPS hosts
and origins, and the reviewed migration already applied.

## Create the database schema

The app does not use Supabase Data API/PostgREST for these tables. For a fresh
project, disable **Data API** in project API settings before running Alembic.
Alembic creates seven service tables and `alembic_version` without RLS. Then,
before starting the app, run the reviewed hardening SQL file
[`infra/supabase/private-tables.sql`](../infra/supabase/private-tables.sql) in
the Supabase SQL Editor as the project database owner (`postgres`). It enables
RLS on all eight tables and revokes browser-role grants without adding anon or
authenticated policies. The trusted database owner used by the API and worker
continues to access them. Do not expose the tables through Data API or invent
`auth.uid()` policies: the app uses anonymous visitor cookies, not Supabase
Auth. Do not alter Supabase's own Storage tables. The existing project received
the canonical schema and hardening together in one transaction through the
Supabase integration. Its Data API setting was not changed; do not rerun setup. The
database linter's `rls_enabled_no_policy` INFO finding is expected: absent
browser policies deliberately denies browser-role access. See the
[Supabase linter explanation](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy).

For a different fresh project, after disabling Data API, apply the current
migration from the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m dotenv -f infra/.env run -- .venv/Scripts/python.exe -m alembic -c alembic.ini upgrade head
```

This creates seven service tables and `alembic_version`. It only prepares an
empty project; it does not copy any data from a local SQLite database. Keep
backups and retention appropriate to the data you store.

Then run `infra/supabase/private-tables.sql` in the SQL Editor. Preserve the
Alembic schema head `0001_durable_jobs`; do not replace or manually edit
`alembic_version`.

## Check database and bucket

Use this read-only check from the same PowerShell session. It tests only the
database connection and private bucket, without printing URLs or exception text:

```powershell
.venv/Scripts/python.exe -m dotenv -f infra/.env run -- .venv/Scripts/python.exe -c "from infra.preflight import check_database, check_bucket; checks=[('database', lambda: check_database('alembic.ini')), ('private bucket', check_bucket)]; failed=False
for name, check in checks:
 try: check(); print(f'PASS: {name}')
 except Exception: print(f'FAIL: {name} (check private configuration and provider availability)'); failed=True
raise SystemExit(1 if failed else 0)"
```

The current project has completed the database migration and RLS hardening, but
the check will remain incomplete until its credentials are added and the private
bucket is created. `scripts/start-local.ps1` deliberately forces SQLite and
local file storage; it starts the local app and does not use Supabase. The
production host is Linux. Follow [Deployment](DEPLOYMENT.md) for API and worker
startup; full deployment preflight also requires a valid Nebius model and key.

## Troubleshooting

- **Database connection fails:** confirm host, port, pooler username, URL-encoded
  password and `sslmode=require`; check that the project is awake and reachable.
- **Bucket check fails:** confirm project URL, service-role key, exact bucket
  name and that the bucket's public setting is off.
- **Migration fails:** check the sanitized error locally and confirm the
  connection targets the project. Never paste a
  credential-bearing URL or full exception into an issue or chat.
- **API still uses SQLite/local storage:** verify `MODEL_ANATOMY_STORAGE_BACKEND`
  and `MODEL_ANATOMY_DATABASE_URL` are set in `infra/.env`, and that each
  process was launched through the dotenv wrapper.
