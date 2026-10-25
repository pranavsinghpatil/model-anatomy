# Supabase setup for Model Anatomy

This guide connects a fresh Supabase project to the API and worker. The app uses
PostgreSQL directly and private Supabase Storage; it does not use Supabase Auth,
the Supabase JavaScript client, or PostgREST.

## Provisioned project status (October 20, 2026)

The `model-anatomy` project is created and reports **ACTIVE_HEALTHY** in the
`pranavsinghpatil's Org` organization, Mumbai (`ap-south-1`), project ref
`kkvpwozxbkimwinffclg`. The selected dashboard estimate was $0/month; confirm
current billing and limits in the account. [Open the project](https://supabase.com/dashboard/project/kkvpwozxbkimwinffclg).

The project is running the canonical Alembic head `0002_model_reports`.
All eight service tables plus `alembic_version` have RLS enabled and no browser
role access. TLS PostgreSQL connectivity and the private
`model-anatomy-private` bucket (`public=false`) were verified. The API, worker,
and frontend were then tested against Supabase with a real SmolLM2-135M
inspection, durable catalog publication, cached reuse, and visitor isolation.
See [database catalog acceptance evidence](DATABASE_CATALOG.md).

Credentials are saved in the ignored local `infra/.env`. Use these links to
open the project settings:

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
not load `.env` automatically. For migrations and connection checks use the
dotenv wrapper below. For the Windows development app, use the explicit
Supabase launcher from the repository root:

```powershell
./scripts/start-local.ps1 -Supabase -EnvFile ./infra/.env
```

The launcher reads only allowlisted private settings, requires TLS and a
session/direct database connection on port 5432, and verifies database, storage,
and worker readiness. It never migrates or silently falls back to SQLite.
PostgreSQL startup requires the current Alembic head and never calls
`create_all`. The frontend receives no database or service-role credentials.
With no `-Supabase` flag, the launcher selects the separate SQLite development
profile. Linux production deployment still requires the checks in
[Deployment](DEPLOYMENT.md).

## Create the database schema

The app does not use Supabase Data API/PostgREST for these tables. For a fresh
project, disable **Data API** in project API settings before running Alembic.
The initial migration creates private service tables; the registry migration
adds `model_reports` with RLS. Before starting the app, run the reviewed
hardening SQL file
[`infra/supabase/private-tables.sql`](../infra/supabase/private-tables.sql) in
the Supabase SQL Editor as the project database owner (`postgres`). It enables
RLS on all nine tables and revokes browser-role grants without adding anon or
authenticated policies. The trusted database owner used by the API and worker
continues to access them. Do not expose the tables through Data API or invent
`auth.uid()` policies: the app uses anonymous visitor cookies, not Supabase
Auth. Do not alter Supabase's own Storage tables. The existing project already
has the canonical migrations and hardening applied. Its Data API setting was
not changed; do not rerun setup. The
database linter's `rls_enabled_no_policy` INFO finding is expected: the absence
of browser policies deliberately denies browser-role access. See the
[Supabase linter explanation](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy).

For a different fresh project, after disabling Data API, apply the current
migration from the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m dotenv -f infra/.env run -- .venv/Scripts/python.exe -m alembic -c alembic.ini upgrade head
```

This creates eight service tables and `alembic_version`. It only prepares an
empty project; it does not copy any data from a local SQLite database. Keep
backups and retention appropriate to the data you store.

Then run `infra/supabase/private-tables.sql` in the SQL Editor. Preserve the
Alembic schema head `0002_model_reports`; do not replace or manually edit
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

The database and private bucket checks passed for the current project. Runtime
verification also confirmed the app writes real reports to Supabase and reuses
them before reconstruction. See [database catalog](DATABASE_CATALOG.md) for
versioned identity, quotas, retention, and the exact model used.

## Troubleshooting

- **Database connection fails:** confirm host, port, pooler username, URL-encoded
  password and `sslmode=require`; check that the project is awake and reachable.
- **Bucket check fails:** confirm project URL, service-role key, exact bucket
  name and that the bucket's public setting is off.
- **Migration fails:** check the sanitized error locally and confirm the
  connection targets the project. Never paste a
  credential-bearing URL or full exception into an issue or chat.
- **API still uses SQLite/local storage:** verify `MODEL_ANATOMY_STORAGE_BACKEND`
  and `MODEL_ANATOMY_DATABASE_URL` are set in `infra/.env`, and launch with `./scripts/start-local.ps1 -Supabase -EnvFile ./infra/.env`.
