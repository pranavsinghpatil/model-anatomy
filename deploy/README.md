# Vercel Sandbox demo deployment

This workflow runs the frontend, FastAPI API, and Python worker inside Vercel
Sandbox. It currently uses SQLite and local file storage, without Supabase or
AI explanations. Authentication uses an explicitly entered Vercel Access Token;
Vercel CLI login is not required.

```text
deploy/
|-- 01_vercel_create_project.py       # Step 1: create a Vercel Project
|-- 02_prepare_sandbox.py             # Step 2: upload committed source and prepare the environment
|-- 03_start_sandbox_demo.py          # Step 3: start or resume the demo
|-- 04_create_production_entrypoint.py # Step 4: upload the committed frontend for a production URL
`-- support/
    `-- sandbox_demo_runtime.py       # Remote process supervisor uploaded by Step 3
```

Keep `support/` alongside Step 3. Do not run the supervisor separately.
Step 3 no longer depends on source files in the local `.tmp/` directory.

## Steps 1 and 2

Continue using the existing `01` and `02` scripts. Step 2 exports the
**committed source** from the local `feat/deploy` Git branch, without a GitHub
Token. It generates `deploy/sandbox-state.json`, which Step 3 uses to locate
the same Sandbox.

Do not repeat Steps 1 and 2 for a Sandbox that has already been prepared.

## Step 3: start or resume

Run from the repository root:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-project \
  --with vercel-sandbox==0.10.0 \
  python -B deploy/03_start_sandbox_demo.py
```

Enter your Vercel Access Token locally when prompted. Do not share it.
The script reads `sandbox-state.json` from its own directory and writes the
current URL and verification results to `demo-state.json` in the same directory.
Both generated files are excluded by this directory's `.gitignore`.
Use `--state PATH` or `--output PATH` to specify different locations.

When using a regular Vercel domain as a proxy entry point, add
`--public-origin https://your-demo.vercel.app`, replacing the example with your
actual domain. This allows the additional origin; it does not create a domain
or routing rule. If a running demo has a different origin configuration, the
script preserves it and asks you to stop the Sandbox in the dashboard before
running Step 3 with the new option. Include the same option on subsequent resumes.

The script uses the Sandbox created by Step 2 rather than creating another one.
It checks the existing runtime and reuses only the same SQLite demo data
directory. It does not replace other data profiles or an existing Supabase runtime.
After readiness checks pass, it prints `Demo URL:`. Remote services continue
running after the script exits. Readiness checks do not submit model analyses.

Check paths and syntax without a Token or Vercel connection:

```bash
python3 -B deploy/03_start_sandbox_demo.py --dry-run
```

The script sets a 45-minute session limit. After a session stops, run Step 3
again to resume the environment and restart services. Use the URL printed by
that invocation.
Runtime data, logs, temporary files, and caches remain under
`<app_dir>/.tmp/vercel-demo/` inside the Sandbox. The local `uv` cache remains
under the repository's `.tmp/` directory. These are generated files, not
deployment source files that need to be moved manually.

## Step 4: create a Production entry point

If the existing Project displays `No Production Deployment`, create its first
Production deployment from local committed files:

```bash
python3 -B deploy/04_create_production_entrypoint.py --dry-run
python3 -B deploy/04_create_production_entrypoint.py
```

The script reads only the committed `frontend/` tree from the local `feat/deploy`
branch, even if a different branch is checked out or the working tree has changes.
It uses `deploy/sandbox-state.json` for the existing Project and Team IDs.
Enter a Token belonging to the Vercel account that owns that Project.
No GitHub repository connection, GitHub Token, or Vercel CLI login is required.
The request contains file contents and explicit source branch/commit provenance;
it does not contain `gitSource` or Git author metadata.

It configures the regular frontend build as Vite with `frontend` as the Root
Directory, `npm ci` as the Install Command, `npm run build` as the Build Command,
and `dist` as the Output Directory. Development Command remains automatic.
It refuses to replace an existing or currently building Production deployment.
Results are saved to the ignored `deploy/production-state.json`, without the Token.

This step establishes a Production frontend deployment and domain. Complete the
Sandbox Rewrite configuration before treating the new domain as the full app.
Run offline helper tests with `python3 -B -m unittest discover -s deploy/tests -v`.

## A readable URL

Vercel generates the Sandbox's `*.vercel.run` address. The current
[Python SDK](https://vercel.com/docs/sandbox/python-sdk-reference#routes)
provides route lookup and port configuration, without an option to rename the
generated address. Renaming the Project does not turn the Sandbox address into
a matching `*.vercel.app` address.

A regular Vercel Project domain can provide an entry point that forwards requests
to the Sandbox through the CDN.
[Project-Level Routing Rules](https://vercel.com/docs/routing/project-routing-rules)
support Rewrite configuration under `CDN > Routing Rules` in the dashboard.
Save and then Publish the rule. The frontend, API, and worker still run inside
the Sandbox, and the session limit still applies.

Add the new entry point to the API's allowed origins using Step 3's
`--public-origin` option. Otherwise, analysis submissions may return
`403 origin_rejected`.
See [VERCEL_URL.md](VERCEL_URL.md) for the initial free domain setup and
dashboard instructions.
