# Deploy Model Anatomy to your own Vercel account

Start here after pulling this repository. This guide uses your own Vercel
account, project, domain, and credentials. No previous operator's account,
Sandbox, local state file, or private configuration is needed.

The complete deployment uses:

- A regular Vercel Production deployment for the React frontend and the small
  Python `/api/demo` control Function.
- One persistent Vercel Sandbox for FastAPI, the Python worker, SQLite, saved
  reports, and Caddy. The API and worker inspect real Hugging Face artifacts.
- GitHub Actions for manual deployment and operator recovery. Your computer
  does not have to remain running. No separate Linux host or Vercel CLI login
  is required.

This is the agreed Hobby demo profile: SQLite/local storage, no Supabase, and
no AI explanation provider. A new deployment starts with empty demo data.
Existing Supabase data, local reports, and previous operators' data are not
imported. This guide does not configure a durable production database.

## 1. Prepare the GitHub repository

1. Pull the full deployment change. The deployment scripts alone are not enough:
   include the frontend controls, Python Function, and backend admission guard.
2. Merge the reviewed implementation from `feat/deploy` into default branch
   `main` through a PR. GitHub only shows **Run workflow** when the workflow
   exists on the default branch. Do not deploy an older application branch
   that lacks the controls and admission guard.
3. In **GitHub > Repository > Actions**, enable Actions if prompted. You need
   access to run workflows and manage the repository's Actions Secrets/Variables.
   For a fork, configure these in the fork; upstream Secrets are not inherited.
4. Keep both `.github/workflows/deploy-demo.yml` and `start-demo.yml`.
   There is no push-triggered deployment or scheduled recovery workflow.

The same repository supports one deployment destination at a time. Replacing its
three Vercel Secrets and domain Variable switches future workflow runs to the
new account. Follow the [account handover checklist](GITHUB_ACTIONS.md#switch-the-deployment-to-another-account)
when replacing an existing operator.

## 2. Create your Vercel Project in the browser

Vercel's **Add New > Project** page imports a repository or deploys a template;
it is not an empty-project form. Create a temporary frontend deployment first:

1. Sign into your own Vercel account and select your own Hobby team/workspace.
2. Choose **Add New > Project** and import this repository if you own it and it
   is visible. For this repository, set:

   | Setting | Value for this repository's temporary frontend |
   | --- | --- |
   | Project Name | A name of your choice |
   | Root Directory | `frontend` |
   | Framework Preset | `Vite` |
   | Install Command | `npm ci` |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |
   | Development Command | Automatic; Override off |

   Leave `VITE_DEMO_CONTROLS` unset for this temporary build. No Supabase or
   Sandbox secrets are needed at this stage. If the repository is missing from
   import, use a Vite template from Vercel's template selector instead; allow its
   own build defaults. That template is only a placeholder, not the app source.
3. Deploy and wait for a **Ready / Production** deployment. This temporary
   frontend is not the complete working app. Its backend is not connected yet.
4. Open **Project > Settings > Git > Connected Git Repository > Disconnect**.
   This is required, including when a template created a different Git repo.
   Afterward Actions explicitly uploads source files; pushes must not create
   competing Vercel deployments.
5. Open **Project > Settings > Domains**. Keep the assigned free `vercel.app`
   domain, or edit/add an available `your-demo.vercel.app` name and assign it to
   **Production**, without a redirect or Git branch assignment. Record the full
   origin, such as `https://your-demo.vercel.app`.
6. Leave **CDN > Routing Rules** empty for a new project. Do not create an
   all-path Sandbox rewrite. The workflow generates the API-only rewrites.
7. Check **Settings > Deployment Protection**. The chosen Production domain
   must be accessible to signed-out judges. Verify it in an incognito window;
   a Vercel login screen means its protection setting still blocks visitors.

Do not reuse a previous operator's domain: it remains assigned to their project
unless separately transferred. A newly created, unlinked empty project is also
supported by the controller; the browser route above supplies a placeholder so
no local project-creation command is necessary.

## 3. Store your account settings privately

Find these values in your own Vercel dashboard:

| Setting | Where to find the value |
| --- | --- |
| Access Token | **Personal Account Settings > Tokens > Create**. Scope it to the team owning this project and keep it valid for the demo period. |
| Project ID | **Project > Settings > General > Project ID**; starts with `prj_`. |
| Team ID | Switch to the owning team/workspace, then **Team Settings > General > Team ID**; starts with `team_`, including Hobby workspaces. |
| Public origin | **Project > Settings > Domains**; copy your chosen Production origin with `https://`, without a path. |

In **GitHub > Repository > Settings > Secrets and variables > Actions**, add:

| Tab | Exact name | Value |
| --- | --- | --- |
| **Secrets > New repository secret** | `VERCEL_TOKEN` | Your Vercel Access Token |
| **Secrets > New repository secret** | `VERCEL_PROJECT_ID` | Your Project ID |
| **Secrets > New repository secret** | `VERCEL_TEAM_ID` | Your owning Team ID |
| **Variables > New repository variable** | `DEMO_PUBLIC_ORIGIN` | Your Production origin, for example `https://your-demo.vercel.app` |

The domain is public, so it is a Variable. The three account credentials are
Secrets, never committed JSON, README values, workflow inputs, or `VITE_`
variables. If you cannot access these settings, ask the repository owner to add
them. GitHub supplies its own checkout token; no extra GitHub Token is needed.
Different GitHub and Vercel login emails are supported by explicit Vercel Token
authentication without CLI login or Git author metadata.

For optional local cloud commands, copy [.env.example](.env.example) to
`deploy/.env`, set your own four values, and keep it private. `deploy/.env` is
Git-ignored; CI ignores it and reads injected Secrets/Variables instead. A fresh
checkout must never require someone else's private env file. No cloud credentials
belong in `frontend/.env` or `backend/.env`.

## 4. Run the first complete deployment

1. Open **GitHub > Actions > Deploy Demo > Run workflow**.
2. Set **Use workflow from** to `main` for the reviewed deployment implementation.
3. Select **Application source branch**: `feat/deploy` or `main`, whichever
   contains the complete current application. This selects app code separately
   from the workflow implementation.
4. Check **First setup only - replace this project's placeholder frontend**
   (`initialize_project`). Use it for this new project's first deployment only;
   leave it unchecked on later updates.
5. Run and wait for green. The workflow creates a fresh persistent Sandbox,
   uploads committed source, installs locked dependencies, builds the frontend,
   starts FastAPI/worker/Caddy, and checks health and source identity.
6. It then publishes the matching ordinary Production frontend, visitor controls,
   server-only Function credentials, and API-only rewrites. Read **Summary** for
   the selected branch, commit, Sandbox, and public URL.
7. Open your `DEMO_PUBLIC_ORIGIN` in an incognito window. The **Interactive demo**
   strip must appear at the top, before the app header, with **Start Demo** and
   **Stop Demo**. The workflow leaves the prepared Sandbox running.

You do not need scripts 01-04, a pre-existing Sandbox, a legacy CDN rule, or a
Linux terminal for this browser setup. Initialization authorizes replacing the
placeholder in the selected project; check your three Secrets and domain before
running it. It does not transfer an old account's app or delete its resources.

The workflow creates these **sensitive, Production-only** Vercel environment
settings automatically: `DEMO_VERCEL_TOKEN`, `DEMO_VERCEL_PROJECT_ID`, and
`DEMO_VERCEL_TEAM_ID`. Do not enter them separately in GitHub. The Function uses
them server-side; they never enter the browser bundle. The controls build uses
`VITE_DEMO_CONTROLS=true npm run build`. Development Command stays automatic.

## 5. Verify the complete app

1. Confirm `/api/demo` returns a JSON status through your public domain.
2. When ready, open `/health`: database, storage, and worker checks should pass,
   `analysis_available` should be `true`, and `storage_backend` should be `local`.
3. Open `/app` and submit `HuggingFaceTB/SmolLM2-135M`. Wait for an actual report
   and inspect its graph/tensors. Unsupported models can return partial artifact
   inspection or explicit failures; deployment does not broaden model support.
4. While a job is queued/running, confirm Stop is disabled or refused. The server
   also checks other visitors' jobs, independently of the displayed status.
5. After completion, click Stop, then reload. The frontend and Start button must
   remain available while Sandbox is stopped. Click Start, wait for readiness,
   and reopen the saved report. Measure the real resume delay here.
6. After natural timeout, confirm that status polling leaves Sandbox stopped
   until someone explicitly clicks Start.

No authenticated cloud acceptance or resume latency is claimed by offline tests.
If any step fails, use the [operations and troubleshooting guide](GITHUB_ACTIONS.md).

## 6. Understand manual and automatic behavior

| Event or control | Result |
| --- | --- |
| Push a new commit | No demo deployment. Use Deploy Demo to publish it. The separate quality workflow may run tests. |
| Actions **Deploy Demo** | Deploys the selected branch's latest checked-out commit into a new empty Sandbox; checks and publishes it, then stops the previously published Sandbox in the same project if one exists. |
| Actions **Start Demo** | Resumes the last successfully published Sandbox, preserving code/data. Does not fetch commits or create a replacement. Use after at least one successful Deploy Demo. |
| Visitor opens/reloads a page | Frontend loads and polls status only; no Sandbox wake-up. |
| Visitor **Start Demo** | Explicitly resumes the shared published Sandbox and starts its services; no new code or new Sandbox. |
| Visitor **Stop Demo** | Stops the shared session only if no visitor has queued/running analysis; saved data remains in its persistent filesystem/snapshot. |
| No visitor clicks Stop | Hobby session expires naturally after at most 45 minutes. New visitors do not reset its clock. |
| Sandbox stops or times out | No scheduled restart, no cron, and no automatic start from polling. A visitor or operator must Start. |
| Public-start deadline passes | New visitor Start requests are refused after December 16, 2026 in New York; status/Stop remain available. No midnight forced stop. Operator Actions Start remains available. |

Everyone shares one Sandbox. Natural timeout may interrupt analysis even though
manual Stop refuses busy sessions. Hobby quotas still apply: at 8 GB a full
45-minute session consumes 6 GB-hours; 420 GB-hours cover at most 52.5 cumulative
runtime hours before setup/other usage. Restarting does not refill quota. This
cannot guarantee continuous December 1-15 availability. Monitor your own Usage
cycle, Active CPU, memory, snapshots, transfer, and Actions allowances.

For updates, branch selection, token rotation, migration, failures, and account
handover, continue with [GITHUB_ACTIONS.md](GITHUB_ACTIONS.md).
[VERCEL_URL.md](VERCEL_URL.md) explains the domain and distinguishes legacy CDN
migration from a fresh deployment.

## Choose how to run the app locally

| Mode | Frontend | API and worker | Start Demo / Stop Demo |
| --- | --- | --- | --- |
| Normal localhost | Local Vite server | Local Python processes using the existing backend profile | Hidden with `VITE_DEMO_CONTROLS=false`. |
| Local UI inspection | Local Vite server | Local Python processes; no Sandbox controller | Visible with `VITE_DEMO_CONTROLS=true`, but status fails and controls are disabled. |
| Interactive cloud demo | Vercel Production frontend and Function | Shared Vercel Sandbox | Functional after Deploy Demo. |

### Run the app on localhost

Use three Bash terminals from the repository root. Install Python 3.11+, `uv`,
and Node.js as described in the [application prerequisites](../README.md#run-the-application).
Keep an established database/storage profile; these commands use the existing
`backend/.env` and process environment, not `deploy/.env`.

Install dependencies once if needed:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" UV_PYTHON_INSTALL_DIR="$PWD/.tmp/python" \
  uv --cache-dir ./.tmp/uv-cache sync --frozen --extra api --extra engine --extra dev
TMPDIR="$PWD/.tmp" npm --cache ./.tmp/npm-cache ci --prefix frontend
```

Terminal 1 - API:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-sync python -B -m model_anatomy.api
```

Terminal 2 - worker:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-sync python -B -m model_anatomy.service.worker
```

Terminal 3 - frontend:

```bash
mkdir -p .tmp
cd frontend
TMPDIR="$PWD/../.tmp" VITE_DEMO_CONTROLS=false npm run dev
```

Keep all three running and open <http://127.0.0.1:5173>. Check local API
<http://127.0.0.1:8001/health> after the worker starts. Custom ports require
matching `API_PROXY_TARGET` and backend CORS configuration. Stop each local
process with `Ctrl+C` in its terminal.

To inspect the strip visually, restart only the frontend from `frontend/`:

```bash
TMPDIR="$PWD/../.tmp" VITE_DEMO_CONTROLS=true npm run dev
```

The strip appears only when the effective flag is exactly `true`. An explicit
shell value overrides frontend env files. Restart Vite after changes; built
frontends need another build/deployment. Plain Vite and `npm run preview` do not
execute the Python `/api/demo` Function. The local `/api` proxy points to FastAPI,
which has no Sandbox-control endpoint. Expect a status error and disabled
buttons; no local control bridge is implemented. Mocked Playwright responses
are not a live controller. Use the deployed public origin for real Start/Stop
recordings; no local processes are needed for that.

## Optional local deployment helpers

These files are reusable implementation/manual tools, not prerequisites for the
browser workflow above. Run commands in your local terminal, not a Sandbox shell.
Keep `support/` next to the numbered scripts; Step 3 has no source dependency on
files in the local `.tmp/` folder.

| File | Purpose |
| --- | --- |
| `01_vercel_create_project.py` | Optional API-based creation of an unlinked Project. Reads private Team/Token settings; requires `--name`; writes IDs to ignored `deploy/.env`, without printing them. Refuses to replace an existing configured Project. |
| `02_prepare_sandbox.py` | Exports committed Git source, prepares a new Sandbox, and saves ignored preparation state. Does not start services. |
| `03_start_sandbox_demo.py` | Starts/resumes the prepared Sandbox from ignored state; optional `--public-origin`. Does not publish the interactive Production frontend. |
| `04_create_production_entrypoint.py` | Earlier one-time static frontend bootstrap from committed `feat/deploy`; requires Step 2 state. Does not provide working visitor controls. |
| `05_github_demo.py` | Deploy/Start controller used by Actions; interactive publication and recovery follow Production metadata. |
| `github_demo.json` | Shared protocol/legacy routing-rule name only. No account IDs, Token, or operator's domain. |
| `.env.example` | Placeholder template for the ignored private local profile. |
| `support/` | Credential loading, remote runtime, safe stop, routing migration, and interactive rollout implementation. |

For a Sandbox already prepared with Step 2, optional local Step 3 is:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-project \
  --with vercel-sandbox==0.10.0 \
  python -B deploy/03_start_sandbox_demo.py \
  --public-origin https://your-demo.vercel.app
```

Use your actual origin. Credentials come from private `deploy/.env` or injected
environment; a local Token prompt is a fallback. Preserve any existing data
profile. Generated state JSON omits Project/Team IDs and is Git-ignored. Data,
logs, and caches remain under the Sandbox app's `.tmp/vercel-demo/`; local
scratch files and caches belong under repository `.tmp/`. Optional helpers do
not make a raw `sb-....vercel.run` frontend into the external control entry point.

References: [Vercel project creation](https://vercel.com/docs/projects/managing-projects),
[Project ID](https://vercel.com/docs/projects/project-settings/general-settings),
[Access Tokens](https://vercel.com/kb/guide/how-do-i-use-a-vercel-api-access-token),
[manual Actions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow),
[Secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets),
[Variables](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-variables).
