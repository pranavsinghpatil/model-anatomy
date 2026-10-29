# A free vercel.app entry point

Use this guide when the Project currently has only a Sandbox and displays
`No Production Deployment`. Reuse the existing Vercel Project; no Vercel CLI
or purchased domain is required.

Requests follow this route: browser -> regular Vercel Project CDN -> Sandbox.
Once forwarding is configured, the Sandbox still serves the frontend, FastAPI,
worker, and SQLite data. This entry point does not automatically resume a stopped
Sandbox or extend its session duration.

## 1. Prepare a regular Production Deployment without Git integration

The Project can remain under your own Vercel account even when a teammate owns
the GitHub repository. For a personal GitHub repository, collaborators cannot
connect it to a Vercel Project through Git integration. Repository read/write
access alone does not grant that permission. This workflow instead uploads
the local branch's committed frontend through Vercel's file deployment API.

1. In the existing Project's dashboard, open `Settings > Build and Deployment`
   and keep these settings:

   | Setting | Value |
   | --- | --- |
   | Root Directory | `frontend` |
   | Framework Preset | `Vite` |
   | Install Command | `npm ci` |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |
   | Development Command | Automatic; Override off |

2. Leave `Settings > Git` unconnected. No Production Branch tracking setting
   is needed for this file upload workflow; the script explicitly selects
   the local `feat/deploy` branch and the `production` target.
3. From the repository root, check the upload without requesting a Token or
   contacting Vercel:

   ```bash
   python3 -B deploy/04_create_production_entrypoint.py --dry-run
   ```

4. Check the printed Project ID, Team ID, branch, commit, and build settings.
   Then create the deployment:

   ```bash
   python3 -B deploy/04_create_production_entrypoint.py
   ```

5. Enter a Vercel Access Token generated under the account that owns the existing
   Project. The Token is entered locally and is not saved. The helper reads
   `deploy/sandbox-state.json`, checks the selected Project and Team, uploads the
   committed frontend files, and waits for `READY`. It does not replace an
   existing or currently building Production deployment.
6. Open the Project's `Deployments` page and confirm that the deployment is
   `Ready` and `Production`. If a build fails, inspect its build logs. If polling
   times out or the connection is interrupted, check the dashboard and
   `deploy/production-state.json` before retrying.

Your Vercel login email and GitHub email may differ. In this workflow the
Vercel Token, Project ID, and Team ID determine the destination. The request
does not attach `gitSource` or Git author metadata, so it does not rely on
GitHub account linking or commit-author recognition. The actual source branch
and commit are recorded separately as deployment provenance. This does not
rewrite commits or change your Git identity.

This stage establishes the regular deployment and domain. The API is not yet
forwarded to the Sandbox, so this is not a complete app acceptance check.
The regular Project does not need Supabase or Python worker environment variables.
The existing `feat/deploy` source committed to GitHub can build the frontend.
The local Step 3 script uploads its supervisor independently through the SDK;
the deployment helpers do not need to be uploaded to GitHub first.

References: [GitHub repository import permissions](https://vercel.com/docs/git/vercel-for-github#missing-git-repository),
[file deployment API](https://vercel.com/docs/rest-api/deployments/create-a-new-deployment),
[build configuration](https://vercel.com/docs/builds/configure-a-build),
[account authentication and email settings](https://vercel.com/docs/accounts).

## 2. Add a readable free domain

1. Open `Settings > Domains > Add Domain`.
2. Enter the full name you want, such as `model-anatomy-demo.vercel.app`.
   This is an example; the name must not already belong to another user.
3. Assign it to `Production` and save. Do not configure a redirect to the old URL.
4. Wait for a valid domain configuration, then record the full entry point URL,
   including `https://`.

Vercel manages DNS for `*.vercel.app`; no domain purchase or manual DNS records
are needed. You can also use the default `vercel.app` domain already assigned
to the Production Deployment.
Reference: [adding a domain](https://vercel.com/docs/domains/working-with-domains/add-a-domain).

## 3. Allow the entry point to access the API

Replace `your-demo.vercel.app` below with the actual domain from the previous
step. If the Sandbox is still running, first open its original `/health` and
`/api/v1/catalog/discovery` endpoints and record the storage backend and `total`.
Then select this Sandbox under the Vercel Project's `Sandboxes` tab and click
`Stop`. Keep the persistent Sandbox; do not delete it or repeat Step 2 to create
another one.

Run from the repository root:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" uv --cache-dir ./.tmp/uv-cache run --no-project \
  --with vercel-sandbox==0.10.0 \
  python -B deploy/03_start_sandbox_demo.py \
  --public-origin https://your-demo.vercel.app
```

Enter your Token, wait for readiness, and record the printed `Demo URL:`.
Both the original Sandbox origin and the new entry point are allowed.
The script resumes the same Sandbox with the same data directory. It does not
modify any routing rules in the Vercel dashboard.

## 4. Configure forwarding in the dashboard

1. Return to the regular Project and open `CDN > Routing Rules > Add Rule`.
2. Choose a rule name, such as `Sandbox demo`.
3. Select `Regex` matching and enter `^/(.*)$`.
4. Select the `Rewrite` action. Use the URL printed by the current Step 3 run
   followed by `/$1` as the destination, for example:

   ```text
   https://sb-3b1er9u5fw5e.vercel.run/$1
   ```

5. Cover all HTTP methods. Do not forward only `GET`; analysis submissions
   use `POST`.
6. Click `Save`. Use `Test Rules` to check `/`, `/health`, and `/api/v1/analyses`.
   Confirm that `/health` resolves to the Sandbox's exact `/health` path and
   `/assets/example.js` resolves to its exact `/assets/example.js` path. The
   destination must not contain a literal path placeholder or an extra path suffix.
7. Click `Publish` to apply the rule.

A Rewrite keeps the new `vercel.app` address visible in the browser.
Reference: [Project-Level Routing Rules](https://vercel.com/docs/routing/project-routing-rules).

The explicit regex capture avoids ambiguous wildcard interpolation in the CDN
editor. An HTTP 200 for `/` alone does not verify this rule: a broken rewrite
can still return the SPA HTML while JavaScript and CSS files return 404.
Verify the HTML's actual asset URLs and `/health` after publishing.

## 5. Verify the complete app

1. Open `/health` through the new entry point. Confirm that `database`, `storage`,
   and `worker` are `true`, and `storage_backend` is `local`.
2. Open `/api/v1/catalog/discovery` and compare the total with the count recorded
   before stopping the Sandbox.
3. Open `/app`, submit `HuggingFaceTB/SmolLM2-135M`, and confirm that the existing
   report and architecture graph open.
4. Refresh the analysis page and confirm that the route still loads.

If you receive `403 origin_rejected`, check that Step 3's `--public-origin`
exactly matches the full origin displayed in the browser and that services were
restarted to load the new configuration.
Include the same `--public-origin` on subsequent Sandbox resumes. If Step 3
prints a different Sandbox URL, update the Rewrite destination in the dashboard
and Publish again.

These are configuration instructions. The proxy entry point has not received
live acceptance verification before its domain is created.
