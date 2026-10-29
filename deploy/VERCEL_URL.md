# A free vercel.app entry point

Use this guide when the Project currently has only a Sandbox and displays
`No Production Deployment`. Reuse the existing Vercel Project; no Vercel CLI
or purchased domain is required.

Requests follow this route: browser -> regular Vercel Project CDN -> Sandbox.
Once forwarding is configured, the Sandbox still serves the frontend, FastAPI,
worker, and SQLite data. This entry point does not automatically resume a stopped
Sandbox or extend its session duration.

## 1. Prepare a regular Production Deployment

In the existing Project's dashboard:

1. Open `Settings > Build and Deployment` and configure:

   | Setting | Value |
   | --- | --- |
   | Root Directory | `frontend` |
   | Framework Preset | `Vite` |
   | Install Command | `npm ci` |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |

2. Open `Settings > Git` and connect this repository. The GitHub account used
   for browser authorization needs repository access. Your Vercel and GitHub
   usernames do not need to match. This workflow does not use CLI login.
3. Open `Settings > Environments > Production > Branch Tracking`, enter
   `feat/deploy`, and save.
4. Open `Deployments > Create Deployment` and select or enter `feat/deploy`.
   If the interface requests a Git reference, you can also enter the repository's
   `https://github.com/<owner>/<repo>/tree/feat/deploy` URL.
5. Wait for the branch deployment to display `Ready` and `Production`. If it is
   marked Preview, check Branch Tracking first. You can also select
   `Promote to Production` from that deployment's menu.

This stage establishes the regular deployment and domain. The API is not yet
forwarded to the Sandbox, so this is not a complete app acceptance check.
The regular Project does not need Supabase or Python worker environment variables.
The existing `feat/deploy` source committed to GitHub can build the frontend.
The local Step 3 script uploads its supervisor independently through the SDK;
the deployment helpers do not need to be uploaded to GitHub first.

References: [Vercel Git deployment and branch settings](https://vercel.com/docs/git),
[build configuration](https://vercel.com/docs/builds).

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
3. Select `Path pattern` matching and enter `/:path*`.
4. Select the `Rewrite` action. Use the URL printed by the current Step 3 run
   followed by `/:path*` as the destination, for example:

   ```text
   https://sb-3b1er9u5fw5e.vercel.run/:path*
   ```

5. Cover all HTTP methods. Do not forward only `GET`; analysis submissions
   use `POST`.
6. Click `Save`. Use `Test Rules` to check `/`, `/health`, and `/api/v1/analyses`.
7. Click `Publish` to apply the rule.

A Rewrite keeps the new `vercel.app` address visible in the browser.
Reference: [Project-Level Routing Rules](https://vercel.com/docs/routing/project-routing-rules).

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
