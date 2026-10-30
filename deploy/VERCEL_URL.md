# Choose and understand the public demo URL

For a fresh deployment, follow [README](README.md). A teammate uses their own
Vercel Project/domain and their own credentials. The current interactive setup
does not require manually creating a CDN forwarding rule.

## Choose a free readable address

1. Open your Vercel Project, then **Settings > Domains**.
2. Keep its default `vercel.app` domain, or edit/add an available name such as
   `your-demo.vercel.app`. The name must not belong to another project/operator.
3. Assign it to **Production**, without a redirect or Git branch assignment.
   Wait for valid configuration. Vercel handles DNS for its own `vercel.app`
   names; no domain purchase or external DNS setup is needed.
4. In **GitHub > Repository Settings > Secrets and variables > Actions >
   Variables**, set `DEMO_PUBLIC_ORIGIN` to the full origin, for example
   `https://your-demo.vercel.app`, without a path.
5. Run **Deploy Demo**. For a new project's placeholder, select initialization
   on the first run. For later URL changes, assign the new domain first, update
   the Variable, and Deploy Demo again so the controller's allowed origin and
   Sandbox API configuration match. This creates a new empty demo environment.
6. Open that address signed out and run the [acceptance checks](README.md#5-verify-the-complete-app).

Do not assume a project name reserves the exact matching domain; use the domain
actually assigned in Settings. Do not redirect it to `sb-....vercel.run`.
A previous account's URL does not move merely because GitHub Secrets change.
See [account handover](GITHUB_ACTIONS.md#switch-the-deployment-to-another-account).

## How the interactive URL works

```text
Browser -> your-demo.vercel.app
           |-- frontend assets and SPA routes -> regular Production deployment
           |-- /api/demo                     -> Python control Function
           |-- /api/v1/* and /health          -> selected Sandbox
           `-- deployment identity marker    -> selected Sandbox
```

The static frontend and control Function remain available when Sandbox stops.
Only an explicit Start resumes it. API-only rewrites are generated with each
Production upload, together with matching Sandbox identity. The shared tracked
config contains no operator's domain. Readiness verifies the HTML's actual
JavaScript/CSS URLs as well as health and source identity.

Vercel automatically assigns the Sandbox's `sb-....vercel.run` address. The
pinned Python SDK configures routes/ports but does not rename this address to a
chosen `vercel.app` name. The readable URL belongs to the ordinary Project.
Renaming a Sandbox or Project does not change an existing generated Sandbox URL.
The raw Sandbox frontend is not the external visitor-control entry point.

## Migrating the earlier manual CDN setup

This section applies only to an existing all-Sandbox deployment, not a fresh
teammate's project. Older scripts 01-04 prepared the Sandbox and a static
Production frontend, then used this project-level rule:

| Field | Legacy value |
| --- | --- |
| Name | Shared `route_name` in `github_demo.json`, normally `Model Anatomy Sandbox` |
| Request Path | Regex `^/(.*)$` |
| Action | Rewrite |
| Destination | The current Sandbox origin followed by `/$1`, such as `https://sb-example.vercel.run/$1` |
| Methods | All, including POST |

The earlier manual Step 3 needed `--public-origin` to allow that Project origin;
Save and Publish were required for CDN edits. That all-path route sent frontend,
API, and worker traffic into Sandbox, so timeout made the entire URL unavailable.
It never supplied a working external `/api/demo` controller.

For migration, configure your private credentials/domain and use Deploy Demo
with initialization unchecked. It validates the uniquely named legacy rule,
builds the interactive deployment, then disables and publishes the old rule
before checking the public controller and stopping the old Sandbox. See
[the migration procedure](GITHUB_ACTIONS.md#existing-all-sandbox-deployment-migration).
After successful migration, leave the legacy rule disabled or remove it. New
projects and later interactive runs need no such rule.

Do not re-enable or recreate this catch-all for the interactive frontend: it
would hide the page and controls whenever Sandbox stops. Unpublished edits,
duplicate legacy names, unrelated active rules, and malformed legacy rewrites
cause the workflow to stop for inspection.

References: [project creation](https://vercel.com/docs/projects/managing-projects),
[domains](https://vercel.com/docs/domains/working-with-domains),
[adding a domain](https://vercel.com/docs/domains/working-with-domains/add-a-domain),
[project routing](https://vercel.com/docs/routing/project-routing-rules),
[Python Sandbox SDK](https://vercel.com/docs/sandbox/python-sdk-reference).
