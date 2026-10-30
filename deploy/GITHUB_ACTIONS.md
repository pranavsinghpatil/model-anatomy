# Operate the Vercel demo with GitHub Actions

For an independent deployment into your own account, follow
[the complete first-time setup](README.md#1-prepare-the-github-repository).
This page describes daily operation, automatic behavior, account handover,
legacy migration, and failure recovery. The deployment target is selected by
private credentials and a repository domain Variable, not a previous operator's
tracked config or local state.

## Configuration and ownership

One repository has one active deployment destination. Configure these at
**Repository > Settings > Secrets and variables > Actions**:

| Storage | Exact name | Meaning |
| --- | --- | --- |
| Repository Secret | `VERCEL_TOKEN` | Token scoped to the Vercel team that owns the project |
| Repository Secret | `VERCEL_PROJECT_ID` | That team's Project ID |
| Repository Secret | `VERCEL_TEAM_ID` | Owning Team ID |
| Repository Variable | `DEMO_PUBLIC_ORIGIN` | Assigned Production origin, such as `https://your-demo.vercel.app` |

Create Secrets on the **Secrets** tab with **New repository secret**, and the
public domain on **Variables > New repository variable**. Set all four for your
own account/project. A fork must configure its own settings. The domain must
already be assigned to the selected project, without a redirect or branch
assignment. A missing domain does not fall back to anyone else's URL.

For optional local cloud commands, use ignored `deploy/.env`, following
[.env.example](.env.example). CI never loads that file. Environment values take
precedence locally. The tracked `github_demo.json` contains only shared behavior
and the legacy route name; it contains no account ID, Token, or operator domain.
Diagnostic state JSON omits Project/Team IDs and Tokens. Never put account
credentials in workflow inputs, README values, frontend env files, or `VITE_`
variables. The selected application source is exported from committed Git files;
untracked local secrets are not uploaded.

Leave **Vercel Project > Settings > Git** disconnected. Actions uploads files
using explicit Token authentication, with separate branch/commit provenance.
GitHub and Vercel login emails may differ. No Vercel CLI, Git author rewrite,
GitHub import permission, or extra GitHub Token is required for these runs.
GitHub automatically supplies its own repository checkout token.

## Deploy Demo: first run and code updates

Before the first run, merge the complete deployment implementation into default
branch `main`. Both the workflow and selected application branch must include
the controls, Function, and backend admission protection. GitHub exposes the
manual button only after the workflow exists on its default branch.

1. Push the desired application commits to `feat/deploy` or `main`.
2. Open **Actions > Deploy Demo > Run workflow**.
3. Set **Use workflow from** to `main`. This selects the reviewed controller.
4. Choose **Application source branch** separately: `feat/deploy` or `main`.
   The source checkout's resolved commit is recorded; later pushes are not
   silently included in that already-running deployment.
5. On a fresh project's first run, check **First setup only - replace this
   project's placeholder frontend** (`initialize_project`). Leave it unchecked
   for normal updates or migration from the earlier Sandbox setup. The flag
   permits replacing a placeholder, not arbitrary incompatible Model Anatomy
   protocols or another account's resources. A truly empty, unlinked project
   with an assigned domain also works without an earlier Production deployment.
6. Run and wait for green, then read **Summary**. It identifies the branch,
   commit, Sandbox, and your public URL. Open that URL in an incognito window.

During this manually initiated run, Actions automatically:

1. Checks the configured account, project, domain, Git disconnection, and routing.
   Unpublished routing edits or unrelated active CDN rules stop deployment.
2. Records available previous health/storage/catalog evidence. Validates the
   selected committed source before provisioning; missing controls fail early.
3. Creates a fresh persistent Sandbox with empty SQLite/local storage. Installs
   locked Python dependencies and the CPU engine, builds the frontend, and starts
   API, worker, and Caddy. Checks services, cookies, SPA/assets, and source marker.
4. Uploads matching Production frontend/Function source, generated Sandbox
   configuration, and immutable API-only rewrites. Configures private control
   credentials, waits for the build, and checks the public controller.
5. For legacy migration only, disables/publishes the existing all-path CDN rule.
   New projects require no CDN rule and no local scripts 01-04.
6. After public checks pass, stops the previously published Sandbox in this same
   project if one exists. It does not stop a previous operator's other account.
7. Writes a Summary and uploads redacted diagnostics retained for seven days.
   The runner exits; it does not host the app or need to remain running.

The workflow creates three **sensitive, Production-only** Vercel environment
settings from the GitHub Secrets: `DEMO_VERCEL_TOKEN`, `DEMO_VERCEL_PROJECT_ID`,
and `DEMO_VERCEL_TEAM_ID`. They stay server-side. Generated Function config and
frontend files contain no account credentials. The control Function has only
`vercel-sandbox==0.10.0`, uses Python 3.12, and requests a 300-second maximum.

Build settings are supplied with each uploaded deployment:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Framework | Vite |
| Install Command | `npm ci` |
| Build Command | `VITE_DEMO_CONTROLS=true npm run build` |
| Output Directory | `dist` |
| Development Command | Automatic; Override off |
| Node.js | `22.x` |

**Deploy Demo resets demo data.** Existing analysis links/reports are not moved
to its new Sandbox. Deploy after existing work finishes: publication eventually
stops the previous environment. The domain stays the same within the selected
project. Changing accounts does not transfer a domain or data.

A GitHub push alone does not update this deployment. Changing Vercel's Git
Production Branch does not select the source for these file uploads. Select
`main` in the manual **Application source branch** field to deploy main; no
code change or local Linux command is needed after setup.

## Start Demo: resume the published version

Open **Actions > Start Demo > Run workflow**, use workflow from `main`, and run.
This resumes the Sandbox identified by current Production deployment metadata,
restarts services if needed, and verifies the public URL. It preserves its
existing code, SQLite database, and reports. It does not fetch branch changes,
create a new Sandbox, or republish frontend source.

Run Deploy Demo successfully first. If no published demo exists, Start fails
rather than creating an unknown replacement. An expired/deleted snapshot or
missing Sandbox requires an explicit new Deploy Demo, with empty data.

There is no separate Stop workflow. A visitor can use Stop Demo when globally
idle, or the operator can stop the Sandbox in Vercel's dashboard when appropriate.
The dashboard's provider Stop does not apply the application's busy-job guard.

## Visitor controls and automatic behavior

The **Interactive demo** strip appears above the app header on home, catalog,
and inspection pages. The build enables it with `VITE_DEMO_CONTROLS=true`.
It works at your configured `DEMO_PUBLIC_ORIGIN`, which is checked for mutation
requests. The raw `sb-....vercel.run` frontend is not this external controller.

| Trigger | Automatic behavior |
| --- | --- |
| Page open or reload | Loads the static frontend and requests status. Does not start Sandbox. |
| Status polling | Observes readiness and busy jobs. Does not resume, create, or restart anything. |
| Visitor Start | Resumes the prepared published Sandbox, starts services, and polls readiness. No source update, data reset, or clock extension. |
| Visitor Stop | Checks all visitors' queued/running jobs on the server. If any exist, refuses Stop. Otherwise stops the exact inspected session and preserves saved files. |
| Natural 45-minute timeout | Sandbox stops, possibly interrupting work. Static frontend and Function remain available; the next Start must be explicit. |
| GitHub push | May run separate quality CI; does not run Deploy Demo or Start Demo. |
| Calendar/cron | No scheduled restart or recovery workflow is installed. |
| End of public-start window | Refuses new public Starts. Does not force-stop an active session at midnight. |

All visitors share one Sandbox. Repeated Start requests do not intentionally
create new environments or reset the session timer. The UI waits for health,
database, storage, and worker checks before reporting ready. Status failures
disable both buttons until a successful retry; Stop always rechecks server-side.

Safe Stop serializes the global idle check with SQLite job admission. A
per-session fence prevents new analyses while provider Stop is in flight. If a
response is lost, the old session stays fenced until it ends; retrying Start
within that same session does not silently remove the fence. A genuinely new
resumed session has a new identity and accepts work again. No busy-job guard can
extend Vercel's natural session limit.

Public Start remains allowed through **December 16, 2026 in New York**, ending
December 17 at 00:00 EST / 05:00 UTC. This demo deadline is enforced in code;
it is not an account setting and is unchanged by switching credentials. Status
and Stop remain available afterward, as does the operator's manual Actions
Start. To extend a future demo, review/update the policy and boundary tests
rather than assuming a new account resets the date.

If an older Recover Demo workflow was previously enabled, disable it and remove
its scheduled file from default `main`. The earlier `DEMO_RECOVERY_ENABLED`
Variable is no longer used. Leaving an older scheduled workflow on GitHub can
still trigger that old implementation; the current two manual workflows have
no schedule or push trigger.

## Switch the deployment to another account

This is a sequential handover, with one active repository destination. It does
not transfer Vercel ownership or support choosing between two live destinations.

1. Wait for in-progress and queued Deploy/Start runs to finish. Do not change
   Secrets halfway through a run.
2. The new owner follows [README steps 2 and 3](README.md#2-create-your-vercel-project-in-the-browser)
   to create their own Project, disconnect Git, choose their own Production
   domain, and obtain their own account settings. No old credentials are needed.
3. Replace **all three** repository Secrets with the new owner's values and
   replace `DEMO_PUBLIC_ORIGIN` on the Variables tab. Treat these as one
   configuration set; do not mix old and new account values.
4. Run Deploy Demo with initialization selected for the new placeholder project
   and the intended application branch. Complete the full acceptance checks in
   [README step 5](README.md#5-verify-the-complete-app).
5. Share the new domain with judges. The previous domain is not automatically
   transferred; a transfer is separate from this workflow.
6. Only after verifying the new deployment, the previous operator stops their
   old Sandbox in their own account and revokes their old deployment Token.
   Replacing GitHub Secrets does not revoke credentials already installed in
   the old project's Function or stop old resources. Old snapshots also remain
   until expiry or deliberate deletion by that account's owner.
7. If using local cloud tools, configure a new private `deploy/.env`. Do not
   reuse another account's `sandbox-state.json`, `demo-state.json`, or
   `production-state.json`; the browser workflow needs none of these files.

Rotating a Token in the same account also requires a new Deploy Demo to publish
the Function with the replacement credential. Merely changing the GitHub Secret
or running Start Demo does not update an existing Function's environment.
Deploy Demo still creates empty data, so coordinate the rotation accordingly.

## Existing all-Sandbox deployment migration

Skip this section for a fresh project with no legacy rule. For the earlier
manual setup, `route_name` in `github_demo.json` selects the old rule, normally
`Model Anatomy Sandbox`. It must uniquely identify an unconditional Regex
Rewrite `^/(.*)$` to `https://sb-....vercel.run/$1`, without unpublished edits.

Use your current credentials and set `DEMO_PUBLIC_ORIGIN` to that project's
assigned Production origin. Run Deploy Demo with initialization unchecked.
It validates the current legacy Sandbox, builds the new interactive entry point,
and disables the catch-all only after the new build is ready. Keep it disabled
afterward, or remove it after verifying migration. Later interactive workflows
can operate without any legacy rule.

Do not recreate or re-enable a catch-all on a fresh/interactive deployment. It
would send the frontend and `/api/demo` into Sandbox, hiding controls when stopped.
The controller follows published Production metadata, not that old rule's target.

## Limits, failures, and troubleshooting

Hobby limits include 5 Active CPU-hours, 420 provisioned-memory GB-hours, 20 GB
Sandbox transfer per cycle, a 15 GB snapshot allowance, and 45-minute sessions.
At 8 GB, one full idle session still consumes 6 GB-hours. A full memory allowance
covers at most 52.5 runtime hours before preparation and other usage. CPU usage
sums actual computation across cores. Restarting refills neither quota. Follow
your actual Usage period rather than assuming a December 1 reset. Continuous
December 1-15 operation would require 2,880 GB-hours; buttons save idle quota but
cannot guarantee continuous free availability.

The latest snapshot is retained for 60 days after last use, with one snapshot
per Sandbox. Previous deployed Sandboxes retain their own snapshots; stopping
them does not delete storage. Monitor Vercel Usage and clean up unneeded old
resources deliberately. Repository Actions quotas also apply.

Deploy and Start share queued concurrency and do not cancel an active run.
Avoid manual Production/routing changes during runs. On failure, the controller
restores prior Production/routing only when ownership of the change is known.
It cancels pending builds before candidate cleanup. Lost publication responses
or conflicting changes preserve potentially published Sandboxes for inspection.
On a truly first Production deployment there is no older deployment to restore;
a published candidate is preserved if final checks fail. Do not retry blindly
after a network failure or cancellation.

| Symptom | Action |
| --- | --- |
| No Run workflow button | Put the workflow on default `main`, enable Actions, and check repository permissions. |
| Missing credential/domain setting | Check the exact Secrets/Variables names above. No local env file is available to CI. |
| Project/domain/account preflight fails | Ensure all four settings refer to the same new owner's project; disconnect Git and check Production domain assignment. |
| Placeholder replacement refused | For a genuinely new project only, select initialization on its first Deploy Demo. Do not use it to bypass an unknown protocol warning. |
| No Interactive demo strip | Use the configured public origin after successful Deploy Demo. Check controls-enabled build; scripts 01-04 alone do not publish it. |
| Local strip has disabled buttons | Expected with plain Vite: no local `/api/demo` Function is implemented. See [local modes](README.md#choose-how-to-run-the-app-locally). |
| Deployed strip reports status error | Inspect `/api/demo` and Function logs, the generated deployment config, and server-only credentials; do not expose their values. |
| Start is disabled | Read status: already ready, operation pending, unknown status, stop fence, or public-start deadline. |
| Stop is refused | An analysis may belong to another visitor, or another session operation is underway. Do not bypass with provider Stop during work. |
| Whole site fails when Sandbox stops | Check for an enabled project CDN catch-all. Frontend and controller must come from regular Production. |
| Signed-out visitor sees Vercel login | Review Deployment Protection for the chosen Production domain. |
| Build fails | Inspect the failed run and Vercel build logs. Check selected source includes the full change and the project has remaining quota. |
| Snapshot expired, Sandbox deleted, or no published demo | Run explicit Deploy Demo; Start does not silently provision a replacement. |

For an interrupted legacy migration, first verify the new interactive Production
build identity. Only then disable the old catch-all and Publish, and complete
frontend/controller/health/inspection checks before stopping the old Sandbox.
If identity is uncertain, inspect diagnostics and dashboard targets first.

## Verification and remaining live checks

From the repository root, offline deployment checks are:

```bash
mkdir -p .tmp
TMPDIR="$PWD/.tmp" PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s deploy/tests -q
TMPDIR="$PWD/.tmp" uv run --no-sync ruff check deploy frontend/api frontend/demo_control
```

The deployment suite covers account-independent settings, private project
creation, fresh-project initialization, legacy migration, updates, rollback,
public-control policy, actual SQLite lock/fence behavior, and Function handlers.
It does not contact Vercel. Browser tests use mocked API/control responses.

The current account-handover increment passes 85 deployment tests. Earlier
interactive-control validation also passed 45 API/admission tests, 111
service/worker/catalog/quota tests, 10 desktop/mobile control browser tests,
ordinary and controls-enabled frontend builds, and frontend formatting. See
[the CI commands](../AGENTS.md#reproducible-build-and-test-setup).

Live Token permissions, Function packaging, routing propagation, the signed-out
visitor flow, report persistence after resume, and actual resume latency still
require the first authenticated Deploy Demo and acceptance checks. No successful
cloud deployment is inferred from offline tests.

Actionlint 1.7.12 predates GitHub's documented `concurrency.queue: max`. Its
unsupported-queue diagnostic is the only compatibility suppression used when
validating these workflows; the setting follows GitHub's official documentation.

References: [manual runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow),
[Secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets),
[Variables](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-variables),
[concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency),
[Python Functions](https://vercel.com/docs/functions/runtimes/python/api-directory),
[Function limits](https://vercel.com/docs/functions/limitations),
[Sandbox persistence](https://vercel.com/kb/guide/vercel-sandbox-duration-and-persistence),
[Sandbox quotas](https://vercel.com/docs/sandbox/pricing).
