"""Publish the static frontend and visitor controls alongside a fresh Sandbox.

Production deployment metadata identifies the current Sandbox. Its immutable
rewrites and Function configuration are published together, without Git login.
The legacy project-level catch-all is disabled once, after the new build is ready.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace
from urllib.parse import quote

from support.credentials import access_token
from support.project_routes import ProjectRoutes, sandbox_origin

PROTOCOL = "visitor-controls-v1"
PROJECT_SETTINGS = {
    "rootDirectory": "frontend", "framework": "vite", "installCommand": "npm ci",
    "buildCommand": "VITE_DEMO_CONTROLS=true npm run build", "outputDirectory": "dist",
    "devCommand": None, "nodeVersion": "22.x",
}


def deployment_config(candidate: dict, public_origin: str) -> dict:
    sandbox_origin(candidate["demo_url"] + "/$1")
    return {
        "sandbox_name": candidate["sandbox_name"], "sandbox_origin": candidate["demo_url"],
        "commit": candidate["commit"], "app_dir": candidate["app_dir"], "public_origin": public_origin,
    }


def frontend_files(archive: bytes, candidate: dict, public_origin: str) -> list[dict]:
    import tarfile

    files = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        try:
            backend = source.extractfile("backend/model_anatomy/service/jobs.py")
        except KeyError:
            backend = None
        if backend is None:
            raise RuntimeError("The selected source is missing the backend demo admission protocol")
        with backend:
            tree = ast.parse(backend.read())
        protocols = [
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            and any(isinstance(target, ast.Name) and target.id == "DEMO_CONTROL_PROTOCOL" for target in node.targets)
        ]
        if protocols != [PROTOCOL]:
            raise RuntimeError("Merge the backend demo admission protection into the selected branch before deploying it")
        for member in source.getmembers():
            if not member.isfile() or not member.name.startswith("frontend/"):
                continue
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError("Cannot read committed frontend source")
            with stream:
                files.append({"file": member.name, "data": base64.b64encode(stream.read()).decode(), "encoding": "base64"})
    required = {
        "frontend/package.json", "frontend/package-lock.json", "frontend/index.html",
        "frontend/api/demo.py", "frontend/demo_control/__init__.py",
        "frontend/src/components/DemoControls.tsx", "frontend/requirements.txt", "frontend/.python-version",
    }
    names = {item["file"] for item in files}
    if not required.issubset(names):
        raise RuntimeError("Merge the visitor demo controls into the selected application branch before deploying it")
    generated = {"frontend/vercel.json", "frontend/demo_control/deployment.json"}
    if generated & names:
        raise RuntimeError("Generated Vercel configuration must not be tracked in the application source")
    origin = candidate["demo_url"]
    configuration = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "functions": {"api/demo.py": {
            "maxDuration": 300,
            "excludeFiles": "{src/**,e2e/**,public/**,node_modules/**,dist/**,scripts/**}",
        }},
        "rewrites": [
            {"source": "/api/v1/:path*", "destination": origin + "/api/v1/:path*"},
            {"source": "/health", "destination": origin + "/health"},
            {"source": "/.well-known/model-anatomy-deployment.json", "destination": origin + "/.well-known/model-anatomy-deployment.json"},
            {"source": "/((?!api(?:/|$)|assets(?:/|$)|health(?:/|$)|\\.well-known(?:/|$)).*)", "destination": "/index.html"},
        ],
    }
    for name, value in [("frontend/vercel.json", configuration), (
        "frontend/demo_control/deployment.json", deployment_config(candidate, public_origin),
    )]:
        files.append({"file": name, "data": json.dumps(value), "encoding": "utf-8"})
    return files


class Production:
    def __init__(self, routes: ProjectRoutes):
        self.routes = routes
        self.project_path = "/v9/projects/" + quote(routes.project_id, safe="")

    def project(self) -> dict:
        value = self.routes.request("GET", self.project_path)
        if value.get("id") != self.routes.project_id or value.get("accountId") != self.routes.team_id:
            raise RuntimeError("The selected Vercel project/team could not be verified")
        return value

    def current(self, *, required: bool = True) -> dict | None:
        target = (self.project().get("targets") or {}).get("production") or {}
        if not target.get("id"):
            if not required:
                return None
            raise RuntimeError("The project has no Production deployment")
        return self.detail(target["id"])

    def detail(self, deployment_id: str) -> dict:
        value = self.routes.request("GET", "/v13/deployments/" + quote(deployment_id, safe=""))
        if value.get("projectId") != self.routes.project_id:
            raise RuntimeError("The deployment belongs to another project")
        return value

    def configure_credentials(self) -> None:
        # These three private, server-only settings are derived from Actions
        # Secrets. Never log this request or store its body in diagnostic JSON.
        response = self.routes.request("POST", "/v10/projects/" + quote(self.routes.project_id, safe="") + "/env", [
            {"key": "DEMO_VERCEL_" + key, "value": value, "type": "sensitive", "target": ["production"]}
            for key, value in [("TOKEN", self.routes.token), ("PROJECT_ID", self.routes.project_id), ("TEAM_ID", self.routes.team_id)]
        ], upsert="true")
        created = response.get("created", [])
        created = [created] if isinstance(created, dict) else created
        expected = {"DEMO_VERCEL_TOKEN", "DEMO_VERCEL_PROJECT_ID", "DEMO_VERCEL_TEAM_ID"}
        if response.get("failed") or {item.get("key") for item in created} != expected:
            raise RuntimeError("The private Production control settings could not all be confirmed")

    def create(self, archive: bytes, candidate: dict, config: dict, branch: str) -> dict:
        files = frontend_files(archive, candidate, config["public_origin"])
        self.configure_credentials()
        return self.routes.request("POST", "/v13/deployments", {
            "name": self.project()["name"], "project": self.routes.project_id, "target": "production",
            "projectSettings": dict(PROJECT_SETTINGS), "files": files,
            "meta": {
                "modelAnatomyProtocol": PROTOCOL, "modelAnatomySourceBranch": branch,
                "modelAnatomySourceCommit": candidate["commit"],
                "modelAnatomySandbox": candidate["sandbox_name"], "modelAnatomySandboxOrigin": candidate["demo_url"],
            },
        })

    def wait(self, deployment_id: str) -> dict:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            value = self.detail(deployment_id)
            state = value.get("readyState")
            if state == "READY":
                return value
            if state in {"ERROR", "CANCELED"}:
                raise RuntimeError("The frontend/controls build failed; inspect its Vercel build logs")
            print(f"Frontend deployment: {state}", flush=True)
            time.sleep(5)
        raise RuntimeError("The frontend build may still be running; inspect Vercel before retrying")

    def cancel_pending(self, deployment_id: str) -> None:
        if self.detail(deployment_id).get("readyState") in {"READY", "ERROR", "CANCELED"}:
            return
        self.routes.request("PATCH", "/v12/deployments/" + quote(deployment_id, safe="") + "/cancel", {})
        if self.detail(deployment_id).get("readyState") not in {"ERROR", "CANCELED"}:
            raise RuntimeError("The pending build cancellation could not be confirmed")

    def restore(self, previous_id: str, expected_id: str) -> None:
        if self.current()["id"] != expected_id:
            raise RuntimeError("Production changed again; no rollback was attempted")
        self.routes.request("POST", f"/v10/projects/{quote(self.routes.project_id, safe='')}/promote/{quote(previous_id, safe='')}", {})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.current()["id"] == previous_id:
                return
            time.sleep(2)
        raise RuntimeError("The previous Production deployment could not be confirmed")


async def published(config: dict, production: Production, route: dict, ops, *, initialize: bool = False) -> tuple[dict | None, dict | None]:
    detail = await asyncio.to_thread(production.current, required=False)
    rule = route["rule"]
    if detail is None:
        if rule is not None:
            raise RuntimeError("A legacy CDN rule without a Production deployment requires inspection")
        return None, None
    meta = detail.get("meta", {})
    if meta.get("modelAnatomyProtocol") == PROTOCOL:
        if rule is not None and rule.get("enabled", True):
            raise RuntimeError("The old catch-all still overrides the interactive frontend; inspect routing")
        origin = meta["modelAnatomySandboxOrigin"]
        sandbox_origin(origin + "/$1")
        current = await ops.current_sandbox(config, origin)
        if current["sandbox_name"] != meta.get("modelAnatomySandbox") or current["commit"] != meta.get("modelAnatomySourceCommit"):
            raise RuntimeError("Production metadata does not match its prepared Sandbox")
        return detail, current
    if rule is None:
        if not initialize or any(key.startswith("modelAnatomy") for key in meta):
            raise RuntimeError("For a new project's placeholder only, run Deploy Demo with Initialize project selected")
        return detail, None
    if not rule.get("enabled", True):
        raise RuntimeError("No interactive Production deployment or active legacy demo was found")
    return detail, await ops.current_sandbox(config, route["origin"])


def deployment_id(detail: dict | None) -> str | None:
    return detail["id"] if detail else None


def verify_entrypoint(config: dict, candidate: dict, ops) -> dict:
    evidence = ops.verify(config["public_origin"], candidate)
    status = ops.read_url(config["public_origin"] + "/api/demo")
    if status.get("state") != "running" or status.get("ready") is not True:
        raise RuntimeError("The public Start/Stop controller did not verify the running Sandbox")
    evidence["controls"] = True
    return evidence


async def execute(args, config: dict, ops) -> None:
    if args.action == "recover":
        print("Scheduled recovery is disabled for visitor-controlled demos. Use Start Demo explicitly.")
        return
    routes = ProjectRoutes(config["project_id"], config["team_id"], access_token(), config["route_name"])
    await asyncio.to_thread(routes.check_project, config["public_origin"], require_production=False)
    route = await asyncio.to_thread(routes.live, allow_disabled=True, allow_missing=True)
    production = Production(routes)
    before, previous = await published(config, production, route, ops, initialize=getattr(args, "initialize_project", False))
    evidence = {"action": args.action, "previous": previous, "public_origin": config["public_origin"]}
    baseline_origin = ((before or {}).get("meta", {}).get("modelAnatomySandboxOrigin") or route["origin"] or config["public_origin"])
    evidence["baseline"] = await asyncio.to_thread(ops.record_baseline, baseline_origin)
    ops.save(args.work_dir / "result.json", evidence)
    if evidence["baseline"]["health"].get("storage_backend") not in (None, "local"):
        raise RuntimeError("Another storage profile was preserved")
    if args.action == "start":
        if previous is None:
            raise RuntimeError("Run Deploy Demo first; there is no published Sandbox to resume")
        await ops.START.start(SimpleNamespace(output=args.work_dir / "demo-state.json", public_origin=config["public_origin"]), previous)
        evidence["verification"] = await asyncio.to_thread(verify_entrypoint, config, previous, ops)
        evidence["ready"] = True
        ops.save(args.work_dir / "result.json", evidence)
        await asyncio.to_thread(ops.append_summary, f"## Start Demo\n\nReady: {config['public_origin']}\n\nExisting code and data were preserved.\n")
        return
    revision = await asyncio.to_thread(ops.source_revision, args.source_repo)
    revision, archive, count = await asyncio.to_thread(ops.PREPARE.committed_source, args.source_repo, args.source_branch, revision)
    # Fail before provisioning if main has not received the required controls.
    placeholder = {"sandbox_name": "validation", "commit": revision, "demo_url": "https://sb-validation.vercel.run", "app_dir": "/vercel/model-anatomy"}
    frontend_files(archive, placeholder, config["public_origin"])
    print(f"Source: {args.source_branch} at {revision}; files: {count}", flush=True)
    state_path = args.work_dir / "sandbox-state.json"
    created = None
    migrated = None
    publication_attempted = False
    migration_attempted = False
    try:
        await ops.PREPARE.prepare(SimpleNamespace(
            state=state_path, project_id=config["project_id"], team_id=config["team_id"], branch=args.source_branch,
        ), revision, archive)
        candidate = json.loads(state_path.read_text())
        output = args.work_dir / "demo-state.json"
        await ops.START.start(SimpleNamespace(output=output, public_origin=config["public_origin"]), candidate)
        candidate["demo_url"] = json.loads(output.read_text())["demo_url"]
        evidence["current"] = candidate
        await asyncio.to_thread(ops.verify, candidate["demo_url"], candidate)
        # Recheck both independently editable resources before publication.
        if deployment_id(await asyncio.to_thread(production.current, required=False)) != deployment_id(before) or await asyncio.to_thread(routes.live, allow_disabled=True, allow_missing=True) != route:
            raise RuntimeError("Production or routing changed during preparation; the existing demo was preserved")
        publication_attempted = True
        created = await asyncio.to_thread(production.create, archive, candidate, config, args.source_branch)
        evidence["deployment_id"] = created["id"]
        ops.save(args.work_dir / "result.json", evidence)
        await asyncio.to_thread(production.wait, created["id"])
        if (await asyncio.to_thread(production.current))["id"] != created["id"]:
            raise RuntimeError("Another Production deployment became active; inspect Vercel")
        if route["rule"] is not None and route["rule"].get("enabled", True):
            migration_attempted = True
            migrated = await asyncio.to_thread(routes.set_enabled, route, False)
        for attempt in range(10):
            try:
                evidence["verification"] = await asyncio.to_thread(verify_entrypoint, config, candidate, ops)
                break
            except (OSError, ValueError, RuntimeError):
                if attempt == 9:
                    raise
                await asyncio.sleep(3)
        evidence["ready"] = True
    except BaseException:
        # Creation can succeed even when its response is lost. Never stop the
        # candidate unless Production is positively known to target the old one.
        try:
            if (publication_attempted and not created) or (migration_attempted and not migrated):
                raise RuntimeError("A publication response was lost; remote changes may still complete")
            if created:
                await asyncio.to_thread(production.cancel_pending, created["id"])
            current = await asyncio.to_thread(production.current, required=False)
            if created and deployment_id(current) == created["id"]:
                if before is None:
                    raise RuntimeError("The first Production deployment needs inspection; there is no previous deployment to restore")
                await asyncio.to_thread(production.restore, before["id"], created["id"])
            if migrated:
                await asyncio.to_thread(routes.set_enabled, migrated, True)
            if deployment_id(await asyncio.to_thread(production.current, required=False)) == deployment_id(before) and state_path.is_file():
                await ops.stop_sandbox(json.loads(state_path.read_text()))
            else:
                print("Cleanup skipped: inspect the current Production deployment and Sandbox.")
        except Exception:  # noqa: BLE001 - Preserve any uncertain live target.
            print("Rollback/cleanup could not be confirmed. Inspect Vercel before retrying.")
        ops.save(args.work_dir / "result.json", evidence)
        raise
    if previous is not None:
        try:
            await ops.stop_sandbox(previous)
            evidence["previous_stopped"] = True
        except Exception:  # noqa: BLE001 - Do not invalidate a healthy published deployment.
            evidence["previous_stopped"] = False
            print("::warning::The new demo is ready; stopping the previous Sandbox was not confirmed.")
    ops.save(args.work_dir / "result.json", evidence)
    await asyncio.to_thread(ops.append_summary, (
        f"## Deploy Demo\n\nReady: {config['public_origin']}\n\nSource: `{args.source_branch}` at `{revision}`\n\n"
        f"Sandbox: `{candidate['sandbox_name']}`\n\nCatalog records: {evidence['verification']['catalog_total']}\n\n"
        "Visitors can use Start Demo / Stop Demo on the page. Sessions end naturally after 45 minutes.\n"
    ))
