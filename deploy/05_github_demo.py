"""Deploy a fresh demo or resume the published Sandbox from GitHub Actions."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from support.credentials import access_token, bind_destination, destination, without_destination
from support.project_routes import ProjectRoutes
from support.recovery_window import recovery_allowed

DEPLOY = Path(__file__).resolve().parent
MARKER = "/.well-known/model-anatomy-deployment.json"


def helper(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DEPLOY / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the Sandbox deployment helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = helper("sandbox_prepare", "02_prepare_sandbox.py")
START = helper("sandbox_start", "03_start_sandbox_demo.py")


def read_url(url: str, *, as_json: bool = True):
    request = Request(url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=20) as response:
        return json.load(response) if as_json else response.read(1024 * 1024).decode()


def verify(origin: str, expected: dict | None = None) -> dict:
    evidence = START.public_checks(origin)
    page = read_url(origin + "/", as_json=False)
    assets = set(re.findall(r'''(?:src|href)=["'](/assets/[^"']+\.(?:js|css))["']''', page))
    if not any(path.endswith(".js") for path in assets):
        raise RuntimeError("The frontend HTML has no built JavaScript asset")
    for path in sorted(assets):
        with urlopen(origin + path, timeout=20) as response:
            content_type = response.headers.get_content_type()
            expected_types = {"text/css"} if path.endswith(".css") else {"text/javascript", "application/javascript"}
            if response.status != 200 or content_type not in expected_types:
                raise RuntimeError("A frontend asset has an unexpected HTTP status or content type")
    if expected is not None:
        marker = read_url(origin + MARKER)
        if marker != {key: expected[key] for key in ["commit", "sandbox_name"]}:
            raise RuntimeError("The public URL is serving another Sandbox or source commit")
        evidence["deployment"] = marker
    evidence["assets"] = sorted(assets)
    return evidence


def record_baseline(origin: str) -> dict:
    evidence = {}
    for key, path in [("health", "/health"), ("catalog", "/api/v1/catalog/discovery")]:
        try:
            document = read_url(origin + path)
            evidence[key] = (
                {name: document.get(name) for name in ["storage_backend", "checks", "analysis_available"]}
                if key == "health" else {"total": document.get("total")}
            )
        except HTTPError as exc:
            evidence[key] = {"http_status": exc.code}
        except (OSError, ValueError):
            evidence[key] = {"unavailable": True}
    return evidence


async def current_sandbox(config: dict, origin: str) -> dict:
    from vercel import sandbox
    from vercel.api import session
    from vercel.sandbox import SandboxCredentials, SandboxQueryByCreatedAt, SandboxServiceOptions, TagFilter

    async def credentials():
        return SandboxCredentials(token=access_token(), project_id=config["project_id"], team_id=config["team_id"])

    matches = []
    async with session(service_options=[SandboxServiceOptions(credentials_factory=credentials)]):
        query = SandboxQueryByCreatedAt(tag=TagFilter(key="app", value="model-anatomy"))
        async for listed in sandbox.query_sandboxes(query=query):
            box = await sandbox.get_sandbox(name=listed.name)
            if not any(route.port == 8080 and not route.system and route.url.rstrip("/") == origin for route in box.routes):
                continue
            tags = box.tags or {}
            if tags.get("purpose") != "demo" or not re.fullmatch(r"[0-9a-f]{40}", tags.get("commit", "")):
                raise RuntimeError("The published Sandbox lacks the expected demo source provenance")
            if not box.cwd or not PurePosixPath(box.cwd).is_absolute():
                raise RuntimeError("The published Sandbox has no absolute working directory")
            matches.append({
                "project_id": config["project_id"], "team_id": config["team_id"],
                "sandbox_name": box.name, "commit": tags["commit"],
                "branch": tags.get("branch"), "app_dir": str(PurePosixPath(box.cwd) / "model-anatomy"),
                "prepared": True,
                "status": str(box.status) if box.status is not None else None,
            })
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely locate the published Sandbox; no environment was replaced")
    return matches[0]


async def stop_sandbox(state: dict) -> None:
    from vercel import sandbox
    from vercel.api import session
    from vercel.sandbox import SandboxCredentials, SandboxServiceOptions

    state = bind_destination(state)

    async def credentials():
        return SandboxCredentials(token=access_token(), project_id=state["project_id"], team_id=state["team_id"])

    async with session(service_options=[SandboxServiceOptions(credentials_factory=credentials)]):
        box = await sandbox.get_sandbox(name=state["sandbox_name"])
        await box.stop()


async def publish_checked(routes, baseline: dict, candidate: dict, check) -> dict:
    """Keep the previous target unless both publication and proxy checks succeed."""
    published = await asyncio.to_thread(routes.publish, baseline, candidate["demo_url"])
    try:
        return await check()
    except Exception:  # noqa: BLE001 - Compensate for any failed check after publishing.
        current = await asyncio.to_thread(routes.live)
        if current != published:
            raise RuntimeError("Proxy checks failed and CDN changed again; inspect routing before retrying") from None
        await asyncio.to_thread(routes.publish, current, baseline["origin"])
        raise RuntimeError("Proxy checks failed; the previous CDN target was restored") from None


def save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(without_destination(document), indent=2) + "\n", encoding="utf-8")


def source_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def append_summary(summary: str) -> None:
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
            output.write(summary)


async def execute(args: argparse.Namespace, config: dict) -> None:
    if config.get("interactive_controls"):
        from support.interactive_demo import execute as interactive_execute

        await interactive_execute(args, config, SimpleNamespace(**globals()))
        return
    if args.action == "recover" and not recovery_allowed():
        print("Automatic recovery has ended; no cloud operations performed.", flush=True)
        return
    routes = ProjectRoutes(config["project_id"], config["team_id"], access_token(), config["route_name"])
    await asyncio.to_thread(routes.check_project, config["public_origin"])
    baseline = await asyncio.to_thread(routes.live)
    previous = await current_sandbox(config, baseline["origin"])
    evidence = {"action": args.action, "previous": previous, "public_origin": config["public_origin"]}
    if args.action == "recover" and previous.get("status") != "stopped":
        status = previous.get("status")
        if status not in {"running", "pending", "stopping", "snapshotting"}:
            raise RuntimeError("The published Sandbox is not in a recoverable stopped state; inspect Vercel")
        evidence["skipped"] = f"Published Sandbox is {status}; existing processes were preserved"
        save(args.work_dir / "result.json", evidence)
        await asyncio.to_thread(append_summary, evidence["skipped"] + "\n")
        print(evidence["skipped"], flush=True)
        return
    evidence["baseline"] = await asyncio.to_thread(record_baseline, baseline["origin"])
    save(args.work_dir / "result.json", evidence)
    if evidence["baseline"]["health"].get("storage_backend") not in (None, "local"):
        raise RuntimeError("The published runtime uses another storage profile; it was preserved")
    state_path = args.work_dir / "sandbox-state.json"
    demo_path = args.work_dir / "demo-state.json"
    start_args = SimpleNamespace(output=demo_path, public_origin=config["public_origin"])
    candidate = None
    try:
        if args.action == "deploy":
            revision = await asyncio.to_thread(source_revision, args.source_repo)
            revision, payload, count = await asyncio.to_thread(
                PREPARE.committed_source, args.source_repo, args.source_branch, revision,
            )
            print(f"Source branch: {args.source_branch}; commit: {revision}; files: {count}", flush=True)
            prepare_args = SimpleNamespace(
                state=state_path, project_id=config["project_id"], team_id=config["team_id"], branch=args.source_branch,
            )
            await PREPARE.prepare(prepare_args, revision, payload)
            candidate = json.loads(state_path.read_text())
        else:
            candidate = previous
            save(state_path, candidate)
        if args.action == "recover" and not recovery_allowed():
            print("Automatic recovery ended during inspection; no session was resumed.", flush=True)
            return
        await START.start(start_args, candidate)
        demo = json.loads(demo_path.read_text())
        evidence["current"] = {**candidate, "demo_url": demo["demo_url"]}

        async def check():
            last_error = None
            for attempt in range(8):
                try:
                    return await asyncio.to_thread(
                        verify, config["public_origin"], candidate if args.action == "deploy" else None,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    last_error = exc
                    if attempt < 7:
                        await asyncio.sleep(3)
            raise RuntimeError("The fixed demo URL did not pass readiness and asset checks") from last_error

        await asyncio.to_thread(verify, demo["demo_url"], candidate if args.action == "deploy" else None)
        if args.action == "deploy" or demo["demo_url"] != baseline["origin"]:
            evidence["verification"] = await publish_checked(routes, baseline, demo, check)
        else:
            evidence["verification"] = await check()
        evidence["ready"] = True
        save(args.work_dir / "result.json", evidence)
    except BaseException:
        # Never stop a candidate while it might be the published target.
        if args.action == "deploy" and state_path.is_file():
            created = json.loads(state_path.read_text())
            if created.get("sandbox_name") and created["sandbox_name"] != previous["sandbox_name"]:
                try:
                    current = await asyncio.to_thread(routes.live)
                    if current["origin"] == baseline["origin"]:
                        await stop_sandbox(created)
                    else:
                        print("Cleanup skipped: inspect CDN routing and the candidate Sandbox in Vercel.", flush=True)
                except Exception:  # noqa: BLE001 - Preserve an uncertain published target during cleanup.
                    print("Cleanup could not be confirmed. Inspect routing and Sandbox status in Vercel.", flush=True)
        raise

    if args.action == "deploy":
        try:
            await stop_sandbox(previous)
            evidence["previous_stopped"] = True
        except Exception:  # noqa: BLE001 - Report cleanup failure without invalidating the healthy new demo.
            evidence["previous_stopped"] = False
            print("::warning::The new demo is ready, but stopping the previous Sandbox was not confirmed.")
    save(args.work_dir / "result.json", evidence)
    summary = (
        f"## {args.action.title()} Demo\n\n"
        f"Ready: {config['public_origin']}\n\n"
        f"Sandbox: `{candidate['sandbox_name']}`\n\nCommit: `{candidate['commit']}`\n\n"
        f"Catalog records: {evidence['verification']['catalog_total']}\n\n"
        "Session limit: 45 minutes. Use Start Demo after the session stops.\n"
    )
    print(summary, flush=True)
    await asyncio.to_thread(append_summary, summary)


def load_configuration(path: Path) -> dict:
    config = json.loads(path.read_text())
    if any(key in config for key in ("project_id", "team_id", "token", "public_origin")):
        raise RuntimeError("Use private deploy/.env or Actions Secrets/Variables for account-specific deployment settings")
    config.update(destination())
    origin = os.environ.get("DEMO_PUBLIC_ORIGIN", "").strip()
    if not origin:
        raise RuntimeError("Set DEMO_PUBLIC_ORIGIN as an Actions repository Variable or in private deploy/.env")
    config["public_origin"] = START.https_origin(origin)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["deploy", "start", "recover"])
    parser.add_argument("--config", type=Path, default=DEPLOY / "github_demo.json")
    parser.add_argument("--work-dir", type=Path, default=DEPLOY.parent / ".tmp/github-demo")
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--source-branch", choices=["main", "feat/deploy"], default="feat/deploy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--initialize-project", action="store_true", help="Replace a new project's placeholder frontend on the first deployment")
    args = parser.parse_args()
    if args.action == "recover" and not recovery_allowed():
        print("Automatic recovery has ended; no configuration or credentials read.")
        return
    config = load_configuration(args.config)
    if not config["project_id"].startswith("prj_") or not config["team_id"].startswith("team_"):
        raise RuntimeError("Configure the Vercel project and team IDs")
    if args.action == "deploy" and args.source_repo is None:
        parser.error("Deploy requires --source-repo pointing to the selected source checkout")
    args.work_dir = args.work_dir.resolve()
    if args.action == "deploy" and (args.work_dir / "sandbox-state.json").exists():
        raise RuntimeError("Use a fresh work directory for each new deployment")
    if args.dry_run:
        if args.action == "deploy":
            revision = source_revision(args.source_repo)
            commit, _payload, count = PREPARE.committed_source(args.source_repo, args.source_branch, revision)
            print(f"Source branch: {args.source_branch}; commit: {commit}; files: {count}")
        print(json.dumps(without_destination(config), indent=2))
        print("Dry run passed. No Token requested or cloud operations performed.")
        return
    asyncio.run(execute(args, config))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Check the workflow artifacts and Vercel before retrying.")
    except Exception as exc:  # noqa: BLE001 - Redact credential-bearing SDK errors at the CLI boundary.
        print(f"Failed: {type(exc).__name__}", flush=True)
        if isinstance(exc, RuntimeError):
            print(str(exc), flush=True)
        raise SystemExit(1)
