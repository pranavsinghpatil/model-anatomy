"""Upload committed frontend files from feat/deploy to the existing Vercel Project.

Uses the Vercel REST API with an explicitly prompted Token. No Vercel CLI,
GitHub connection, GitHub Token, or Git author metadata is used.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from support.credentials import access_token, bind_destination, without_destination

DEPLOY_DIR = Path(__file__).resolve().parent
BRANCH = "feat/deploy"
PROJECT_SETTINGS = {
    "rootDirectory": "frontend",
    "framework": "vite",
    "installCommand": "npm ci",
    "buildCommand": "npm run build",
    "outputDirectory": "dist",
    "devCommand": None,
    "nodeVersion": "22.x",
}


def frontend_files(repo: Path) -> tuple[str, list[dict[str, str]], int]:
    """Read the named branch's committed files, independently of the working tree."""
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/heads/{BRANCH}^{{commit}}"],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit, "frontend"],
        cwd=repo, check=True, capture_output=True,
    ).stdout
    files = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts or path.parts[0] != "frontend":
                raise RuntimeError("The frontend archive contains an unsupported path or file type")
            if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
                raise RuntimeError("A private environment file is tracked in frontend; deployment stopped")
            stream = source.extractfile(member)
            if stream is None:
                raise RuntimeError("A committed frontend file could not be read")
            with stream:
                content = stream.read()
            total += len(content)
            files.append({
                "file": path.as_posix(),
                "data": base64.b64encode(content).decode("ascii"),
                "encoding": "base64",
            })
    paths = {item["file"] for item in files}
    if not {"frontend/package.json", "frontend/package-lock.json", "frontend/index.html"} <= paths:
        raise RuntimeError("The branch does not contain the required Vite frontend files")
    return commit, files, total


def deployment_payload(project_id: str, project_name: str, commit: str, files: list[dict[str, str]]) -> dict:
    return {
        "name": project_name,
        "project": project_id,
        "target": "production",
        "projectSettings": dict(PROJECT_SETTINGS),
        "files": files,
        "meta": {"modelAnatomySourceBranch": BRANCH, "modelAnatomySourceCommit": commit},
    }


def check_project(project: dict, project_id: str, team_id: str) -> None:
    if project.get("id") != project_id or project.get("accountId") != team_id:
        raise RuntimeError("The returned Project does not match the selected Project and Team")
    if project.get("link"):
        raise RuntimeError("This upload workflow requires an unlinked Project. Check Settings > Git first")
    production = (project.get("targets") or {}).get("production") or {}
    if production and production.get("readyState") not in {"ERROR", "CANCELED"}:
        raise RuntimeError(
            "A Production deployment already exists or is building; it was preserved. "
            "Check Deployments in the dashboard before creating another deployment"
        )


def api(method: str, path: str, token: str, team_id: str, body: dict | None = None) -> dict:
    address = "https://api.vercel.com" + path + "?" + urlencode({"teamId": team_id})
    data = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
    request = Request(
        address, data=data, method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        code = "request_failed"
        message = "Check the Project, Team, Token scope, and Vercel dashboard"
        try:
            error = json.loads(exc.read()).get("error") or {}
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"Vercel HTTP {exc.code} ({code}): {message}".replace(token, "[redacted]")) from None
    except URLError:
        raise RuntimeError("Cannot reach api.vercel.com. Check the network and dashboard before retrying") from None


def create_entrypoint(args: argparse.Namespace, state: dict, commit: str, files: list[dict[str, str]]) -> None:
    token = access_token()
    project_id = state["project_id"]
    team_id = state["team_id"]
    project = api("GET", "/v9/projects/" + quote(project_id, safe=""), token, team_id)
    check_project(project, project_id, team_id)
    print(f"Project: {project['name']}", flush=True)
    print(f"Uploading {len(files)} committed frontend files from {BRANCH} at {commit}", flush=True)
    created = api(
        "POST", "/v13/deployments", token, team_id,
        deployment_payload(project_id, project["name"], commit, files),
    )
    deployment_id = created["id"]
    result = {
        "project_id": project_id, "team_id": team_id, "branch": BRANCH, "commit": commit,
        "deployment_id": deployment_id, "ready": False,
    }

    def save(detail: dict) -> None:
        result.update({key: detail.get(key) for key in ["readyState", "url", "alias", "target"]})
        result["ready"] = detail.get("readyState") == "READY"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(without_destination(result), indent=2) + "\n", encoding="utf-8")

    save(created)
    print(f"Created deployment: {deployment_id}", flush=True)
    print(f"Local state: {args.output}", flush=True)
    deadline = time.monotonic() + args.timeout
    prior = None
    heartbeat = 0.0
    detail = created
    while True:
        status = detail.get("readyState")
        if status != prior or time.monotonic() >= heartbeat:
            print(f"Deployment state: {status}", flush=True)
            prior = status
            heartbeat = time.monotonic() + 30
        save(detail)
        if status == "READY":
            break
        if status in {"ERROR", "CANCELED"}:
            raise RuntimeError("The deployment did not become ready. Open its build logs in the Vercel dashboard")
        if time.monotonic() >= deadline:
            raise RuntimeError("Polling timed out. The remote build may still be running; check the dashboard")
        time.sleep(5)
        detail = api("GET", "/v13/deployments/" + quote(deployment_id, safe=""), token, team_id)
    print(f"Production Deployment ready: https://{detail['url']}", flush=True)
    for alias in detail.get("alias") or []:
        print(f"Production domain: https://{alias}", flush=True)
    print("Choose the readable domain in Settings > Domains, then configure the Sandbox rewrite.", flush=True)
    print("Frontend readiness does not establish API or worker readiness through this domain.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEPLOY_DIR.parent)
    parser.add_argument("--state", type=Path, default=DEPLOY_DIR / "sandbox-state.json")
    parser.add_argument("--output", type=Path, default=DEPLOY_DIR / "production-state.json")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    state = bind_destination(json.loads(args.state.read_text(encoding="utf-8")))
    commit, files, total = frontend_files(args.repo)
    if args.dry_run:
        payload = deployment_payload(state["project_id"], "resolved-from-existing-project", commit, files)
        print(json.dumps({
            "destination_configured": True, "branch": BRANCH,
            "commit": commit, "file_count": len(files), "source_bytes": total,
            "request_bytes": len(json.dumps(payload, separators=(",", ":")).encode()),
            "project_settings": PROJECT_SETTINGS, "git_source": False, "git_author_metadata": False,
            "target": "production", "output": str(args.output),
        }, indent=2))
        print("Dry run OK. No Token requested, files written, or Vercel calls made.")
        return
    create_entrypoint(args, state, commit, files)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Check production-state.json and the dashboard before retrying.") from None
    except subprocess.CalledProcessError:
        raise SystemExit("Cannot read the committed feat/deploy frontend. Check --repo and the local branch.") from None
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
