"""Small server-only controller for one published, prepared demo Sandbox.

GET and Stop never acquire a new session. Only an explicit Start can resume it.
No dependencies from the model reconstruction application are imported here.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

CONTROL_END = datetime(2026, 12, 17, 5, tzinfo=UTC)
CONFIG_PATH = Path(__file__).with_name("deployment.json")


class ControlError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def start_allowed() -> bool:
    return datetime.now(UTC) < CONTROL_END


def configuration() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = {"sandbox_name", "sandbox_origin", "commit", "app_dir", "public_origin"}
    if set(config) != required or not re.fullmatch(r"[a-f0-9]{40}", config["commit"]):
        raise ControlError(503, "The demo deployment configuration is invalid.")
    if not re.fullmatch(r"https://sb-[a-z0-9-]+\.vercel\.run", config["sandbox_origin"]):
        raise ControlError(503, "The demo target is invalid.")
    return config


def validate_request(method: str, headers: dict, body: bytes) -> str:
    if method == "GET":
        return "status"
    headers = {key.lower(): value for key, value in headers.items()}
    origin = headers.get("origin")
    # Non-browser clients may not omit Origin to bypass the same-origin check.
    if origin != configuration()["public_origin"] or headers.get("sec-fetch-site") not in (None, "same-origin"):
        raise ControlError(403, "Open the public demo page to use its controls.")
    if headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise ControlError(415, "Use a JSON demo control request.")
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        raise ControlError(400, "Invalid demo control request.") from None
    if not isinstance(value, dict) or set(value) != {"action"} or value["action"] not in {"start", "stop"}:
        raise ControlError(400, "Choose Start Demo or Stop Demo.")
    return value["action"]


def healthy(config: dict) -> bool:
    try:
        request = Request(config["sandbox_origin"] + "/health", headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=4) as response:
            health = json.load(response)
        return (
            health.get("storage_backend") == "local"
            and health.get("analysis_available") is True
            and all(health.get("checks", {}).get(key) is True for key in ["database", "storage", "worker"])
        )
    except (HTTPError, URLError, TimeoutError, ValueError):
        return False


async def passive_session(box):
    if box.current_session is not None:
        return box.current_session
    from vercel import sandbox

    async for runtime in sandbox.query_sessions(name=box.name):
        if runtime.id == box.current_session_id:
            return runtime
    raise ControlError(503, "Cannot inspect the current demo session. Nothing was restarted.")


async def remote(runtime, config: dict, action: str) -> dict:
    app = PurePosixPath(config["app_dir"])
    argv = ["-B", str(app / ".tmp/vercel-demo/control.py"), action, runtime.id]
    if action == "reserve-stop":
        argv.append(uuid4().hex)
    result = await runtime.run_process(str(app / ".venv/bin/python"), argv, capture_output=True, check=True)
    value = json.loads(result.stdout)
    if value.get("state") == "unavailable":
        raise ControlError(503, "Cannot verify the demo database. Nothing was stopped.")
    return value


def response(state: str, *, ready: bool = False, queued: int = 0, running: int = 0, idle_checked: bool = False) -> dict:
    return {
        "state": state, "ready": ready, "queued": queued, "running": running,
        "start_allowed": start_allowed() and state in {"stopped", "starting", "running"},
        "stop_allowed": (state == "running" or (state == "starting" and idle_checked)) and queued == 0 and running == 0,
        "session_limit_minutes": 45,
    }


async def operate(box, config: dict, action: str) -> tuple[int, dict]:
    status = str(box.status)
    if action == "start" and not start_allowed():
        raise ControlError(403, "The demo start period ended on December 16, 2026 in New York.")
    if status in {"pending", "stopping", "snapshotting"}:
        return 202, response("starting" if status == "pending" else "stopping")
    if status == "stopped" and action != "start":
        return 200, response("stopped")
    if status not in {"running", "stopped"}:
        raise ControlError(503, "The demo needs operator attention. No replacement environment was created.")

    runtime = await box.session() if action == "start" and status == "stopped" else await passive_session(box)
    if action == "start":
        state = await remote(runtime, config, "start")
        return 202, response(state["state"])
    if action == "stop":
        reservation = await remote(runtime, config, "reserve-stop")
        if reservation["state"] == "busy":
            value = response("running", **{key: reservation[key] for key in ["queued", "running"]})
            value["error"] = {"message": "Stop refused: another visitor has a queued or running analysis."}
            return 409, value
        if reservation["state"] != "reserved":
            raise ControlError(409, "Another demo control operation is in progress. Try again shortly.")
        # Pin the stop to the inspected session: a late request must never stop
        # a newer session started by another visitor.
        await runtime.stop()
        return 202, response("stopping")
    state = await remote(runtime, config, "status")
    if state["state"] == "stopping":
        return 200, response("stopping")
    ready = await asyncio.to_thread(healthy, config)
    return 200, response(
        "running" if ready else "starting", ready=ready, queued=state["queued"], running=state["running"], idle_checked=True,
    )


async def dispatch(action: str) -> tuple[int, dict]:
    from vercel import sandbox
    from vercel.api import session
    from vercel.sandbox import SandboxCredentials, SandboxServiceOptions

    if action == "start" and not start_allowed():
        raise ControlError(403, "The demo start period has ended.")
    config = configuration()

    if action != "status":
        await asyncio.to_thread(require_current_deployment, config)

    async def credentials():
        values = [os.environ.get("DEMO_VERCEL_" + key) for key in ["TOKEN", "PROJECT_ID", "TEAM_ID"]]
        if not all(values):
            raise ControlError(503, "The demo controls are not configured.")
        return SandboxCredentials(token=values[0], project_id=values[1], team_id=values[2])

    async with asyncio.timeout(240), session(service_options=[SandboxServiceOptions(credentials_factory=credentials)]):
        box = await sandbox.get_sandbox(name=config["sandbox_name"])
        tags = box.tags or {}
        if tags.get("app") != "model-anatomy" or tags.get("purpose") != "demo" or tags.get("commit") != config["commit"]:
            raise ControlError(503, "The prepared demo identity could not be verified.")
        if not any(route.port == 8080 and not route.system and route.url.rstrip("/") == config["sandbox_origin"] for route in box.routes):
            raise ControlError(503, "The prepared demo route could not be verified.")
        return await operate(box, config, action)


def require_current_deployment(config: dict) -> None:
    """A superseded deployment URL must not start an old, unpublished demo."""
    token = os.environ.get("DEMO_VERCEL_TOKEN", "")
    project_id = os.environ.get("DEMO_VERCEL_PROJECT_ID", "")
    team_id = os.environ.get("DEMO_VERCEL_TEAM_ID", "")
    if not all([token, project_id, team_id]):
        raise ControlError(503, "The demo controls are not configured.")

    def read(path: str) -> dict:
        request = Request(
            "https://api.vercel.com" + path + "?" + urlencode({"teamId": team_id}),
            headers={"Authorization": "Bearer " + token},
        )
        with urlopen(request, timeout=8) as response:
            return json.load(response)

    project = read("/v9/projects/" + quote(project_id, safe=""))
    if project.get("id") != project_id or project.get("accountId") != team_id:
        raise ControlError(503, "The demo project could not be verified.")
    target = (project.get("targets") or {}).get("production") or {}
    if not target.get("id"):
        raise ControlError(503, "The published demo could not be verified.")
    deployment = read("/v13/deployments/" + quote(target["id"], safe=""))
    meta = deployment.get("meta", {})
    if (
        deployment.get("projectId") != project_id
        or meta.get("modelAnatomyProtocol") != "visitor-controls-v1"
        or meta.get("modelAnatomySandbox") != config["sandbox_name"]
        or meta.get("modelAnatomySourceCommit") != config["commit"]
        or meta.get("modelAnatomySandboxOrigin") != config["sandbox_origin"]
    ):
        raise ControlError(409, "This demo version has been replaced. Open the current public demo page.")
