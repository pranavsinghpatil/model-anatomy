"""Session-bound demo controls, executed inside the prepared Linux Sandbox.

Stop reservations hold SQLite's writer lock until the VM stops. JobService uses
BEGIN IMMEDIATE too, so a submission cannot be accepted between the global
idle check and the stop request. No application database schema is changed.
"""

from __future__ import annotations

import fcntl
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

STOP_LEASE_SECONDS = 120


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def configuration(control: Path) -> dict:
    config = read_json(control / "runtime.json")
    expected = str(control / "data")
    if config.get("profile") != "sandbox-sqlite-demo" or config.get("data_dir") != expected:
        raise RuntimeError("The runtime is not the prepared SQLite demo")
    environment = config.get("environment", {})
    if environment.get("MODEL_ANATOMY_STORAGE_BACKEND") != "local":
        raise RuntimeError("Another storage profile was preserved")
    if environment.get("MODEL_ANATOMY_DATABASE_URL") != "sqlite:///" + expected + "/jobs.sqlite3":
        raise RuntimeError("The demo database path does not match the runtime profile")
    return config


def database(control: Path) -> sqlite3.Connection:
    configuration(control)
    # mode=rw fails closed when the durable database is missing.
    return sqlite3.connect((control / "data/jobs.sqlite3").as_uri() + "?mode=rw", uri=True, timeout=3)


def counts(connection: sqlite3.Connection) -> dict:
    rows = connection.execute(
        "SELECT status, COUNT(*) FROM analysis_jobs WHERE status IN ('queued', 'running') GROUP BY status"
    ).fetchall()
    return {"queued": dict(rows).get("queued", 0), "running": dict(rows).get("running", 0)}


def stopping(control: Path, session_id: str) -> bool:
    state = read_json(control / "stop-reservation.json")
    # The admission fence lasts for the entire session, even if a stop request
    # loses its response. A new session has another identity and ignores it.
    return state.get("session_id") == session_id


def status(control: Path, session_id: str) -> dict:
    configuration(control)
    if stopping(control, session_id):
        return {"state": "stopping", "queued": 0, "running": 0}
    with database(control) as connection:
        return {"state": "running", **counts(connection)}


def start(control: Path, session_id: str) -> dict:
    config = configuration(control)
    with (control / "operation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "stopping" if stopping(control, session_id) else "starting"}
        if stopping(control, session_id):
            return {"state": "stopping"}
        config["environment"].update({
            "MODEL_ANATOMY_DEMO_CONTROL_FILE": str(control / "stop-reservation.json"),
            "MODEL_ANATOMY_DEMO_SESSION_ID": session_id,
        })
        write_json(control / "runtime.json", config)
        # The supervisor owns its own lifetime lock; repeated clicks are safe.
        subprocess.Popen(
            [config["python"], "-B", str(control / "supervisor.py"), str(control / "runtime.json")],
            cwd=config["app_dir"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    return {"state": "starting"}


def hold_stop(control: Path, session_id: str, token: str, *, lease_seconds: float = STOP_LEASE_SECONDS) -> None:
    result = control / f"stop-{token}.json"
    with (control / "operation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write_json(result, {"state": "conflict"})
            return
        connection = None
        try:
            connection = database(control)
            connection.execute("BEGIN IMMEDIATE")
            active = counts(connection)
            if active["queued"] or active["running"]:
                write_json(result, {"state": "busy", **active})
                return
            reservation = {"session_id": session_id, "token": token, "expires_at": time.time() + lease_seconds}
            write_json(control / "stop-reservation.json", reservation)
            write_json(result, {"state": "reserved", **active})
            # Keep both locks across the provider stop call. A lost request only
            # blocks admission temporarily; it never force-stops the application.
            deadline = time.monotonic() + lease_seconds
            while time.monotonic() < deadline:
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except (OSError, ValueError, sqlite3.Error, RuntimeError):
            write_json(result, {"state": "unavailable"})
        finally:
            if connection is not None:
                connection.rollback()
                connection.close()
            # Keep the per-session admission fence if the network request was
            # lost. Resuming a later session changes the service identity.


def reserve_stop(control: Path, session_id: str, token: str) -> dict:
    from model_anatomy.service.jobs import DEMO_CONTROL_PROTOCOL

    config = configuration(control)
    environment = config["environment"]
    if (
        DEMO_CONTROL_PROTOCOL != "visitor-controls-v1"
        or environment.get("MODEL_ANATOMY_DEMO_SESSION_ID") != session_id
        or environment.get("MODEL_ANATOMY_DEMO_CONTROL_FILE") != str(control / "stop-reservation.json")
    ):
        raise RuntimeError("The runtime does not implement the expected demo admission protection")
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        raise ValueError("Invalid stop operation identifier")
    result = control / f"stop-{token}.json"
    # Bound tiny diagnostic files left behind by interrupted Function requests.
    for stale in control.glob("stop-*.json"):
        if re.fullmatch(r"stop-[a-f0-9]{32}\.json", stale.name) and stale.stat().st_mtime < time.time() - 300:
            stale.unlink(missing_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-B", str(Path(__file__).resolve()), "hold-stop", session_id, token],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if result.is_file():
            value = read_json(result)
            result.unlink(missing_ok=True)
            return value
        if process.poll() is not None:
            return {"state": "unavailable"}
        time.sleep(0.05)
    return {"state": "unavailable"}


def main() -> None:
    control = Path(__file__).resolve().parent
    action, session_id = sys.argv[1:3]
    if action == "hold-stop":
        hold_stop(control, session_id, sys.argv[3])
        return
    operations = {"status": status, "start": start}
    result = (
        reserve_stop(control, session_id, sys.argv[3])
        if action == "reserve-stop" else operations[action](control, session_id)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - Never expose filesystem or database internals to visitors.
        print(json.dumps({"state": "unavailable"}), flush=True)
        raise SystemExit(1)
