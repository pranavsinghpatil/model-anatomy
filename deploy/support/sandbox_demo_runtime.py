"""Remote supervisor uploaded by deploy/03_start_sandbox_demo.py.

This runs inside the Linux Sandbox, not on the operator's computer.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    control = config_path.parent
    lock = (control / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("The demo supervisor is already running", flush=True)
        return

    environment = os.environ.copy()
    environment.update(config["environment"])
    for name in config["directories"]:
        Path(name).mkdir(parents=True, exist_ok=True)

    # Create the SQLite schema once before API and worker startup can overlap.
    subprocess.run(
        [config["python"], "-B", "-c", (
            "from model_anatomy.api.settings import Settings; "
            "from model_anatomy.service.jobs import JobService; "
            "settings = Settings(); "
            "assert settings.environment == 'local'; "
            "assert settings.storage_backend == 'local'; "
            "assert settings.resolved_database_url.startswith('sqlite:'); "
            "service = JobService(settings); service.initialize(); "
            "service.db.engine.dispose(); print('SQLite and local storage initialized')"
        )], cwd=config["app_dir"], env=environment, check=True,
    )

    stopping = False

    def request_stop(*_args) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    children: list[tuple[str, subprocess.Popen]] = []
    logs = []
    status_path = control / "processes.json"
    try:
        for name, command in config["services"].items():
            log = (control / f"{name}.log").open("ab", buffering=0)
            logs.append(log)
            process = subprocess.Popen(
                command, cwd=config["app_dir"], env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            )
            children.append((name, process))
            print(f"Started {name}, pid={process.pid}", flush=True)
        status_path.write_text(json.dumps({
            "profile": "sandbox-sqlite-demo", "origin": config["origin"],
            "data_dir": config["data_dir"],
            "pids": {name: child.pid for name, child in children},
            "running": True,
        }, indent=2) + "\n")

        while not stopping:
            exited = [(name, child.returncode) for name, child in children if child.poll() is not None]
            if exited:
                print(f"Service exited: {exited}", flush=True)
                break
            time.sleep(0.5)
    finally:
        for _name, child in children:
            if child.poll() is None:
                child.terminate()
        for _name, child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in logs:
            log.close()
        status_path.write_text(json.dumps({
            "profile": "sandbox-sqlite-demo", "origin": config["origin"],
            "data_dir": config["data_dir"], "running": False,
            "returncodes": {name: child.returncode for name, child in children},
        }, indent=2) + "\n")
        lock.close()


if __name__ == "__main__":
    with (Path(sys.argv[1]).parent / "supervisor.log").open("ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        run(Path(sys.argv[1]))
