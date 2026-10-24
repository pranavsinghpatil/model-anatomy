"""Start the local API, worker and Vite on Linux; never install or inspect models."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def frontend_configuration() -> tuple[int, str]:
    from dotenv import dotenv_values

    values: dict[str, str | None] = {}
    for name in (".env", ".env.local", ".env.development", ".env.development.local"):
        path = ROOT / "frontend" / name
        if path.is_file():
            values.update(dotenv_values(path))
    values.update(os.environ)
    raw = values.get("FRONTEND_PORT") or "5173"
    if not re.fullmatch(r"[0-9]+", raw) or not 1024 <= int(raw) <= 65535:
        raise ValueError("FRONTEND_PORT must be an integer from 1024 to 65535.")
    return int(raw), values.get("API_PROXY_TARGET") or "http://127.0.0.1:8001"


def require_free_port(port: int) -> None:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is already in use. Existing processes were not stopped.") from None


def require_frontend() -> str:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is missing. Install Node 22.12 or later before starting the frontend.")
    result = subprocess.run([node, "--version"], capture_output=True, text=True, check=True, timeout=5)
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)\s*", result.stdout)
    version = tuple(int(part) for part in match.groups()) if match else (0, 0, 0)
    if not (version >= (22, 12, 0) or (20, 19, 0) <= version < (21, 0, 0)):
        raise RuntimeError("The locked Vite requires Node ^20.19.0 or >=22.12.0.")
    if not (ROOT / "frontend/node_modules/vite/bin/vite.js").is_file():
        raise RuntimeError(
            "Frontend dependencies are missing. From frontend/, run: "
            "npm ci. Nothing was installed automatically."
        )
    return node


def require_local_database(settings) -> None:
    from sqlalchemy.engine import make_url

    if settings.environment != "local" or settings.storage_backend != "local":
        raise RuntimeError("This launcher is for local SQLite/storage. Use the deployment setup for production.")
    url = make_url(settings.resolved_database_url)
    if url.drivername != "sqlite" or url.database in {None, ":memory:"}:
        raise RuntimeError("This local launcher requires a durable SQLite file.")
    database = Path(url.database).resolve()
    if not database.is_file():
        return
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='service_mutex'").fetchone():
            return
        row = connection.execute("SELECT worker_owner, worker_until FROM service_mutex WHERE id=1").fetchone()
        if row and row[0] and row[1] and row[1] > time.time():
            raise RuntimeError("A worker already owns this database. Stop that worker before using this launcher.")


def wait_ready(url: str, processes: dict[str, subprocess.Popen], timeout: float, *, api: bool) -> None:
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        for name, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited during startup. Check this launch's log directory.")
        try:
            with opener.open(url, timeout=1) as response:
                if not api:
                    return
                body = response.read(65537)
                if len(body) <= 65536 and json.loads(body).get("analysis_available") is True:
                    return
        except (URLError, OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Local services did not become ready before the startup deadline. Check the logs.")


def stop_owned(processes: dict[str, subprocess.Popen]) -> None:
    for process in reversed(list(processes.values())):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in reversed(list(processes.values())):
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check configuration and dependencies; do not start services")
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("Use scripts/start-local.ps1 on Windows; this launcher requires Linux process groups.")

    from model_anatomy.dependencies import require_dependencies
    require_dependencies("api", "engine", component="Local application")
    from model_anatomy.api.settings import Settings

    os.chdir(ROOT)
    processes: dict[str, subprocess.Popen] = {}
    log_directory = None
    try:
        settings = Settings()
        port, target = frontend_configuration()
        proxy = urlsplit(target)
        if (
            proxy.scheme != "http"
            or proxy.hostname not in {"127.0.0.1", "localhost"}
            or proxy.port != settings.port
            or proxy.path not in {"", "/"}
            or proxy.query
            or proxy.fragment
            or proxy.username
            or proxy.password
        ):
            raise RuntimeError("Set frontend/.env API_PROXY_TARGET to the local backend URL and port from backend/.env.")
        if port == settings.port:
            raise RuntimeError("The frontend and API must use different ports.")
        node = require_frontend()
        require_free_port(settings.port)
        require_free_port(port)
        require_local_database(settings)
        if args.check_only:
            print(f"Preflight passed: API {settings.port}, frontend {port}; packages available; no models inspected.")
            return 0
        log_root = settings.data_dir / "local-launcher"
        log_root.mkdir(parents=True, exist_ok=True)
        log_directory = Path(tempfile.mkdtemp(prefix="run-", dir=log_root))
        print(f"Logs: {log_directory}", flush=True)
        with ExitStack() as stack:
            def start(name: str, command: list[str], cwd: Path) -> None:
                log = stack.enter_context((log_directory / f"{name}.log").open("wb"))
                processes[name] = subprocess.Popen(
                    command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )

            try:
                start("api", [sys.executable, "-m", "model_anatomy.api"], ROOT)
                start("worker", [sys.executable, "-m", "model_anatomy.service.worker"], ROOT)
                wait_ready(f"http://127.0.0.1:{settings.port}/health", processes, 30, api=True)
                start("frontend", [node, str(ROOT / "frontend/node_modules/vite/bin/vite.js"),
                                   "--host", "127.0.0.1", "--clearScreen", "false"],
                      ROOT / "frontend")
                wait_ready(f"http://127.0.0.1:{port}", processes, 30, api=False)
                print(f"Application: http://127.0.0.1:{port}\nAPI docs: http://127.0.0.1:{settings.port}/docs", flush=True)
                print("Keep this terminal open. Ctrl+C stops only this launch's processes.", flush=True)
                while True:
                    for name, process in processes.items():
                        if process.poll() is not None:
                            raise RuntimeError(f"{name} stopped. Check the log directory.")
                    time.sleep(0.5)
            finally:
                stop_owned(processes)
    except KeyboardInterrupt:
        print("Stopped this launch's processes.")
        return 0
    except RuntimeError as exc:
        # RuntimeError messages above are fixed diagnostics without credentials.
        print(str(exc), file=sys.stderr)
        if log_directory:
            print(f"Logs: {log_directory}", file=sys.stderr)
        return 1
    except (ValueError, OSError, subprocess.SubprocessError):
        # Settings and provider exceptions may contain private values.
        print("Configuration or startup check failed. Check backend/.env and frontend/.env.", file=sys.stderr)
        if log_directory:
            print(f"Logs: {log_directory}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
