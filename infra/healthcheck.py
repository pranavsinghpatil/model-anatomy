"""Quiet container probes. A worker process probe is not a progress guarantee."""

import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def check_api() -> None:
    request = Request("http://127.0.0.1:8000/health", headers={"Host": "localhost"})
    with urlopen(request, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("API not ready")


def check_worker() -> None:
    # Compose's init process is PID 1; inspect its worker child without ps/procps.
    commands = (path.read_bytes().split(b"\0") for path in Path("/proc").glob("[0-9]*/cmdline"))
    if not any(b"model_anatomy.service.worker" in command for command in commands):
        raise RuntimeError("Worker process missing")
    import psycopg

    url = os.environ["MODEL_ANATOMY_DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(url, connect_timeout=5, options="-c default_transaction_read_only=on") as connection:
        row = connection.execute(
            "SELECT worker_owner, worker_until FROM public.service_mutex WHERE id = 1"
        ).fetchone()
        if row is None or not row[0] or row[1] is None or row[1] <= time.time():
            raise RuntimeError("Worker lease is stale")


def main() -> int:
    try:
        {"api": check_api, "worker": check_worker}[sys.argv[1]]()
    except Exception:  # noqa: BLE001 - this boundary must suppress credential-bearing errors
        # Exception text can contain connection URLs and must never reach logs.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
