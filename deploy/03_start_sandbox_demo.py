"""Resume the prepared Sandbox and serve a real SQLite-backed Model Anatomy demo.

Credentials are prompted locally. No Vercel CLI, Supabase, or fresh Sandbox creation.
Run from the repository root; the companion supervisor lives in deploy/support.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import http.cookiejar
import json
import subprocess
import time
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPCookieProcessor, build_opener


DEPLOY_DIR = Path(__file__).resolve().parent
RUNTIME_SCRIPT = DEPLOY_DIR / "support" / "sandbox_demo_runtime.py"


PROBE = r'''
import json, socket, sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

def get(url):
    try:
        try:
            response = urlopen(url, timeout=2)
        except HTTPError as exc:
            response = exc
        with response:
            return json.load(response)
    except (URLError, TimeoutError, ValueError):
        return None

def listening(port):
    with socket.socket() as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(('127.0.0.1', port)) == 0

root = Path(sys.argv[1])
marker_path = root / '.tmp/vercel-demo/runtime.json'
marker = None
if marker_path.is_file():
    try:
        config = json.loads(marker_path.read_text())
        marker = {k: config.get(k) for k in ['profile', 'data_dir', 'origin', 'public_origin']}
    except ValueError:
        pass
health = get('http://127.0.0.1:8001/health')
catalog = get('http://127.0.0.1:8001/api/v1/catalog/discovery') if health else None
print(json.dumps({
    'api_listening': listening(8001), 'entry_listening': listening(8080),
    'health': health,
    'catalog_total': catalog.get('total') if isinstance(catalog, dict) else None,
    'profile': marker,
}))
'''


def configuration(app_dir: str, origin: str, public_origin: str | None = None) -> dict:
    app = PurePosixPath(app_dir)
    control = app / ".tmp/vercel-demo"
    data = control / "data"
    temporary = control / "tmp"
    python = str(app / ".venv/bin/python")
    database = "sqlite:///" + str(data / "jobs.sqlite3")
    directories = [data, temporary, control / "cache", control / "config", control / "huggingface"]
    origins = list(dict.fromkeys(value for value in [origin, public_origin] if value))
    hosts = list(dict.fromkeys(["127.0.0.1", "localhost", *[urlsplit(value).hostname for value in origins]]))
    environment = {
        "MODEL_ANATOMY_ENVIRONMENT": "local",
        "MODEL_ANATOMY_STORAGE_BACKEND": "local",
        "MODEL_ANATOMY_DATA_DIR": str(data),
        "MODEL_ANATOMY_DATABASE_URL": database,
        "DATABASE_URL": database,
        "MODEL_ANATOMY_PORT": "8001",
        "MODEL_ANATOMY_ALLOWED_HOSTS": json.dumps(hosts),
        "MODEL_ANATOMY_CORS_ORIGINS": json.dumps(origins),
        "MODEL_ANATOMY_DOCS_ENABLED": "false",
        "MODEL_ANATOMY_SUPABASE_URL": "",
        "MODEL_ANATOMY_SUPABASE_KEY": "",
        "MODEL_ANATOMY_SUPABASE_SERVICE_ROLE_KEY": "",
        "NEBIUS_API_KEY": "",
        "MODEL_ANATOMY_NEBIUS_API_KEY": "",
        "NEBIUS_MODEL": "",
        "MODEL_ANATOMY_NEBIUS_MODEL": "",
        "MODEL_ANATOMY_NEBIUS_MODEL_ID": "",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TMPDIR": str(temporary), "TMP": str(temporary), "TEMP": str(temporary),
        "HF_HOME": str(control / "huggingface"),
        "XDG_CACHE_HOME": str(control / "cache"),
        "XDG_CONFIG_HOME": str(control / "config"),
        "XDG_DATA_HOME": str(control / "caddy-data"),
    }
    return {
        "profile": "sandbox-sqlite-demo", "origin": origin, "public_origin": public_origin,
        "app_dir": app_dir,
        "data_dir": str(data), "python": python,
        "directories": [str(path) for path in directories], "environment": environment,
        "services": {
            "api": [python, "-B", "-m", "uvicorn", "model_anatomy.api.app:create_app", "--factory",
                    "--host", "127.0.0.1", "--port", "8001", "--no-access-log", "--no-proxy-headers",
                    "--limit-concurrency", "32"],
            "worker": [python, "-B", "-m", "model_anatomy.service.worker"],
            "caddy": ["caddy", "run", "--config", str(control / "Caddyfile"), "--adapter", "caddyfile"],
        },
    }


def https_origin(value: str) -> str:
    """Accept a browser origin, without credentials, paths, queries or fragments."""
    parsed = urlsplit(value)
    if (
        any(char.isspace() for char in value)
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError("Use an HTTPS origin such as https://your-demo.vercel.app")
    hostname = parsed.hostname.encode("idna").decode("ascii")
    return f"https://[{hostname}]" if ":" in hostname else f"https://{hostname}"


def caddyfile(app_dir: str) -> str:
    root = json.dumps(str(PurePosixPath(app_dir) / "frontend/dist"))
    return """{
    admin off
    auto_https off
}
:8080 {
    bind 0.0.0.0
    encode gzip
    @backend path /api /api/* /health
    handle @backend {
        reverse_proxy 127.0.0.1:8001 {
            header_up Host 127.0.0.1:8001
            header_up -X-Forwarded-For
            header_up -X-Forwarded-Host
            header_up -X-Forwarded-Proto
        }
    }
    handle /assets/* {
        root * ROOT_PLACEHOLDER
        header Cache-Control "public, max-age=31536000, immutable"
        file_server
    }
    handle {
        root * ROOT_PLACEHOLDER
        header Cache-Control "no-cache"
        try_files {path} /index.html
        file_server
    }
}
""".replace("ROOT_PLACEHOLDER", root)


def public_checks(origin: str) -> dict:
    cookies = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookies))

    def read(path: str, *, as_json: bool = True):
        with opener.open(origin + path, timeout=15) as response:
            return json.load(response) if as_json else response.read(1024 * 1024).decode()

    health = read("/health")
    if not health.get("analysis_available") or health.get("storage_backend") != "local":
        raise RuntimeError("The public URL does not have a ready SQLite analysis service")
    if not all(health.get("checks", {}).get(key) is True for key in ["database", "storage", "worker"]):
        raise RuntimeError("One of the database/storage/worker checks is unavailable")
    catalog = read("/api/v1/catalog/discovery")
    if not isinstance(catalog.get("total"), int):
        raise RuntimeError("The catalog endpoint did not return the expected JSON")
    analyses = read("/api/v1/analyses")
    if not isinstance(analyses.get("analyses"), list):
        raise RuntimeError("The visitor analyses endpoint did not return the expected JSON")
    if not any(cookie.name == "model_anatomy_visitor" for cookie in cookies):
        raise RuntimeError("The public API did not establish the visitor cookie")
    page = read("/", as_json=False)
    nested = read("/app", as_json=False)
    if 'id="root"' not in page or 'id="root"' not in nested:
        raise RuntimeError("The frontend root or SPA route is not being served")
    return {"health": health, "catalog_total": catalog["total"], "visitor_cookie": True, "frontend": True}


async def start(args: argparse.Namespace, prepared: dict) -> None:
    from vercel import sandbox
    from vercel.api import session
    from vercel.sandbox import SandboxCredentials, SandboxServiceOptions

    token = getpass.getpass("Vercel Access Token (hidden): ").strip()
    if not token:
        raise RuntimeError("An access token is required")

    async def credentials() -> SandboxCredentials:
        return SandboxCredentials(token=token, project_id=prepared["project_id"], team_id=prepared["team_id"])

    result = {key: prepared[key] for key in ["project_id", "team_id", "sandbox_name", "commit", "app_dir"]}
    result["ready"] = False
    result["public_origin"] = args.public_origin

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    async with session(service_options=[SandboxServiceOptions(credentials_factory=credentials)]):
        box = await sandbox.get_sandbox(name=prepared["sandbox_name"])
        print(f"Using existing Sandbox: {box.name}", flush=True)
        await box.update(execution_time_limit=timedelta(minutes=45), persistent=True)
        runtime = await box.session()  # Intentionally leaves the remote session running.
        result["session_id"] = runtime.id
        app_dir = prepared["app_dir"]
        control = PurePosixPath(app_dir) / ".tmp/vercel-demo"
        python = str(PurePosixPath(app_dir) / ".venv/bin/python")
        probe = await runtime.run_process(python, ["-B", "-c", PROBE, app_dir], capture_output=True, check=True)
        baseline = json.loads(probe.stdout)
        result["prior_runtime"] = baseline
        save()
        print("Recorded prior health, storage backend and catalog total", flush=True)

        occupied = baseline["api_listening"] or baseline["entry_listening"]
        expected_data = str(control / "data")
        marker = baseline.get("profile") or {}
        if occupied and (marker.get("profile") != "sandbox-sqlite-demo" or marker.get("data_dir") != expected_data):
            raise RuntimeError("Existing runtime has a different profile; it was not replaced")
        if (baseline.get("health") or {}).get("storage_backend") == "supabase":
            raise RuntimeError("Existing Supabase runtime was preserved; refusing to replace it")
        if occupied and marker.get("public_origin") != args.public_origin:
            raise RuntimeError(
                "The running demo has a different public origin and was preserved. "
                "Stop this Sandbox in the Vercel dashboard, then rerun Step 3 with the desired --public-origin."
            )

        await box.update(ports=[8080])
        origin = next((route.url.rstrip("/") for route in box.routes if route.port == 8080 and not route.system), None)
        if not origin:
            raise RuntimeError("No public route was returned for port 8080")
        result["demo_url"] = origin
        result["data_dir"] = expected_data
        save()

        if not occupied:
            config = configuration(app_dir, origin, args.public_origin)
            await runtime.fs.mkdir(str(control))
            remote_script = str(control / "supervisor.py")
            await runtime.fs.write_text(remote_script, RUNTIME_SCRIPT.read_text(encoding="utf-8"))
            await runtime.fs.write_text(str(control / "runtime.json"), json.dumps(config, indent=2) + "\n")
            await runtime.fs.write_text(str(control / "Caddyfile"), caddyfile(app_dir))
            await runtime.run_process("caddy", ["validate", "--config", str(control / "Caddyfile"), "--adapter", "caddyfile"], check=True)
            server = await runtime.create_process(
                python, ["-B", remote_script, str(control / "runtime.json")], cwd=app_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            result["supervisor_process_id"] = server.id
            save()
            print("Started API, worker and web entrypoint", flush=True)
        elif marker.get("origin") != origin:
            raise RuntimeError("A running demo has another origin; it was preserved. Stop it before reconfiguring")
        else:
            print("Reusing the existing demo processes and SQLite data", flush=True)

        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                check = await runtime.run_process(python, ["-B", "-c", PROBE, app_dir], capture_output=True, check=True)
                health = json.loads(check.stdout).get("health") or {}
                if health.get("analysis_available") is True:
                    break
                await asyncio.sleep(2)
            else:
                raise RuntimeError("API/worker readiness did not complete within 90 seconds")
            last_error = None
            for _ in range(6):
                try:
                    result["verification"] = await asyncio.to_thread(public_checks, origin)
                    last_error = None
                    break
                except (OSError, ValueError, RuntimeError) as exc:
                    last_error = exc
                    await asyncio.sleep(2)
            if last_error is not None:
                raise last_error
        except Exception:
            print("Readiness failed. Recent service logs:", flush=True)
            tail = "from pathlib import Path; import sys; root=Path(sys.argv[1]); [(print('--- '+p.name+' ---'), print(p.read_text(errors='replace')[-4000:])) for p in sorted(root.glob('*.log'))]"
            await runtime.run_process(python, ["-B", "-c", tail, str(control)], check=False)
            raise

        result["ready"] = True
        save()
        print(f"\nDemo URL: {origin}", flush=True)
        if args.public_origin:
            print(f"Allowed proxy origin: {args.public_origin} (configure the Vercel rewrite separately)", flush=True)
        print("Health: database/storage/worker OK; storage_backend=local", flush=True)
        print(f"Catalog records: {result['verification']['catalog_total']}", flush=True)
        print("Frontend, SPA routing and visitor cookie: OK", flush=True)
        print(f"Local state: {args.output}", flush=True)
        print("The Sandbox stays running after this script exits.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEPLOY_DIR / "sandbox-state.json")
    parser.add_argument("--output", type=Path, default=DEPLOY_DIR / "demo-state.json")
    parser.add_argument(
        "--public-origin", type=https_origin,
        help="Optional HTTPS proxy origin to allow, e.g. https://your-demo.vercel.app",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    prepared = json.loads(args.state.read_text())
    required = ["project_id", "team_id", "sandbox_name", "app_dir", "commit"]
    if not prepared.get("prepared") or any(not prepared.get(key) for key in required):
        raise SystemExit("The preparation state is incomplete; finish preparation first")
    if not PurePosixPath(prepared["app_dir"]).is_absolute():
        raise SystemExit("The prepared app_dir must be an absolute Sandbox path")
    if args.dry_run:
        config = configuration(prepared["app_dir"], "https://demo.invalid", args.public_origin)
        compile(PROBE, "<runtime-probe>", "exec")
        compile(RUNTIME_SCRIPT.read_text(encoding="utf-8"), "<supervisor>", "exec")
        print(json.dumps({"sandbox_name": prepared["sandbox_name"], "commit": prepared["commit"],
                          "profile": config["profile"], "data_dir": config["data_dir"],
                          "public_origin": args.public_origin,
                          "allowed_origins": json.loads(config["environment"]["MODEL_ANATOMY_CORS_ORIGINS"]),
                          "public_port": 8080, "output": str(args.output)}, indent=2))
        print("Dry run OK. No credentials requested and no Vercel calls made.")
        return
    asyncio.run(start(args, prepared))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. The remote Sandbox may still be running; keep the state files.")
    except Exception as exc:
        print(f"Failed: {type(exc).__name__}")
        if isinstance(exc, RuntimeError):
            print(str(exc))
        if isinstance(exc, HTTPError):
            print(f"HTTP status: {exc.code}")
        elif getattr(exc, "status_code", None) is not None:
            print(f"HTTP status: {exc.status_code}")
        raise SystemExit(1)
