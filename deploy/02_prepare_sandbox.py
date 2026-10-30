"""Prepare committed Model Anatomy source in Vercel Sandbox; do not start the app.

Run with the isolated SDK, without changing the application's Python environment:
  uv run --no-project --with vercel-sandbox==0.10.0 python deploy/02_prepare_sandbox.py --repo PATH

The SDK credentials come from VERCEL_TOKEN or a local prompt and are never saved.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import subprocess
import tarfile
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath

from support.credentials import access_token, destination, load_private_environment, without_destination

BRANCH = "feat/deploy"


def committed_source(repo: Path, branch: str = BRANCH, revision: str | None = None) -> tuple[str, bytes, int]:
    root = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("An explicit revision must be a full Git commit SHA")
    revision = subprocess.check_output(
        ["git", "-C", root, "rev-parse", f"{revision or 'refs/heads/' + branch}^{{commit}}"],
        text=True,
    ).strip()
    payload = subprocess.check_output(
        ["git", "-C", root, "archive", "--format=tar", revision]
    )
    forbidden_suffixes = (
        ".pem", ".key", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm",
        ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".h5",
        ".npz", ".npy",
    )
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = archive.getmembers()
        names = {item.name for item in members}
        required = {"pyproject.toml", "uv.lock", "frontend/package.json", "alembic.ini"}
        if not required.issubset(names):
            raise RuntimeError("The selected branch lacks required application files")
        for item in members:
            path = PurePosixPath(item.name)
            if path.is_absolute() or ".." in path.parts or not (item.isfile() or item.isdir()):
                raise RuntimeError("Archive contains an unsafe path, link, or special file")
            if any(part in {".git", ".tmp", ".venv", "node_modules", ".vercel"} for part in path.parts):
                raise RuntimeError("Archive contains generated state or local credentials")
            private_env = (
                path.name == ".env"
                or (path.name.startswith(".env.") and path.name != ".env.example")
            )
            if private_env or path.name.lower().endswith(forbidden_suffixes):
                raise RuntimeError(f"Archive contains a private config or binary payload: {item.name}")
    return revision, payload, sum(item.isfile() for item in members)


async def prepare(args: argparse.Namespace, revision: str, payload: bytes) -> None:
    from vercel import sandbox
    from vercel.api import session
    from vercel.sandbox import (
        SandboxCredentials,
        SandboxResources,
        SandboxServiceOptions,
        SnapshotRetention,
    )

    team_id = args.team_id or os.environ.get("VERCEL_TEAM_ID", "").strip()
    if not team_id and not os.environ.get("CI"):
        team_id = input("Vercel Team ID: ").strip()
    token = access_token()
    if not team_id.startswith("team_") or not token:
        raise RuntimeError("A team_ ID and a nonempty access token are required")

    async def credentials() -> SandboxCredentials:
        return SandboxCredentials(token=token, project_id=args.project_id, team_id=team_id)

    state = {
        "project_id": args.project_id,
        "team_id": team_id,
        "branch": args.branch,
        "commit": revision,
        "prepared": False,
    }

    def save_state() -> None:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(without_destination(state), indent=2) + "\n", encoding="utf-8")

    async with session(service_options=[SandboxServiceOptions(credentials_factory=credentials)]):
        box = await sandbox.create_sandbox(
            name=f"model-anatomy-{revision[:8]}-{uuid.uuid4().hex[:6]}",
            image="vercel/sandbox/universal:latest",
            execution_time_limit=timedelta(minutes=45),
            resources=SandboxResources(vcpus=4, memory=8192),
            persistent=True,
            snapshot_expiration=timedelta(days=60),
            snapshot_retention=SnapshotRetention(count=1),
            ports=[],
            tags={"app": "model-anatomy", "purpose": "demo", "commit": revision, "branch": args.branch},
        )
        state["sandbox_name"] = box.name
        state["session_id"] = box.current_session_id
        app_dir = str(PurePosixPath(box.cwd or "/vercel/sandbox") / "model-anatomy")
        state["app_dir"] = app_dir
        save_state()
        print(f"Sandbox: {box.name}", flush=True)
        print(f"State file: {args.state}", flush=True)

        async def run(label: str, command: str, argv: list[str], **kwargs):
            print(f"\n>>> {label}", flush=True)
            kwargs["env"] = {**install_environment, **kwargs.pop("env", {})}
            return await box.run_process(command, argv, check=True, **kwargs)

        scratch = str(PurePosixPath(app_dir).parent / ".tmp" / box.name)
        install_environment = {
            "TMPDIR": scratch, "TMP": scratch, "TEMP": scratch,
            "UV_CACHE_DIR": scratch + "/uv-cache",
            "UV_PYTHON_INSTALL_DIR": scratch + "/python",
            "npm_config_cache": scratch + "/npm-cache",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        stage = "upload committed source"
        try:
            await box.fs.mkdir(scratch)
            archive_path = scratch + "/source.tar"
            await box.fs.write_bytes(archive_path, payload, mode=0o600)
            await box.fs.mkdir(app_dir)
            await run(stage, "tar", ["-xf", archive_path, "-C", app_dir])
            await box.fs.remove(archive_path)
            await box.fs.write_text(
                str(PurePosixPath(app_dir) / "DEPLOYED_COMMIT"), revision + "\n"
            )

            stage = "check Node.js and npm"
            await run(stage, "node", ["--version"])
            await run(stage, "npm", ["--version"])

            stage = "find uv"
            probe = await box.run_process(
                "sh", ["-c", "command -v uv"], capture_output=True
            )
            uv = probe.stdout.strip() if probe.returncode == 0 else ""
            if not uv:
                stage = "install uv"
                await run(stage, "curl", [
                    "--fail", "--location", "--silent", "--show-error",
                    "https://astral.sh/uv/0.9.6/install.sh",
                    "--output", scratch + "/uv-install.sh",
                ])
                await run(stage, "sh", [scratch + "/uv-install.sh"], env={
                    "UV_INSTALL_DIR": scratch + "/tools",
                    "UV_NO_MODIFY_PATH": "1",
                })
                uv = scratch + "/tools/uv"

            stage = "install locked Python 3.11 API and engine dependencies"
            await run(stage, uv, [
                "sync", "--frozen", "--extra", "api", "--extra", "engine",
                "--no-dev", "--no-default-groups", "--python", "3.11",
            ], cwd=app_dir, kill_after=20 * 60)

            stage = "check CPU engine and API imports"
            python = str(PurePosixPath(app_dir) / ".venv/bin/python")
            await run(stage, python, ["-c", (
                "import torch, transformers; "
                "from model_anatomy.src.native_provider import require_supported_provider; "
                "from model_anatomy.api.app import create_app; "
                "require_supported_provider(); "
                "assert torch.version.cuda is None, 'Expected CPU-only Torch'; "
                "print('CPU Torch:', torch.__version__); "
                "print('Transformers:', transformers.__version__); "
                "print('Engine and API imports: OK')"
            )], cwd=app_dir, env={
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            })

            frontend_dir = str(PurePosixPath(app_dir) / "frontend")
            stage = "install locked frontend dependencies"
            await run(stage, "npm", ["ci", "--no-audit", "--no-fund"],
                      cwd=frontend_dir, kill_after=10 * 60)
            stage = "build frontend"
            await run(stage, "npm", ["run", "build"],
                      cwd=frontend_dir, kill_after=5 * 60)

            stage = "install the Caddy web entrypoint"
            await run(stage, "apt-get", ["update"], sudo=True, kill_after=5 * 60)
            await run(stage, "apt-get", ["install", "--yes", "--no-install-recommends", "caddy"],
                      sudo=True, env={"DEBIAN_FRONTEND": "noninteractive"}, kill_after=5 * 60)
            await run("check Caddy", "caddy", ["version"])

            state["uv_path"] = uv
            state["prepared"] = True
            state["stage"] = "prepared; database configuration and app startup pending"
            save_state()
            print("\nSource upload, dependency imports, frontend build and Caddy: OK", flush=True)
        except Exception as exc:
            state["stage"] = stage
            state["error_type"] = type(exc).__name__
            save_state()
            print(f"\nPreparation failed at: {stage} ({type(exc).__name__})", flush=True)
            raise
        finally:
            print("Stopping this preparation session and saving its filesystem...", flush=True)
            try:
                await box.stop()
                state["stop_requested"] = True
            except Exception as exc:  # noqa: BLE001 - Cleanup failures must not expose SDK request details.
                state["stop_error_type"] = type(exc).__name__
                print("Stop could not be confirmed; check this sandbox in the Vercel dashboard.", flush=True)
            save_state()

        print(f"\nSandbox name: {box.name}", flush=True)
        print(f"Preparation complete: {state['prepared']}", flush=True)
        print("The API and worker have not been started. Supabase is not contacted.", flush=True)


def main() -> None:
    load_private_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--revision", help="Optional full commit SHA, including a detached Actions checkout")
    parser.add_argument("--project-id", default=os.environ.get("VERCEL_PROJECT_ID"))
    parser.add_argument("--team-id", default=os.environ.get("VERCEL_TEAM_ID"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state", type=Path, default=Path(__file__).resolve().with_name("sandbox-state.json"))
    args = parser.parse_args()
    if not args.project_id or not args.team_id:
        configured = destination()
        args.project_id = args.project_id or configured["project_id"]
        args.team_id = args.team_id or configured["team_id"]
    args.state = args.state.expanduser().resolve()
    if args.state.exists() and not args.dry_run:
        raise SystemExit("State file already exists. Keep it for resuming; do not create another sandbox yet.")
    revision, payload, count = committed_source(args.repo.expanduser().resolve(), args.branch, args.revision)
    print(f"Branch: {args.branch}\nCommit: {revision}\nCommitted files: {count}\nArchive size: {len(payload) / 1024**2:.1f} MiB", flush=True)
    if args.dry_run:
        print("Dry run passed. No credentials read and no Vercel requests sent.")
        return
    asyncio.run(prepare(args, revision, payload))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Check sandbox-state.json and the Vercel dashboard before retrying.")
    except Exception as exc:  # noqa: BLE001 - Redact credential-bearing SDK errors at the CLI boundary.
        print(f"Error type: {type(exc).__name__}")
        status = getattr(exc, "status_code", None)
        if status is not None:
            print(f"HTTP status: {status}")
        raise SystemExit(1)
