"""Prepare committed Model Anatomy source in Vercel Sandbox; do not start the app.

Run with the isolated SDK, without changing the application's Python environment:
  uv run --no-project --with vercel-sandbox==0.10.0 python prepare_sandbox.py --repo PATH

The SDK credentials are prompted locally and never written to the state file.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import io
import json
import subprocess
import tarfile
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath


PROJECT_ID = "prj_mWgEmNh9xUbrq8fp7JSAk4kNP7fF"
BRANCH = "feat/deploy"


def committed_source(repo: Path) -> tuple[str, bytes, int]:
    root = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    revision = subprocess.check_output(
        ["git", "-C", root, "rev-parse", f"refs/heads/{BRANCH}^{{commit}}"],
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

    team_id = input("Vercel Team ID: ").strip()
    token = getpass.getpass("Vercel Access Token (hidden): ").strip()
    if not team_id.startswith("team_") or not token:
        raise RuntimeError("A team_ ID and a nonempty access token are required")

    async def credentials() -> SandboxCredentials:
        return SandboxCredentials(token=token, project_id=PROJECT_ID, team_id=team_id)

    state = {
        "project_id": PROJECT_ID,
        "team_id": team_id,
        "branch": BRANCH,
        "commit": revision,
        "prepared": False,
    }

    def save_state() -> None:
        args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    async with session(service_options=[SandboxServiceOptions(credentials_factory=credentials)]):
        box = await sandbox.create_sandbox(
            name=f"model-anatomy-{revision[:8]}-{uuid.uuid4().hex[:6]}",
            image="vercel/sandbox/universal:latest",
            execution_time_limit=timedelta(minutes=45),
            resources=SandboxResources(vcpus=4, memory=8192),
            persistent=True,
            snapshot_expiration=timedelta(days=3),
            snapshot_retention=SnapshotRetention(count=1),
            ports=[],
            tags={"app": "model-anatomy", "purpose": "demo", "commit": revision},
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
            return await box.run_process(command, argv, check=True, **kwargs)

        stage = "upload committed source"
        try:
            archive_path = "/tmp/model-anatomy-source.tar"
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
                    "--output", "/tmp/model-anatomy-uv-install.sh",
                ])
                await run(stage, "sh", ["/tmp/model-anatomy-uv-install.sh"], env={
                    "UV_INSTALL_DIR": "/tmp/model-anatomy-tools",
                    "UV_NO_MODIFY_PATH": "1",
                })
                uv = "/tmp/model-anatomy-tools/uv"

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
            except Exception as exc:
                state["stop_error_type"] = type(exc).__name__
                print("Stop could not be confirmed; check this sandbox in the Vercel dashboard.", flush=True)
            save_state()

        print(f"\nSandbox name: {box.name}", flush=True)
        print(f"Preparation complete: {state['prepared']}", flush=True)
        print("The API and worker have not been started. Supabase is not contacted.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state", type=Path, default=Path(__file__).resolve().with_name("sandbox-state.json"))
    args = parser.parse_args()
    args.state = args.state.expanduser().resolve()
    if args.state.exists() and not args.dry_run:
        raise SystemExit("State file already exists. Keep it for resuming; do not create another sandbox yet.")
    revision, payload, count = committed_source(args.repo.expanduser().resolve())
    print(f"Branch: {BRANCH}\nCommit: {revision}\nCommitted files: {count}\nArchive size: {len(payload) / 1024**2:.1f} MiB", flush=True)
    if args.dry_run:
        print("Dry run passed. No credentials read and no Vercel requests sent.")
        return
    asyncio.run(prepare(args, revision, payload))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Check sandbox-state.json and the Vercel dashboard before retrying.")
    except Exception as exc:
        print(f"Error type: {type(exc).__name__}")
        status = getattr(exc, "status_code", None)
        if status is not None:
            print(f"HTTP status: {status}")
        raise SystemExit(1)
