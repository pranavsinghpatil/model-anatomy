"""Create an unlinked Vercel Project and save its identifiers privately."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from support.credentials import PRIVATE_ENV, access_token, load_private_environment


def save_project(path: Path, project_id: str, team_id: str) -> None:
    """Preserve other private settings without printing account identifiers."""
    values = {"VERCEL_PROJECT_ID": project_id, "VERCEL_TEAM_ID": team_id}
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output = []
    for line in lines:
        key = line.partition("=")[0].strip()
        if key in values:
            output.append(f"{key}={values.pop(key)}")
        elif key not in {"VERCEL_PROJECT_ID", "VERCEL_TEAM_ID"}:
            output.append(line)
    output.extend(f"{key}={value}" for key, value in values.items())
    # Opening an existing file does not reset its permissions automatically.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as private:
        os.fchmod(private.fileno(), 0o600)
        private.write("\n".join(output) + "\n")


def create_project(name: str, *, private_env: Path = PRIVATE_ENV) -> None:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 100:
        raise RuntimeError("Use a project name of up to 100 lowercase letters, digits, and internal hyphens")
    load_private_environment(private_env)
    existing = os.environ.get("VERCEL_PROJECT_ID", "").strip()
    if existing and existing != "prj_your_project_id":
        raise RuntimeError("A Project ID is already configured; reuse that project instead of creating another")
    team_id = os.environ.get("VERCEL_TEAM_ID", "").strip()
    if not re.fullmatch(r"team_[A-Za-z0-9_-]+", team_id) or team_id == "team_your_team_id":
        raise RuntimeError("Set your VERCEL_TEAM_ID in private deploy/.env before creating a project")
    token = access_token()
    request = Request(
        "https://api.vercel.com/v11/projects?" + urlencode({"teamId": team_id}),
        data=json.dumps({"name": name}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            project = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Project creation failed with HTTP {exc.code}; inspect the dashboard before retrying") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Project creation could not be confirmed; inspect the dashboard before retrying") from None
    if not re.fullmatch(r"prj_[A-Za-z0-9_-]+", project.get("id", "")) or project.get("accountId") != team_id:
        raise RuntimeError("The created project account could not be verified; inspect Vercel before retrying")
    save_project(private_env, project["id"], team_id)
    print("Project created. Its identifiers were saved to private deploy/.env.")
    print("Open the project dashboard to choose its Production domain and review access settings.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="A project name in your own Vercel account")
    args = parser.parse_args()
    create_project(args.name)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
