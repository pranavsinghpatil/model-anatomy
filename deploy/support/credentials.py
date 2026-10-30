"""Explicit token authentication for interactive operators and GitHub Actions."""

from __future__ import annotations

import getpass
import os
import re
import shlex
from pathlib import Path

PRIVATE_ENV = Path(__file__).resolve().parents[1] / ".env"
ENVIRONMENT_KEYS = {"VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID", "DEMO_PUBLIC_ORIGIN"}


def load_private_environment(path: Path = PRIVATE_ENV) -> None:
    """Read literal local values; CI uses injected Secrets and never reads this file."""
    if os.environ.get("CI") or not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().partition("=")
        if not separator or key not in ENVIRONMENT_KEYS or key in os.environ:
            continue
        try:
            values = shlex.split(value, comments=True)
        except ValueError:
            raise RuntimeError("The private deployment env file has an invalid quoted value") from None
        if len(values) > 1:
            raise RuntimeError("Quote values containing spaces in the private deployment env file")
        os.environ[key] = values[0] if values else ""


def destination() -> dict[str, str]:
    load_private_environment()
    result = {}
    for key, variable, prefix in [
        ("project_id", "VERCEL_PROJECT_ID", "prj_"), ("team_id", "VERCEL_TEAM_ID", "team_"),
    ]:
        value = os.environ.get(variable, "").strip()
        if not re.fullmatch(prefix + r"[A-Za-z0-9_-]+", value) or value in {"prj_your_project_id", "team_your_team_id"}:
            raise RuntimeError(f"Set {variable} in GitHub Actions Secrets or the private deploy/.env file")
        result[key] = value
    return result


def bind_destination(state: dict) -> dict:
    configured = destination()
    if any(state.get(key) not in (None, value) for key, value in configured.items()):
        raise RuntimeError("The legacy state belongs to another configured Vercel destination")
    return {**state, **configured}


def without_destination(value):
    """Keep private account identifiers out of generated diagnostic JSON."""
    if isinstance(value, dict):
        return {key: without_destination(item) for key, item in value.items() if key not in {"project_id", "team_id"}}
    if isinstance(value, list):
        return [without_destination(item) for item in value]
    return value


def access_token() -> str:
    load_private_environment()
    token = os.environ.get("VERCEL_TOKEN", "").strip()
    if not token and os.environ.get("CI"):
        raise RuntimeError("Set the VERCEL_TOKEN repository secret before running this workflow")
    if not token:
        token = getpass.getpass("Vercel Access Token (hidden): ").strip()
    if not token:
        raise RuntimeError("A Vercel access token is required")
    return token
