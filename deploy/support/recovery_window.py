"""Bound automatic recovery to the agreed demo window, then disable its schedule."""

from __future__ import annotations

import argparse
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# Midnight after December 16 in New York is 05:00 UTC on December 17, 2026.
RECOVERY_END = datetime(2026, 12, 17, 5, tzinfo=UTC)


def recovery_allowed(now: datetime | None = None) -> bool:
    current = datetime.now(UTC) if now is None else now
    if current.tzinfo is None:
        raise ValueError("Recovery window checks require a timezone-aware timestamp")
    return current < RECOVERY_END


def disable_expired_workflow() -> None:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not token:
        raise RuntimeError("Workflow self-disable requires GitHub's built-in repository and token context")
    request = Request(
        f"https://api.github.com/repos/{repository}/actions/workflows/recover-demo.yml/disable",
        method="PUT", headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            if response.status != 204:
                raise RuntimeError("Automatic recovery workflow disable was not confirmed")
    except HTTPError as exc:
        raise RuntimeError(f"Workflow self-disable returned HTTP {exc.code}; disable Recover Demo in GitHub") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-expired-workflow", action="store_true")
    args = parser.parse_args()
    allowed = recovery_allowed()
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"active={str(allowed).lower()}\n")
    print(f"Automatic recovery active: {allowed}; deadline: {RECOVERY_END.isoformat()}")
    if not allowed and args.disable_expired_workflow:
        disable_expired_workflow()
        print("The expired Recover Demo workflow was disabled.")


if __name__ == "__main__":
    main()
