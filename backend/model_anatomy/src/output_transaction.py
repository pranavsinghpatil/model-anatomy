"""Stage report files before publishing, with rollback for ordinary write failures."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def _sibling(path: Path, label: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".anatomy-{label}-", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def publish_files(contents: dict[Path, str], obsolete: set[Path]) -> None:
    """Replace each file atomically and restore the prior set if publication raises.

    Multiple directories do not support a single filesystem transaction. This protects
    against caught write/rename failures, not process crashes or concurrent writers.
    Recovery failures retain backup files and report their locations.
    """
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    changed: list[Path] = []
    retained: set[Path] = set()
    targets = list(contents) + sorted(obsolete - contents.keys())
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("Report output paths overlap")
    try:
        for path, text in contents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            stage = _sibling(path, "stage")
            staged[path] = stage
            with stage.open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for path in targets:
            if path.exists():
                backup = _sibling(path, "backup")
                backups[path] = backup
                shutil.copy2(path, backup)
            else:
                backups[path] = None
        try:
            for path in targets:
                if path in staged:
                    os.replace(staged[path], path)
                else:
                    path.unlink(missing_ok=True)
                changed.append(path)
        except OSError as original:
            failures = []
            for path in reversed(changed):
                backup = backups[path]
                try:
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
                except OSError:
                    if backup:
                        retained.add(backup)
                    failures.append(str(path))
            if failures:
                raise OSError(
                    f"Output rollback failed for {failures}; retained backups: {sorted(map(str, retained))}"
                ) from original
            raise
    finally:
        for temporary in [
            *staged.values(),
            *(item for item in backups.values() if item is not None),
        ]:
            if temporary not in retained:
                temporary.unlink(missing_ok=True)
