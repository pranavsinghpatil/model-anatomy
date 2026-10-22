"""Stage related report files before publishing them, with ordinary-error rollback."""

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
    """Publish staged UTF-8 files and restore previous targets if publication raises.

    A group spanning multiple directories cannot be crash-atomic. This handles caught
    filesystem write/rename/unlink failures; it does not coordinate concurrent writers.
    Existing symlink targets are rejected so backups cannot silently change their meaning.
    """
    targets = list(contents) + sorted(obsolete - contents.keys())
    resolved = [path.resolve() for path in targets]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Report output paths overlap")
    if any(path.is_symlink() for path in targets):
        raise ValueError("Report output paths cannot be symlinks")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    changed: list[Path] = []
    retained: set[Path] = set()
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
                changed.append(path)
                if path in staged:
                    os.replace(staged[path], path)
                else:
                    path.unlink(missing_ok=True)
        except OSError as original:
            failures: list[str] = []
            for path in reversed(changed):
                backup = backups[path]
                try:
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
                except OSError:
                    if backup is not None:
                        retained.add(backup)
                    failures.append(str(path))
            if failures:
                kept = sorted(map(str, retained))
                raise OSError(
                    f"Output rollback failed for {failures}; retained backups: {kept}"
                ) from original
            raise
    finally:
        for temporary in [*staged.values(), *(item for item in backups.values() if item is not None)]:
            if temporary not in retained:
                temporary.unlink(missing_ok=True)
