"""Resolve the source checkout root for service configuration."""

from pathlib import Path


def project_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / "pyproject.toml").is_file() else Path.cwd().resolve()
