"""Dependency preflight without importing model libraries or installing packages."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

Extra = Literal["api", "engine"]
DEPENDENCIES: dict[Extra, tuple[tuple[str, str], ...]] = {
    "api": (
        ("requests", "requests"),
        ("fastapi", "fastapi"),
        ("pydantic-settings", "pydantic_settings"),
        ("python-dotenv", "dotenv"),
        ("uvicorn", "uvicorn"),
        ("sqlalchemy", "sqlalchemy"),
        ("alembic", "alembic"),
        ("psycopg", "psycopg"),
        ("httpx", "httpx"),
    ),
    "engine": (
        ("requests", "requests"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("safetensors", "safetensors"),
        ("gguf", "gguf"),
    ),
}


def missing_dependencies(*extras: Extra) -> list[str]:
    """Check module availability only; native binary compatibility is not implied."""
    requested = dict(item for extra in extras for item in DEPENDENCIES[extra])
    return [package for package, module in requested.items() if find_spec(module) is None]


def require_dependencies(*extras: Extra, component: str) -> None:
    missing = missing_dependencies(*extras)
    if missing:
        flags = " ".join(f"--extra {extra}" for extra in dict.fromkeys(extras))
        raise SystemExit(
            f"{component} cannot start. Missing packages: {', '.join(missing)}.\n"
            f"From the project root, run: uv sync --frozen {flags}\n"
            "No packages were installed automatically."
        )
