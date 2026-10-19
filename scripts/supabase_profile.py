"""Load and validate the private Supabase values needed by the local preview."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def load_supabase_profile(env_file: str | Path) -> dict[str, str]:
    """Return allowlisted settings; errors never include file contents or secrets."""
    try:
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        database_url = str(values.get("MODEL_ANATOMY_DATABASE_URL") or values.get("DATABASE_URL") or "").strip()
        supabase_url = str(values.get("MODEL_ANATOMY_SUPABASE_URL") or "").strip()
        key = str(values.get("MODEL_ANATOMY_SUPABASE_KEY") or values.get("MODEL_ANATOMY_SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        bucket = str(values.get("MODEL_ANATOMY_SUPABASE_BUCKET") or "model-anatomy-private").strip()
    except (OSError, ValueError, TypeError):
        raise ValueError("Could not read the selected Supabase environment file") from None

    if not database_url or not supabase_url or not key or not bucket:
        raise ValueError("Supabase profile requires database URL, storage URL, service key, and bucket")
    try:
        parsed_db = urlsplit(database_url)
        modes = parse_qs(parsed_db.query).get("sslmode", [])
        valid_database = (
            parsed_db.scheme in {"postgresql", "postgresql+psycopg"}
            and bool(parsed_db.hostname and parsed_db.username and parsed_db.password)
            and parsed_db.port == 5432
            and len(modes) == 1
            and modes[0] in {"require", "verify-ca", "verify-full"}
            and not parsed_db.fragment
            and not any(
                token in f"{parsed_db.hostname} {parsed_db.username}".upper()
                for token in ("REPLACE", "PROJECT", "POOLER_HOST")
            )
        )
    except (ValueError, TypeError):
        valid_database = False
    if not valid_database:
        raise ValueError("Supabase profile database URL must use PostgreSQL on port 5432 with TLS enabled") from None
    try:
        parsed_storage = urlsplit(supabase_url)
        valid_storage = (
            parsed_storage.scheme == "https"
            and bool(parsed_storage.hostname)
            and not parsed_storage.username
            and not parsed_storage.password
            and not parsed_storage.query
            and not parsed_storage.fragment
            and parsed_storage.port in (None, 443)
            and parsed_storage.path in {"", "/"}
        )
    except (ValueError, TypeError):
        valid_storage = False
    if not valid_storage:
        raise ValueError("Supabase profile storage URL must be an HTTPS project origin") from None
    if key.startswith("REPLACE") or "REPLACE" in key.upper() or not bucket:
        raise ValueError("Supabase profile credentials are incomplete")
    return {
        "DATABASE_URL": database_url,
        "MODEL_ANATOMY_DATABASE_URL": database_url,
        "MODEL_ANATOMY_SUPABASE_URL": supabase_url.rstrip("/"),
        "MODEL_ANATOMY_SUPABASE_KEY": key,
        "MODEL_ANATOMY_SUPABASE_BUCKET": bucket,
    }
