"""Read-only dependency checks: no inference, migrations, uploads, or provisioning.

Print fixed status messages only. Provider exceptions and credentials stay private.
Run inside the API image so its frozen dependencies and environment are used.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote, urlsplit

TOKEN_FACTORY_HOSTS = {"api.tokenfactory.nebius.com", "api.tokenfactory.us-central1.nebius.com"}


def required(name: str) -> str:
    value = os.environ.get(f"MODEL_ANATOMY_{name}", "").strip()
    if not value or "REPLACE" in value or "PROJECT" in value or "POOLER_HOST" in value:
        raise ValueError("Required configuration missing")
    return value


def https_base(value: str, *, allowed_hosts: set[str] | None = None) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port not in (None, 443)
        or (allowed_hosts is not None and parsed.hostname not in allowed_hosts)
    ):
        raise ValueError("Invalid provider URL")
    return value.rstrip("/")


def read_json(url: str, headers: dict[str, str]) -> dict:
    import httpx

    # A redirect must never forward a server credential to a different origin.
    with (
        httpx.Client(timeout=15, follow_redirects=False, trust_env=False) as client,
        client.stream("GET", url, headers=headers) as response,
    ):
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > 2 * 1024**2:
                raise ValueError("Provider response exceeds limit")
    result = json.loads(body)
    if not isinstance(result, dict):
        raise TypeError("Provider response must be an object")
    return result


def check_models() -> None:
    model = required("NEBIUS_MODEL")
    if not model.lower().startswith("nvidia/") or "nemotron" not in model.lower():
        raise ValueError("An NVIDIA Nemotron model ID is required")
    base = https_base(
        os.environ.get("MODEL_ANATOMY_NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
        allowed_hosts=TOKEN_FACTORY_HOSTS,
    )
    if urlsplit(base).path != "/v1":
        raise ValueError("Token Factory API must use /v1")
    response = read_json(f"{base}/models", {"Authorization": f"Bearer {required('NEBIUS_API_KEY')}"})
    if not any(isinstance(item, dict) and item.get("id") == model for item in response.get("data", [])):
        raise ValueError("Selected model not listed")


def check_bucket() -> None:
    base = https_base(required("SUPABASE_URL"))
    if urlsplit(base).path not in ("", "/"):
        raise ValueError("Supabase URL must be a project origin")
    key, bucket = required("SUPABASE_KEY"), required("SUPABASE_BUCKET")
    result = read_json(
        f"{base}/storage/v1/bucket/{quote(bucket, safe='')}",
        {"Authorization": f"Bearer {key}", "apikey": key},
    )
    if result.get("id") != bucket or result.get("public") is not False:
        raise ValueError("Expected private bucket missing")


def check_database(config_path: str, *, skip_schema: bool = False) -> None:
    import psycopg
    from sqlalchemy.engine import make_url

    raw = required("DATABASE_URL")
    url = make_url(raw)
    if url.drivername not in ("postgresql", "postgresql+psycopg"):
        raise ValueError("PostgreSQL is required")
    if url.query.get("sslmode") not in ("require", "verify-ca", "verify-full"):
        raise ValueError("Database TLS is required")
    connection_url = url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(
        connection_url,
        connect_timeout=10,
        options="-c default_transaction_read_only=on -c statement_timeout=10000",
    ) as connection:
        connection.execute("SELECT 1")
        if skip_schema:
            return
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        from model_anatomy.service.database import Base

        heads = set(ScriptDirectory.from_config(Config(config_path)).get_heads())
        if not heads:
            raise ValueError("No migration heads")
        actual_heads = {
            row[0] for row in connection.execute("SELECT version_num FROM public.alembic_version")
        }
        if actual_heads != heads:
            raise ValueError("Database migrations are not at head")
        columns = connection.execute(
            "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
        ).fetchall()
        actual_columns: dict[str, set[str]] = {}
        for table_name, column_name in columns:
            actual_columns.setdefault(table_name, set()).add(column_name)
        for table in Base.metadata.sorted_tables:
            if not set(table.columns.keys()).issubset(actual_columns.get(table.name, set())):
                raise ValueError("Database schema is incomplete")


def check_runtime_contract(root: Path) -> None:
    """Check packaged runtime inputs without loading a model or contacting providers."""
    required_paths = (
        "backend/model_anatomy/api/app.py",
        "backend/model_anatomy/service/worker.py",
        "infra/healthcheck.py",
        "alembic.ini",
    )
    if any(not (root / relative).is_file() for relative in required_paths):
        raise ValueError("Required API, worker, healthcheck, or migration configuration is missing")
    if not (root / "migrations").is_dir() or not any((root / "migrations").glob("versions/*.py")):
        raise ValueError("Migration scripts are missing")
    if (root / "output").exists():
        raise ValueError("Runtime image must not contain model reports or snapshots")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static",
        "--offline",
        dest="static",
        action="store_true",
        help="Check packaged runtime files only; make no provider or model claims",
    )
    parser.add_argument(
        "--skip-schema", action="store_true", help="Before migration: check DB connectivity only"
    )
    parser.add_argument("--alembic-config", default="alembic.ini")
    args = parser.parse_args(argv)
    if args.static:
        checks = [("static runtime contract", lambda: check_runtime_contract(Path.cwd()))]
    else:
        checks = []
        checks.extend(
            [
                ("database", lambda: check_database(args.alembic_config, skip_schema=args.skip_schema)),
                ("private storage", check_bucket),
                ("NVIDIA model catalog", check_models),
            ]
        )
    failed = False
    for label, check in checks:
        try:
            check()
        except Exception:  # noqa: BLE001 - this boundary must suppress credential-bearing errors
            # Never print exception text, response bodies, URLs, keys, or env values.
            print(f"FAIL: {label} (check private configuration and provider availability)")
            failed = True
        else:
            suffix = (
                " (connectivity only; schema skipped)" if label == "database" and args.skip_schema else ""
            )
            print(f"PASS: {label}{suffix}")
    if args.static:
        print("NOT CHECKED: provider connectivity, model inspection, inference, deployment")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
