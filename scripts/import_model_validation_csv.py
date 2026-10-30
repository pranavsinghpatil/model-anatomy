"""Import a sanitized external validation CSV into private catalog storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def import_csv(
    csv_path: Path,
    *,
    source_repository: str,
    source_commit: str,
    source_path: str,
    env_file: Path,
    profile_loader=None,
    database_factory=None,
) -> str:
    """Validate before connecting, then store an immutable publication only."""
    from scripts.supabase_profile import load_supabase_profile

    from model_anatomy.api.settings import Settings
    from model_anatomy.service.database import Database
    from model_anatomy.service.external_validation import (
        MAX_CSV_BYTES,
        ExternalValidationService,
        prepare_publication,
    )

    with csv_path.open("rb") as handle:
        raw = handle.read(MAX_CSV_BYTES + 1)
    # Perform strict parsing and redaction before any database work.
    prepare_publication(
        raw, source_repository=source_repository, source_commit=source_commit, source_path=source_path,
    )
    profile = (profile_loader or load_supabase_profile)(env_file)
    make_database = database_factory or Database
    settings = Settings.model_validate({
        "environment": "production", "allowed_hosts": ["localhost"], "cors_origins": ["https://localhost"],
        "database_url": profile["DATABASE_URL"], "supabase_url": profile["MODEL_ANATOMY_SUPABASE_URL"],
        "supabase_service_role_key": profile["MODEL_ANATOMY_SUPABASE_KEY"],
        "storage_backend": "supabase", "supabase_bucket": profile["MODEL_ANATOMY_SUPABASE_BUCKET"],
    })
    database_url = settings.resolved_database_url
    database = None
    try:
        database = make_database(database_url)
        database.initialize(create=False)
        return ExternalValidationService(database).publish(
            raw, source_repository=source_repository, source_commit=source_commit, source_path=source_path,
        )
    except Exception:  # noqa: BLE001 - redact database/provider exception details
        raise RuntimeError("external validation import failed; inspect local service diagnostics") from None
    finally:
        if database is not None:
            database.engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--env-file", type=Path, default=Path("infra/.env"))
    args = parser.parse_args()
    try:
        publication_id = import_csv(
            args.csv,
            source_repository=args.source_repository,
            source_commit=args.source_commit,
            source_path=args.source_path,
            env_file=args.env_file,
        )
    except Exception as exc:  # noqa: BLE001 - print only exception type/message after redaction
        print(f"External validation import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Stored external validation publication {publication_id}")


if __name__ == "__main__":
    main()
