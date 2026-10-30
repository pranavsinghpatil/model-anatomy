"""Explicitly enrich old report summaries from digest-verified private report blobs."""

from __future__ import annotations

import argparse
import json

from sqlalchemy import select

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import Database, ModelReport
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import cache_key, ir_summary
from model_anatomy.service.storage import storage_for

MAX_LIMIT = 1000
METRIC_FIELDS = {"warning_count", "graph_count", "tensor_count", "declared_task", "category"}


def backfill(database, storage, *, limit: int, after_id: str | None = None) -> dict[str, int | str | None]:
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    with database.read() as session:
        reports = session.scalars(select(ModelReport).order_by(ModelReport.id)).all()
    if after_id is not None:
        reports = [report for report in reports if report.id > after_id]
    updated = failed = considered = 0
    last_id = None
    for report in reports:
        if METRIC_FIELDS.issubset(report.summary or {}):
            continue
        if considered >= limit:
            break
        considered += 1
        last_id = report.id
        try:
            document = json.loads(storage.get(report.blob_digest))
            metadata = document.get("metadata") if isinstance(document, dict) else None
            if (
                not isinstance(metadata, dict)
                or document.get("model_id") != report.model_id
                or document.get("schema_version") != report.schema_version
                or metadata.get("revision") != report.revision
                or metadata.get("engine_version", report.engine_version) != report.engine_version
                or report.cache_identity != cache_key(
                    report.model_id, report.revision, report.engine_version,
                    report.schema_version, report.artifact, report.explicit_artifact,
                )
            ):
                raise ValueError("report identity mismatch")
            summary = ir_summary(document)
            with database.write() as session:
                current = session.get(ModelReport, report.id)
                if current and current.blob_digest == report.blob_digest:
                    # Add only the newly introduced metric facts. Historical result status,
                    # quality, limitations and all existing summary semantics stay immutable.
                    merged = dict(current.summary or {})
                    for field in METRIC_FIELDS:
                        if field not in merged:
                            merged[field] = summary.get(field)
                    current.summary = merged
                    updated += 1
        except (ServiceError, ValueError, KeyError, TypeError, AttributeError):
            failed += 1
    return {"considered": considered, "updated": updated, "failed": failed, "last_id": last_id}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", required=True, type=int, help=f"maximum reports to inspect (1-{MAX_LIMIT})")
    parser.add_argument("--after-id", help="resume after the last_id returned by a previous bounded batch")
    args = parser.parse_args()
    settings = Settings()
    database = Database(settings.resolved_database_url)
    database.initialize(create=settings.environment == "local")
    storage = storage_for(settings)
    try:
        print(json.dumps(backfill(database, storage, limit=args.limit, after_id=args.after_id), sort_keys=True))
    finally:
        database.engine.dispose()


if __name__ == "__main__":
    main()
