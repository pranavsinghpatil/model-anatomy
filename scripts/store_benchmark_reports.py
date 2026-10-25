"""Capture validated benchmark IRs and import them into private Supabase storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing.connection
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import model_matrix
from scripts.model_matrix import (
    append_record,
    case_identity,
    dependency_versions,
    source_tree_sha256,
    validate_manifest,
)

SUCCESS_STATUSES = {"completed", "partial", "inspected"}
SERVICE_STORAGE_LIMIT_BYTES = 32 * 1024**2


def _capture_child(case: dict[str, Any], connection: multiprocessing.connection.Connection) -> None:
    """Spawn-safe capture worker; leave diagnostic text out of durable records."""
    report_directory = Path(case["__report_directory"])
    measured_case = {key: value for key, value in case.items() if not key.startswith("__")}
    try:
        record = model_matrix._analyze_case(measured_case, report_directory=report_directory)
        connection.send(("result", record))
    except Exception as exc:  # noqa: BLE001 - child must return a typed, bounded outcome
        classified = model_matrix.classify_failure(exc)
        payload = {
            "classification": classified["status"],
            "stage": classified["stage"],
            "cause_type": classified["cause_type"],
        }
        try:
            connection.send(("error", payload))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _verified_report(case: dict[str, Any], row: dict[str, Any], report_directory: Path) -> bool:
    """Validate a prior captured success before allowing resume to reuse it."""
    if row.get("status") not in SUCCESS_STATUSES or row.get("service_contract_valid") is not True:
        return False
    digest = row.get("report_digest")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return False
    path = report_directory / f"{digest}.json"
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest or len(data) != row.get("serialized_report_bytes"):
            return False
        from model_anatomy.service.jobs import ir_summary
        from model_anatomy.src.ir import ModelIR
        from model_anatomy.src.version import ENGINE_VERSION

        document = json.loads(data)
        ir = ModelIR.from_json(data.decode("utf-8"))
        ir.root_graph.validate()
        target = case.get("reference") or case["model_id"]
        from model_anatomy.service.jobs import request_selector

        model_id, requested_revision, _artifact, _explicit = request_selector(target)
        expected_revision = case["revision"] if requested_revision == "main" else requested_revision
        return (
            document.get("model_id") == case["model_id"] == model_id
            and ir.model_id == case["model_id"]
            and document.get("schema_version") == row.get("report_roundtrip", {}).get("schema_version")
            and document.get("schema_version") == ir.schema_version
            and ir.metadata.get("revision") == case["revision"] == expected_revision
            and ir.metadata.get("engine_version", ENGINE_VERSION) == row.get("engine_version") == ENGINE_VERSION
            and ir.to_dict() == document
            and row.get("report_roundtrip", {}).get("valid") is True
            and row.get("storage_contract_valid") is True
            and row.get("weight_payload_bytes_read") == 0
            and row.get("forward_executed") is False
            and ir_summary(document)["result_status"] == row.get("status")
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def capture_only(
    manifest_path: Path,
    output_path: Path,
    report_directory: Path,
    *,
    timeout: float = 120,
    memory_bytes: int = 4 * 1024**3,
    allow_small: bool = False,
    resume: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run bounded pinned cases and retain only fully validated IR JSON reports."""
    from model_anatomy.src.version import ENGINE_VERSION

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = validate_manifest(manifest, allow_small=allow_small)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    previous = model_matrix.read_existing(output_path) if resume else {}
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_hash = source_tree_sha256()
    environment = {
        "timeout_seconds": timeout,
        "memory_bytes": memory_bytes,
        "python": sys.version.split()[0],
        "platform": model_matrix.platform.platform(),
        "dependencies": dependency_versions(),
    }
    results = []
    for index, case in enumerate(cases, 1):
        identity = case_identity(case, ENGINE_VERSION, runner_hash, engine_hash, environment)
        cached = previous.get(identity)
        if cached and _verified_report(case, cached, report_directory):
            record = {**cached, "resumed": True}
            print(f"[{index}/{len(cases)}] {case['model_id']} -> reuse {record['status']}", flush=True)
        else:
            start = datetime.now(UTC).isoformat()
            print(f"[{index}/{len(cases)}] {case['model_id']} ({case['category']}) starting", flush=True)
            worker_case = {**case, "__report_directory": str(report_directory.resolve())}
            record = model_matrix.run_case(
                worker_case, timeout=timeout, memory_bytes=memory_bytes, worker_target=_capture_child
            )
            record.update(
                {
                    "case_identity": identity,
                    "runner_sha256": runner_hash,
                    "engine_source_sha256": engine_hash,
                    "started_at": start,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "resumed": False,
                }
            )
            append_record(output_path, record)
            print(f"[{index}/{len(cases)}] {case['model_id']} -> {record.get('status')}", flush=True)
        results.append(record)
    counts: dict[str, int] = {}
    for record in results:
        status = record.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "record_file": str(output_path.resolve()),
        "report_directory": str(report_directory.resolve()),
        "cases": len(cases),
        "status_counts": counts,
        "environment": {
            "python": environment["python"],
            "platform": environment["platform"],
            "engine_version": ENGINE_VERSION,
            "runner_sha256": runner_hash,
            "engine_source_sha256": engine_hash,
            "dependencies": environment["dependencies"],
            "workers": 1,
            "wall_timeout_seconds": timeout,
            "process_tree_rss_limit_bytes": memory_bytes,
            "rss_sampling_interval_ms": 50,
            "weight_payload_loading_enabled": False,
            "forward_execution_enabled": False,
        },
        "observed_invariants": {
            "zero_weight_payload_bytes": model_matrix.summarize_observation(results, "weight_payload_bytes_read", 0),
            "forward_not_executed": model_matrix.summarize_observation(results, "forward_executed", False),
        },
        "results": results,
    }


def _load_capture_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise ValueError("capture record file is missing")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line_no == len(lines) and not line.endswith(("\n", "\r")):
            # A concurrent capture may be in the middle of appending its last row.
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid capture record at line {line_no}") from None
        model_id = row.get("model_id") if isinstance(row, dict) else None
        if not isinstance(model_id, str):
            raise TypeError(f"capture record at line {line_no} has no model identity")
        records[model_id] = row
    return records


def _load_report(case: dict[str, Any], row: dict[str, Any], report_directory: Path) -> tuple[bytes, dict]:
    if row.get("status") not in SUCCESS_STATUSES or row.get("service_contract_valid") is not True:
        raise ValueError("case has no validated successful report")
    digest = row.get("report_digest")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("report digest is invalid")
    path = report_directory / f"{digest}.json"
    data = path.read_bytes()
    if (len(data) > SERVICE_STORAGE_LIMIT_BYTES or len(data) != row.get("serialized_report_bytes")
            or hashlib.sha256(data).hexdigest() != digest):
        raise ValueError("report content failed digest or size validation")
    from model_anatomy.service.jobs import request_selector
    from model_anatomy.src.ir import ModelIR
    from model_anatomy.src.version import ENGINE_VERSION

    document = json.loads(data)
    ir = ModelIR.from_json(data.decode("utf-8"))
    ir.root_graph.validate()
    target = case.get("reference") or case["model_id"]
    model_id, requested_revision, _artifact, _explicit = request_selector(target)
    expected_revision = case["revision"] if requested_revision == "main" else requested_revision
    if (
        row.get("model_id") != case["model_id"]
        or row.get("category") != case["category"]
        or row.get("revision") != case["revision"]
        or document.get("model_id") != case["model_id"]
        or ir.model_id != case["model_id"]
        or model_id != case["model_id"]
        or ir.metadata.get("revision") != case["revision"]
        or expected_revision != case["revision"]
        or ir.metadata.get("engine_version") != row.get("engine_version")
        or row.get("engine_version") != ENGINE_VERSION
        or document.get("schema_version") != row.get("report_roundtrip", {}).get("schema_version")
        or document.get("schema_version") != ir.schema_version
        or ir.to_dict() != document
        or row.get("report_roundtrip", {}).get("valid") is not True
        or row.get("storage_contract_valid") is not True
        or row.get("weight_payload_bytes_read") != 0
        or row.get("forward_executed") is not False
    ):
        raise ValueError("report identity or validation evidence disagrees with pinned case")
    return data, document


def store_only(
    manifest_path: Path, record_path: Path, report_directory: Path, env_file: Path, progress_path: Path
) -> dict[str, int]:
    """Persist captured reports using the private registry and content-addressed blob tables."""
    from scripts.model_matrix import validate_manifest
    from scripts.supabase_profile import load_supabase_profile

    from model_anatomy.api.settings import Settings
    from model_anatomy.service.database import Blob, CacheEntry, Database, ModelReport
    from model_anatomy.service.jobs import cache_key, ir_summary, request_selector
    from model_anatomy.service.storage import storage_for

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = validate_manifest(manifest, allow_small=len(manifest.get("cases", [])) < 100)
    records = _load_capture_records(record_path)
    profile = load_supabase_profile(env_file)
    settings_values = {
        "database_url": profile["DATABASE_URL"],
        "supabase_url": profile["MODEL_ANATOMY_SUPABASE_URL"],
        "supabase_service_role_key": profile["MODEL_ANATOMY_SUPABASE_KEY"],
        "storage_backend": "supabase",
        "supabase_bucket": profile["MODEL_ANATOMY_SUPABASE_BUCKET"],
    }
    settings = Settings.model_validate(settings_values)
    if settings.storage_backend != "supabase" or settings.resolved_database_url.startswith("sqlite:"):
        raise ValueError("store-only requires PostgreSQL and private Supabase Storage")
    database = Database(settings.resolved_database_url)
    storage = storage_for(settings)
    counts = {"stored": 0, "reused": 0, "skipped": 0, "failed": 0}
    try:
        try:
            database.initialize(create=False)
            storage.check()
        except Exception as exc:
            append_record(
                progress_path,
                {"model_id": None, "status": "error", "stage": "preflight", "error_type": type(exc).__name__},
            )
            raise
        for case in cases:
            row = records.get(case["model_id"])
            if row is None or row.get("status") not in SUCCESS_STATUSES:
                counts["skipped"] += 1
                reason = "not_captured" if row is None else "no_validated_report"
                append_record(
                    progress_path,
                    {
                        "model_id": case["model_id"],
                        "status": "skipped",
                        "stage": "capture_check",
                        "reason": reason,
                        "captured_status": row.get("status") if row else None,
                    },
                )
                continue
            stage = "report_validation"
            try:
                data, document = _load_report(case, row, report_directory)
                stage = "identity"
                target = case.get("reference") or case["model_id"]
                model_id, requested_revision, artifact, explicit = request_selector(target)
                if requested_revision == "main":
                    requested_revision = case["revision"]
                key = cache_key(
                    model_id, case["revision"], settings.engine_version, settings.schema_version, artifact, explicit
                )
                registry_id = hashlib.sha256(f"{key}:{requested_revision}".encode()).hexdigest()
                digest = hashlib.sha256(data).hexdigest()
                summary = ir_summary(document)
                if summary["result_status"] != row["status"]:
                    raise ValueError("report status disagrees with benchmark evidence")
                if document.get("schema_version") != settings.schema_version:
                    raise ValueError("report schema version disagrees with service configuration")
                with database.read() as session:
                    existing = session.get(ModelReport, registry_id)
                    reusable = bool(
                        existing
                        and existing.cache_identity == key
                        and existing.model_id == model_id
                        and existing.request_revision == requested_revision
                        and existing.revision == case["revision"]
                        and existing.engine_version == settings.engine_version
                        and existing.schema_version == settings.schema_version
                        and existing.artifact == artifact
                        and existing.explicit_artifact == explicit
                        and existing.blob_digest == digest
                    )
                    existing_digest = existing.blob_digest if reusable else None
                if reusable:
                    try:
                        reusable = storage.get(existing_digest) == data
                    except Exception:  # noqa: BLE001 - details are deliberately excluded from progress records
                        reusable = False
                if not reusable:
                    stage = "blob_reservation"
                    with database.write() as session:
                        blob = session.get(Blob, digest)
                        if blob and blob.deleting:
                            raise ValueError("report blob is being retired")
                        if blob is None:
                            session.add(Blob(digest=digest, created_at=time.time(), deleting=False))
                    stage = "storage_upload"
                    stored_digest = storage.put(data)
                    stage = "storage_readback"
                    if stored_digest != digest or storage.get(digest) != data:
                        raise ValueError("private storage readback failed integrity validation")
                stage = "registry_commit"
                with database.write() as session:
                    cache = session.get(CacheEntry, key)
                    expires_at = time.time() + settings.cache_ttl_seconds
                    if cache is None:
                        session.add(CacheEntry(key=key, blob_digest=digest, expires_at=expires_at))
                    else:
                        cache.blob_digest, cache.expires_at = digest, expires_at
                    values = {
                        "cache_identity": key,
                        "model_id": model_id,
                        "request_revision": requested_revision,
                        "revision": case["revision"],
                        "engine_version": settings.engine_version,
                        "schema_version": settings.schema_version,
                        "artifact": artifact,
                        "explicit_artifact": explicit,
                        "blob_digest": digest,
                        "summary": summary,
                        "inspected_at": time.time(),
                    }
                    report = session.get(ModelReport, registry_id)
                    if report is None:
                        session.add(ModelReport(id=registry_id, **values))
                    else:
                        for field, value in values.items():
                            setattr(report, field, value)
                outcome = "reused" if reusable else "stored"
                counts[outcome] += 1
                append_record(progress_path, {"model_id": model_id, "status": row["status"], "stage": "complete", "outcome": outcome, "report_digest": digest})
                print(f"{outcome} report {model_id} [{row['status']}]")
            except Exception as exc:  # noqa: BLE001 - safe bounded per-model diagnostics
                counts["failed"] += 1
                append_record(progress_path, {"model_id": case["model_id"], "status": "error", "stage": stage, "error_type": type(exc).__name__})
                print(f"failed report {case['model_id']} [{stage}: {type(exc).__name__}]", file=sys.stderr, flush=True)
    finally:
        database.engine.dispose()
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture-only", action="store_true", help="capture validated IR JSON to local files")
    mode.add_argument("--store-only", action="store_true", help="import previously captured IRs into Supabase")
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/model-matrix-250.json"))
    parser.add_argument("--output", type=Path, default=Path("output/benchmark-reports.ndjson"))
    parser.add_argument("--report-dir", type=Path, default=Path("output/benchmark-reports"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--env-file", type=Path, default=Path("infra/.env"))
    parser.add_argument("--store-log", type=Path, default=Path("output/benchmark-report-store.ndjson"))
    args = parser.parse_args(argv)
    try:
        if args.capture_only:
            summary = capture_only(
                args.manifest, args.output, args.report_dir,
                timeout=args.timeout, memory_bytes=int(args.memory_gib * 1024**3),
                allow_small=args.allow_small, resume=args.resume, limit=args.limit,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        counts = store_only(args.manifest, args.output, args.report_dir, args.env_file, args.store_log)
        print(json.dumps(counts, sort_keys=True))
        return 0 if counts["failed"] == 0 else 1
    except Exception as exc:  # noqa: BLE001 - error text may include remote details; show type only
        print(f"store_benchmark_reports: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
