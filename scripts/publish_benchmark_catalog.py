"""Publish a checked 250-case run as public catalog evidence, never saved analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_matrix import COVERAGE_KEYS, case_identity, validate_manifest

from model_anatomy.src.artifact_inspection import INSPECTION_MESSAGES

SUCCESS = {"completed", "partial", "inspected"}
STATUSES = SUCCESS | {
    "http_access",
    "network",
    "artifact_error",
    "unsupported",
    "invalid_report",
    "failure",
    "memorylimit",
    "timeout",
}
PUBLIC_FIELDS = (
    "model_id",
    "category",
    "revision",
    "status",
    "architecture",
    "engine_version",
    "started_at",
    "finished_at",
    "warning_count",
    "graph_count",
    "node_count",
    "tensor_count",
    "parameter_counts",
    "coverage",
    "wall_seconds",
    "service_contract_valid",
    "case_identity",
)


def _build_run(manifest_bytes: bytes, summary: dict, *, expected_cases: int) -> dict:
    manifest = json.loads(manifest_bytes)
    cases = validate_manifest(manifest, allow_small=expected_cases < 100)
    if expected_cases < 1 or len(cases) != expected_cases:
        raise ValueError(f"catalog requires exactly {expected_cases} distinct pinned cases")
    if len({c["model_id"].casefold() for c in cases}) != expected_cases:
        raise ValueError("catalog repository IDs must be distinct ignoring case")
    rows = summary.get("results", [])
    if summary.get("cases") != expected_cases or len(rows) != expected_cases:
        raise ValueError(f"catalog requires {expected_cases} measured results")
    environment = summary["environment"]
    engine = environment["engine_version"]
    runner_hash = environment["runner_sha256"]
    source_hash = environment["engine_source_sha256"]
    identity_environment = {
        "timeout_seconds": environment["wall_timeout_seconds"],
        "memory_bytes": environment["process_tree_rss_limit_bytes"],
        "python": environment["python"],
        "platform": environment["platform"],
        "dependencies": environment["dependencies"],
    }
    published = []
    counts = Counter()
    for case, row in zip(cases, rows, strict=True):
        for key in ("model_id", "category", "revision"):
            if row.get(key) != case[key]:
                raise ValueError(f"measured {key} differs from pinned manifest: {case['model_id']}")
        expected_identity = case_identity(case, engine, runner_hash, source_hash, identity_environment)
        if row.get("case_identity") != expected_identity:
            raise ValueError(f"run identity mismatch: {case['model_id']}")
        if row.get("runner_sha256") != runner_hash or row.get("engine_source_sha256") != source_hash:
            raise ValueError("mixed runner or engine sources")
        status = row.get("status")
        if status not in STATUSES:
            raise ValueError(f"unknown outcome: {status}")
        for key in ("started_at", "finished_at"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("every attempt must have timestamps")
        if status in SUCCESS:
            if row.get("engine_version") != engine:
                raise ValueError("mixed engine versions")
            if not all(row.get(key) is True for key in ("service_contract_valid", "storage_contract_valid")):
                raise ValueError("successful result lacks validated service/storage contract")
            if row.get("weight_payload_bytes_read") != 0 or row.get("forward_executed") is not False:
                raise ValueError("successful result lacks static-inspection invariants")
            if row.get("report_roundtrip", {}).get("valid") is not True:
                raise ValueError("successful result lacks report validation")
            if status == "inspected":
                if row.get("architecture") is not None or row.get("graph_count") != 0:
                    raise ValueError("artifact-only result cannot claim an architecture graph")
            elif not row.get("graph_count", 0):
                raise ValueError("architecture outcome requires measured graphs")
            if status == "completed" and (
                any(row.get("coverage", {}).get(key) != 1 for key in COVERAGE_KEYS)
                or row.get("warning_count") != 0
                or bool(row.get("warnings"))
            ):
                raise ValueError("complete report requires full bounded coverage and no warnings")
        item = {key: row[key] for key in PUBLIC_FIELDS if key in row}
        # Native blockers and warnings can include exception text or local paths.
        # Publish only stable engine reason messages and generic warning summaries.
        raw_reasons = row.get("reason_codes", [])
        safe_messages = {
            **INSPECTION_MESSAGES,
            "coverage_not_complete": "The recorded static coverage is incomplete.",
        }
        reasons = list(dict.fromkeys(reason for reason in raw_reasons if reason in safe_messages))
        notes = [safe_messages[reason] for reason in reasons]
        if status in SUCCESS and row.get("warning_count", 0):
            notes.append(
                "Static-analysis warnings were recorded. Run a pinned inspection for full diagnostics."
            )
            if any(
                "equivalence to publisher custom code is unverified" in warning
                for warning in row.get("warnings", [])
            ):
                notes.append(
                    "Publisher custom-code equivalence is unverified; the installed native implementation was used."
                )
        item["warnings"] = []
        item["reason_codes"] = reasons if status in SUCCESS else []
        item["diagnostic_notes"] = (
            notes if status in SUCCESS else ["This attempt did not produce a validated report."]
        )
        if status not in SUCCESS:
            for key in (
                "warning_count",
                "architecture",
                "graph_count",
                "node_count",
                "tensor_count",
                "parameter_counts",
                "coverage",
                "service_contract_valid",
            ):
                item.pop(key, None)
        item.update(
            {
                "engine_version": engine,
                "reference": case.get("reference"),
                "declared_task": case.get("declared_task"),
            }
        )
        # No traceback, filesystem paths, or raw exception messages in the public dataset.
        if status not in SUCCESS:
            stage = row.get("stage")
            if stage not in {"case", "reconstruction", "report_validation"}:
                stage = "case"
            item["error"] = {"message": f"{status} during {stage}"}
        published.append(item)
        counts[status] += 1
    if dict(counts) != summary.get("status_counts"):
        raise ValueError("summary counts disagree with measured rows")
    return {
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "engine_version": engine,
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        "manifest_hash_encoding": "canonical-json-v1",
        "runner_sha256": runner_hash,
        "engine_source_sha256": source_hash,
        "method": "Pinned metadata and bounded static reconstruction; no inference quality or speed measurement.",
        "status_counts": dict(counts),
        "results": published,
    }


def build_catalog(
    manifest_bytes: bytes, summary: dict, *, rechecks: list[tuple[bytes, dict]] | None = None
) -> dict:
    """Validate a complete baseline, then apply explicitly attributed newer observations."""
    catalog = _build_run(manifest_bytes, summary, expected_cases=250)
    if not rechecks:
        return catalog
    original_cases = {case["model_id"]: case for case in json.loads(manifest_bytes)["cases"]}
    latest = {row["model_id"]: row for row in catalog["results"]}

    def provenance(run: dict, kind: str) -> dict:
        environment = run["environment"]
        return {
            "kind": kind,
            "cases": run["cases"],
            "generated_at": run["generated_at"],
            "engine_version": environment["engine_version"],
            "runner_sha256": environment["runner_sha256"],
            "engine_source_sha256": environment["engine_source_sha256"],
            "status_counts": run["status_counts"],
        }

    runs = [provenance(summary, "baseline")]
    for row in latest.values():
        row["run_kind"] = "baseline"
    for manifest_data, run_summary in rechecks:
        cases = validate_manifest(json.loads(manifest_data), allow_small=True)
        for case in cases:
            if original_cases.get(case["model_id"]) != case:
                raise ValueError("rechecks must preserve the original pinned case definition")
        checked = _build_run(manifest_data, run_summary, expected_cases=len(cases))
        runs.append(provenance(run_summary, "recheck"))
        for row in checked["results"]:
            previous = latest[row["model_id"]]
            if datetime.fromisoformat(row["finished_at"]) <= datetime.fromisoformat(previous["finished_at"]):
                raise ValueError("rechecks must be newer than the observation they replace")
            row["previous_observations"] = [
                *previous.get("previous_observations", []),
                {key: previous[key] for key in ("status", "engine_version", "finished_at", "case_identity")},
            ]
            row["run_kind"] = "recheck"
            latest[row["model_id"]] = row
    catalog["results"] = [latest[model_id] for model_id in original_cases]
    catalog["status_counts"] = dict(Counter(row["status"] for row in catalog["results"]))
    catalog["runs"] = runs
    catalog["baseline_cases"] = 250
    catalog["recheck_attempts"] = sum(run["cases"] for run in runs[1:])
    catalog["engine_versions"] = list(dict.fromkeys(run["engine_version"] for run in runs))
    catalog["engine_version"] = " + ".join(catalog["engine_versions"])
    catalog["generated_at"] = runs[-1]["generated_at"]
    catalog["baseline_engine_source_sha256"] = catalog.pop("engine_source_sha256")
    catalog["baseline_runner_sha256"] = catalog.pop("runner_sha256")
    return catalog


def prepare_catalog_payload(catalog: dict, run_payloads: list[dict], manifest_path: Path) -> tuple[dict, list[dict]]:
    """Attach and cross-check the original manifest cases before durable import."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = validate_manifest(manifest)
    manifest_digest = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if catalog.get("manifest_sha256") != manifest_digest:
        raise ValueError("catalog manifest hash differs from the original pinned manifest")
    if len(cases) != 250 or len(catalog.get("results", [])) != 250:
        raise ValueError("benchmark import requires all 250 pinned manifest cases")
    for case, row in zip(cases, catalog["results"], strict=True):
        if any(row.get(key) != case.get(key) for key in ("model_id", "category", "revision")):
            raise ValueError("catalog row does not match the original pinned manifest")
    prepared = dict(catalog)
    prepared["case_definitions"] = cases
    return prepared, run_payloads


def store_catalog_only(
    catalog_path: Path,
    manifest_path: Path,
    run_paths: list[Path],
    env_file: Path,
    *,
    profile_loader=None,
    database_factory=None,
) -> str:
    """Import a prepared catalog into PostgreSQL/Supabase without touching cloud storage."""
    from types import SimpleNamespace

    from scripts.supabase_profile import load_supabase_profile

    from model_anatomy.api.settings import Settings
    from model_anatomy.service.benchmark_catalog import BenchmarkCatalogService
    from model_anatomy.service.database import Database

    load_profile = profile_loader or load_supabase_profile
    make_database = database_factory or Database
    profile = load_profile(env_file)
    settings = Settings.model_validate({
        "environment": "production", "allowed_hosts": ["localhost"], "cors_origins": ["https://localhost"],
        "database_url": profile["DATABASE_URL"], "supabase_url": profile["MODEL_ANATOMY_SUPABASE_URL"],
        "supabase_service_role_key": profile["MODEL_ANATOMY_SUPABASE_KEY"],
        "storage_backend": "supabase", "supabase_bucket": profile["MODEL_ANATOMY_SUPABASE_BUCKET"],
    })
    if settings.storage_backend != "supabase" or not settings.resolved_database_url.startswith("postgresql+"):
        raise ValueError("catalog import requires PostgreSQL and private Supabase configuration")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    run_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    prepared, run_payloads = prepare_catalog_payload(catalog, run_payloads, manifest_path)
    database = make_database(settings.resolved_database_url)
    try:
        database.initialize(create=False)
        return BenchmarkCatalogService(SimpleNamespace(db=database, settings=settings)).publish(prepared, run_payloads)
    except Exception:  # noqa: BLE001 - redact connection credentials in all provider failures
        # Exception messages may include a database host or user-provided data.
        raise RuntimeError("benchmark catalog import failed; inspect local service diagnostics") from None
    finally:
        database.engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-only", action="store_true", help="store an existing catalog and run histories in Supabase")
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/model-matrix-250.json"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("frontend/public/benchmark-catalog.json"))
    parser.add_argument("--recheck-manifest", type=Path, action="append", default=[])
    parser.add_argument("--recheck-summary", type=Path, action="append", default=[])
    parser.add_argument("--env-file", type=Path, default=Path("infra/.env"))
    parser.add_argument("--catalog", type=Path, default=Path("frontend/public/benchmark-catalog.json"))
    parser.add_argument(
        "--runs", nargs="+", type=Path,
        default=[Path("docs/benchmarks/model-matrix-250-baseline.json"), Path("docs/benchmarks/model-matrix-250-rechecks.json")],
    )
    args = parser.parse_args()
    if args.store_only:
        try:
            publication_id = store_catalog_only(args.catalog, args.manifest, args.runs, args.env_file)
            print(f"Stored benchmark publication {publication_id}")
            return
        except Exception as exc:  # noqa: BLE001 - error text may contain credentials or remote diagnostics
            print(f"Benchmark catalog import failed: {type(exc).__name__}", file=sys.stderr)
            raise SystemExit(2) from None
    if args.summary is None:
        parser.error("--summary is required unless --store-only is selected")
    if len(args.recheck_manifest) != len(args.recheck_summary):
        parser.error("each recheck manifest requires a matching recheck summary")
    rechecks = [
        (manifest.read_bytes(), json.loads(summary.read_text(encoding="utf-8")))
        for manifest, summary in zip(args.recheck_manifest, args.recheck_summary, strict=True)
    ]
    catalog = build_catalog(
        args.manifest.read_bytes(), json.loads(args.summary.read_text(encoding="utf-8")), rechecks=rechecks
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Published {len(catalog['results'])} measured cases: {catalog['status_counts']}")


if __name__ == "__main__":
    main()
