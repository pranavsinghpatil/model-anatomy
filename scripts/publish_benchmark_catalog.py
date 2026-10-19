"""Publish a checked 250-case run as public catalog evidence, never saved analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.model_matrix import COVERAGE_KEYS, case_identity, validate_manifest

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
    "warnings",
    "reason_codes",
    "graph_count",
    "node_count",
    "tensor_count",
    "parameter_counts",
    "coverage",
    "wall_seconds",
    "service_contract_valid",
    "case_identity",
)


def build_catalog(manifest_bytes: bytes, summary: dict) -> dict:
    manifest = json.loads(manifest_bytes)
    cases = validate_manifest(manifest)
    if len(cases) != 250:
        raise ValueError("catalog requires exactly 250 distinct pinned cases")
    if len({c["model_id"].casefold() for c in cases}) != 250:
        raise ValueError("catalog repository IDs must be distinct ignoring case")
    rows = summary.get("results", [])
    if summary.get("cases") != 250 or len(rows) != 250:
        raise ValueError("catalog requires 250 measured results")
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
        if status not in SUCCESS:
            for key in (
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
            item["error"] = {
                "message": f"{status} during {row.get('stage', 'case')}",
                "cause_type": row.get("cause_type"),
            }
        published.append(item)
        counts[status] += 1
    if dict(counts) != summary.get("status_counts"):
        raise ValueError("summary counts disagree with measured rows")
    return {
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "engine_version": engine,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runner_sha256": runner_hash,
        "engine_source_sha256": source_hash,
        "method": "Pinned metadata and bounded static reconstruction; no inference quality or speed measurement.",
        "status_counts": dict(counts),
        "results": published,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/model-matrix-250.json"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("frontend/public/benchmark-catalog.json"))
    args = parser.parse_args()
    catalog = build_catalog(args.manifest.read_bytes(), json.loads(args.summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Published {len(catalog['results'])} measured cases: {catalog['status_counts']}")


if __name__ == "__main__":
    main()
