"""Summarize model reconstruction coverage from a captured NDJSON run.

The audit emits only aggregate evidence and a deliberately small failure
projection. It never copies arbitrary report strings, URLs, or file paths into
the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SUCCESS_STATUSES = {"completed", "partial", "inspected"}
COVERAGE_METRICS = (
    "semantic_coverage",
    "module_semantic_coverage",
    "checkpoint_parameter_coverage",
    "tensor_shape_coverage",
)
ARTIFACT_COHORTS = (
    "pipeline_components_require_selection",
    "checkpoint_selection_required",
    "safetensors_unavailable",
    "native_architecture_unavailable",
    "architecture_config_unavailable",
    "gguf_header_only",
    "metadata_budget_exceeded",
    "adapter_requires_base_model",
    "other",
)
_PIN = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_TYPE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class AuditInputError(ValueError):
    """The capture or an optional report is malformed or incomplete."""


def _read_records(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    record_count = 0
    latest: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AuditInputError("could not read NDJSON records") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditInputError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise AuditInputError(f"line {line_number}: expected a JSON object")
        model_id = row.get("model_id")
        status = row.get("status")
        revision = row.get("revision")
        if not isinstance(model_id, str) or not model_id.strip():
            raise AuditInputError(f"line {line_number}: missing model_id")
        if (not _MODEL_ID.fullmatch(model_id) or len(model_id) > 256
                or any(part in {".", ".."} for part in model_id.split("/"))):
            raise AuditInputError(f"line {line_number}: invalid model_id")
        if not isinstance(status, str) or not status.strip():
            raise AuditInputError(f"line {line_number}: missing status")
        if not isinstance(revision, str) or not _PIN.fullmatch(revision):
            raise AuditInputError(f"line {line_number}: missing or invalid revision pin")

        if status == "partial":
            coverage = row.get("coverage")
            warnings = row.get("warnings")
            if not isinstance(coverage, dict):
                raise AuditInputError(f"line {line_number}: partial record is missing coverage")
            if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
                raise AuditInputError(f"line {line_number}: partial record has invalid warnings")
            for metric in COVERAGE_METRICS:
                if metric not in coverage:
                    raise AuditInputError(f"line {line_number}: partial record is missing {metric}")
                value = coverage[metric]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 1
                ):
                    raise AuditInputError(f"line {line_number}: invalid {metric}")
        elif status == "inspected":
            reason_codes = row.get("reason_codes")
            if not isinstance(reason_codes, list) or any(
                not isinstance(item, str) for item in reason_codes
            ):
                raise AuditInputError(f"line {line_number}: inspected record has invalid reason_codes")
        elif status not in SUCCESS_STATUSES:
            if not isinstance(row.get("stage"), str) or not row["stage"].strip():
                raise AuditInputError(f"line {line_number}: failed record is missing stage")
            if not isinstance(row.get("cause_type"), str) or not row["cause_type"].strip():
                raise AuditInputError(f"line {line_number}: failed record is missing cause_type")
            if not _TYPE_NAME.fullmatch(row["stage"]) or not _TYPE_NAME.fullmatch(row["cause_type"]):
                raise AuditInputError(f"line {line_number}: invalid failure classification")

        record_count += 1
        latest[model_id] = row

    if not latest:
        raise AuditInputError("NDJSON contains no model records")
    return record_count, latest


def _safe_reason_codes(row: dict[str, Any]) -> set[str]:
    """Keep only machine-like codes; capture prose can contain sensitive text."""
    return {
        item
        for item in row.get("reason_codes", [])
        if isinstance(item, str) and _SAFE_REASON_CODE.fullmatch(item)
    }


def _read_verified_report(report_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    digest = row.get("report_digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise AuditInputError("record is missing a valid report digest")
    report_path = report_dir / f"{digest}.json"
    try:
        data = report_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise AuditInputError("a referenced report failed digest verification")
        report = json.loads(data)
    except FileNotFoundError as exc:
        raise AuditInputError("a referenced report JSON file is missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditInputError("a referenced report JSON file is unreadable or invalid") from exc
    if not isinstance(report, dict):
        raise AuditInputError("a referenced report JSON value is not an object")
    metadata = report.get("metadata")
    if (report.get("model_id") != row["model_id"] or not isinstance(metadata, dict)
            or metadata.get("revision") != row["revision"]):
        raise AuditInputError("a referenced report differs from its pinned case")
    return report


def _read_report_reason(report_dir: Path, row: dict[str, Any]) -> str | None:
    metadata = _read_verified_report(report_dir, row)["metadata"]
    inspection = metadata.get("inspection") if isinstance(metadata, dict) else None
    if inspection is None:
        return None
    if not isinstance(inspection, dict):
        raise AuditInputError("report inspection metadata is malformed")
    reason = inspection.get("reason_code")
    if reason is None:
        return None
    if not isinstance(reason, str) or not _SAFE_REASON_CODE.fullmatch(reason):
        raise AuditInputError("report inspection reason_code is not a safe machine code")
    return reason


def _artifact_cohort(codes: set[str]) -> str:
    for cohort in ARTIFACT_COHORTS[:-1]:
        if cohort in codes:
            return cohort
    return "other"


def build_audit(records_path: Path, report_dir: Path | None = None) -> dict[str, Any]:
    """Build a deterministic audit from the latest row for each model ID."""
    record_count, latest = _read_records(records_path)
    statuses = Counter(
        status if status in SUCCESS_STATUSES else "failed"
        for status in (row["status"] for row in latest.values())
    )
    partial_rows = [row for row in latest.values() if row["status"] == "partial"]
    warning_sets = []
    for row in partial_rows:
        warnings = row["warnings"]
        if report_dir is not None:
            report = _read_verified_report(report_dir, row)
            analysis = report["metadata"].get("analysis_report")
            warnings = analysis.get("warnings") if isinstance(analysis, dict) else None
            if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
                raise AuditInputError("a partial report has invalid analysis warnings")
        warning_sets.append(warnings)
    partial_cohorts = {
        "unresolved_source_branches": sum(
            any("Conditional source path is unresolved" in warning for warning in warnings)
            for warnings in warning_sets
        ),
        "opaque_calls": sum(
            any("Opaque call" in warning for warning in warnings)
            for warnings in warning_sets
        ),
    }
    metric_cohorts = {
        metric: sum(
            row["coverage"][metric] is None or row["coverage"][metric] < 1
            for row in partial_rows
        )
        for metric in COVERAGE_METRICS
    }

    inspected_rows = [row for row in latest.values() if row["status"] == "inspected"]
    artifact_counts = Counter()
    detail_counts = Counter()
    for row in inspected_rows:
        codes = _safe_reason_codes(row)
        if report_dir is not None:
            report_reason = _read_report_reason(report_dir, row)
            if report_reason is not None:
                codes.add(report_reason)
                detail_counts[report_reason] += 1
        artifact_counts[_artifact_cohort(codes)] += 1

    failures = [
        {
            "model_id": row["model_id"],
            "pin": row["revision"],
            "stage": row["stage"],
            "cause_type": row["cause_type"],
        }
        for row in latest.values()
        if row["status"] not in SUCCESS_STATUSES
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "record_count": record_count,
        "unique_model_count": len(latest),
        "status_counts": {
            status: statuses.get(status, 0)
            for status in ("completed", "partial", "inspected", "failed")
        },
        "partial": {
            "model_count": len(partial_rows),
            "warning_evidence": "verified_full_reports" if report_dir is not None else "capture_records",
            "overlapping_warning_cohorts": partial_cohorts,
            "incomplete_coverage_metric_counts": metric_cohorts,
        },
        "artifact_inspection": {
            "model_count": len(inspected_rows),
            "mutually_exclusive_reason_cohorts": {
                cohort: artifact_counts.get(cohort, 0) for cohort in ARTIFACT_COHORTS
            },
        },
        "failures": failures,
    }
    if report_dir is not None:
        result["artifact_inspection"]["report_reason_code_counts"] = dict(sorted(detail_counts.items()))
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="input capture NDJSON")
    parser.add_argument("--report-dir", type=Path, help="optional local directory of digest-named report JSON")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path, or - for stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = build_audit(args.records, args.report_dir)
    except AuditInputError as exc:
        print(f"audit input error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if str(args.output) == "-":
        sys.stdout.write(rendered)
    else:
        try:
            args.output.write_text(rendered, encoding="utf-8")
        except OSError:
            print("audit output error: could not write JSON output", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
