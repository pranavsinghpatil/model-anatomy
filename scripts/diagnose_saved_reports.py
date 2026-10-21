"""Build a safe, reproducible blocker inventory from integrity-verified saved reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_model_coverage import (
    ARTIFACT_COHORTS,
    COVERAGE_METRICS,
    SUCCESS_STATUSES,
    AuditInputError,
    _read_records,
    _read_verified_report,
)

WARNING_MARKERS = {
    "unresolved_branch": "Conditional source path is unresolved",
    "opaque_call": "Opaque call",
    "unresolved_loop": "Loop could not be statically resolved",
    "unmodeled_exception_path": "Try control flow is partially modeled",
    "unowned_checkpoint_tensor": "checkpoint tensors have no native module owner",
    "unproven_tensor_transform": "Tensor transform ",
}
OPERATION_COHORTS = {
    "numerical_stability": ("torch.clamp", "torch.isinf", "torch.where"),
    "attention_or_mask_dispatch": ("attention_interface", "ATTENTION_FUNCTION", "mask_function"),
    "tuple_split_or_chunk": (".chunk", ".split", ".unbind"),
    "padding_windows_or_resampling": (".pad", ".interpolate", "torch.roll", "window_partition", "window_reverse"),
    "reduction_or_normalization": (".norm", ".mean", ".var", ".sqrt", ".exp", ".t:"),
    "position_ids": ("create_position_ids",),
}
FAILURE_CATEGORIES = {"http_access", "network", "artifact_error", "unsupported", "invalid_report", "memorylimit", "timeout"}


def _count(value, label):
    if type(value) is not int or value < 0:
        raise AuditInputError(f"invalid report {label}")
    return value


def _case(row: dict, directory: Path) -> dict:
    case = {key: row[key] for key in ("model_id", "revision")}
    status = row["status"]
    case["status"] = status if status in SUCCESS_STATUSES | FAILURE_CATEGORIES else "failure"
    if status not in SUCCESS_STATUSES:
        return {**case, "completion_basis": "no_validated_report", "stage": row["stage"], "cause_type": row["cause_type"]}
    report = _read_verified_report(directory, row)
    metadata = report["metadata"]
    engine = row.get("engine_version")
    if (not isinstance(engine, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", engine)
            or metadata.get("engine_version") != engine):
        raise AuditInputError("report engine differs from capture")
    analysis = metadata.get("analysis_report")
    if not isinstance(analysis, dict):
        raise AuditInputError("report analysis is missing")
    warnings = analysis.get("warnings", [] if status == "inspected" else None)
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise AuditInputError("invalid report warnings")
    if _count(row.get("warning_count"), "warning count") != len(warnings):
        raise AuditInputError("report warning count differs from capture")
    case.update(engine_version=row["engine_version"], report_digest=row["report_digest"], warning_count=len(warnings))
    if status == "inspected":
        inspection = metadata.get("inspection")
        if not isinstance(inspection, dict) or inspection.get("kind") != "artifact_inspection":
            raise AuditInputError("report inspection is missing")
        reason = inspection.get("reason_code")
        case.update(completion_basis="artifacts_only", artifact_reason=reason if reason in ARTIFACT_COHORTS else "other")
        return case
    coverage = {key: analysis.get(key) for key in COVERAGE_METRICS}
    capture_coverage = row.get("coverage")
    if not isinstance(capture_coverage, dict):
        raise AuditInputError("capture coverage is not an object")
    for key, value in coverage.items():
        if value is not None and (type(value) not in (int, float) or not 0 <= value <= 1):
            raise AuditInputError("invalid report coverage")
        observed = capture_coverage.get(key)
        if observed is not None and (type(observed) not in (int, float) or not 0 <= observed <= 1):
            raise AuditInputError("invalid capture coverage")
        if value != observed:
            raise AuditInputError("report coverage differs from capture")
    unknown = _count(analysis.get("unknown_nodes"), "unknown node count")
    unmapped = analysis.get("unmapped_checkpoint_tensors")
    if not isinstance(unmapped, list) or any(not isinstance(item, str) for item in unmapped):
        raise AuditInputError("invalid report unmapped tensors")
    gaps = [key for key, value in coverage.items() if value != 1]
    warning_counts = {name: sum(marker in warning for warning in warnings) for name, marker in WARNING_MARKERS.items()}
    warning_counts["other"] = sum(not any(marker in warning for marker in WARNING_MARKERS.values()) for warning in warnings)
    operations = [name for name, markers in OPERATION_COHORTS.items() if any(
        "Opaque call" in warning and any(marker in warning for marker in markers) for warning in warnings
    )]
    basis = "coverage_or_ownership_gap" if gaps or unknown or unmapped else "source_warnings_only" if warnings else "recorded_static_gates_passed"
    if (status == "completed") != (basis == "recorded_static_gates_passed"):
        raise AuditInputError("report status disagrees with recorded completion gates")
    return {**case, "completion_basis": basis, "coverage": coverage, "coverage_gaps": gaps,
            "unknown_nodes": unknown, "unmapped_checkpoint_tensor_count": len(unmapped),
            "warning_counts": warning_counts, "operation_warning_cohorts": operations}


def build_diagnosis(sources: list[tuple[Path, Path]]) -> dict:
    """Overlay pinned observations in explicit source order; emit no raw warning/error prose."""
    if not sources:
        raise AuditInputError("no diagnosis sources")
    latest = {}
    provenance = []
    for records, directory in sources:
        count, rows = _read_records(records)
        provenance.append({"records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
                           "record_count": count, "unique_model_count": len(rows)})
        for model_id, row in rows.items():
            if model_id in latest and row["revision"] != latest[model_id][0]["revision"]:
                raise AuditInputError("overlaid observation changes the model pin")
            latest[model_id] = (row, directory)
    cases = [_case(row, directory) for row, directory in latest.values()]
    partial = [case for case in cases if case["status"] == "partial"]
    return {
        "schema_version": 1,
        "method": "Integrity-verified saved report diagnostics; no reconstruction, inference, or new support claim.",
        "source_order": provenance,
        "case_count": len(cases),
        "verified_report_count": sum(case["status"] in SUCCESS_STATUSES for case in cases),
        "status_counts": dict(Counter(case["status"] for case in cases)),
        "report_engine_counts": dict(Counter(case["engine_version"] for case in cases if "engine_version" in case)),
        "partial_basis_counts": dict(Counter(case["completion_basis"] for case in partial)),
        "overlapping_partial_warning_model_counts": {
            key: sum(case["warning_counts"][key] > 0 for case in partial) for key in [*WARNING_MARKERS, "other"]
        },
        "overlapping_partial_operation_warning_model_counts": {
            key: sum(key in case["operation_warning_cohorts"] for case in partial) for key in OPERATION_COHORTS
        },
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("RECORDS", "REPORT_DIR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_diagnosis([(Path(records), Path(directory)) for records, directory in args.source])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (AuditInputError, OSError):
        print("Diagnosis failed: capture/report integrity, validation, or local file access error", file=sys.stderr)
        return 2
    print(f"Diagnosed {result['case_count']} cases, {result['verified_report_count']} verified reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
