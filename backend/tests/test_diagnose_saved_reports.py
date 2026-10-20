"""Saved diagnostics distinguish warning closure from coverage and verify report identity."""

import hashlib
import json

import pytest
from scripts.audit_model_coverage import COVERAGE_METRICS, AuditInputError
from scripts.diagnose_saved_reports import build_diagnosis, main


def source(tmp_path, *, name="run", status="partial", warnings=None, gaps=None, unknown=0, unmapped=None,
           engine="engine-3", report_changes=None, row_changes=None):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    warnings = ["Opaque call torch.clamp: raw-private-warning"] if warnings is None else warnings
    coverage = {key: 1.0 for key in COVERAGE_METRICS}
    coverage.update(gaps or {})
    analysis = {**coverage, "warnings": warnings, "unknown_nodes": unknown,
                "unmapped_checkpoint_tensors": unmapped or []}
    report = {"model_id": "owner/model", "metadata": {
        "revision": "a" * 40, "engine_version": engine, "analysis_report": analysis,
        "inspection": {"kind": "artifact_inspection", "reason_code": "native_architecture_unavailable"},
    }}
    if report_changes:
        report_changes(report)
    payload = json.dumps(report).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (directory / f"{digest}.json").write_bytes(payload)
    row = {"model_id": "owner/model", "revision": "a" * 40, "status": status,
           "coverage": coverage, "warnings": warnings[:1], "warning_count": len(warnings),
           "engine_version": engine, "report_digest": digest, "reason_codes": []}
    row.update(row_changes or {})
    records = directory / "capture.ndjson"
    records.write_text(json.dumps(row) + "\n")
    return records, directory


def test_full_ratios_with_opaque_operations_are_still_warning_only_partial(tmp_path):
    result = build_diagnosis([source(tmp_path)])
    case = result["cases"][0]
    assert case["completion_basis"] == "source_warnings_only"
    assert case["coverage_gaps"] == []
    assert case["operation_warning_cohorts"] == ["numerical_stability"]
    assert result["partial_basis_counts"] == {"source_warnings_only": 1}
    assert "raw-private-warning" not in json.dumps(result)


@pytest.mark.parametrize("options", [
    {"gaps": {"tensor_shape_coverage": 0.6}},
    {"unknown": 1},
    {"unmapped": ["private.tensor.name"]},
])
def test_missing_shapes_unknown_nodes_and_extra_tensors_are_independent_blockers(tmp_path, options):
    case = build_diagnosis([source(tmp_path, **options)])["cases"][0]
    assert case["completion_basis"] == "coverage_or_ownership_gap"
    assert "private.tensor.name" not in json.dumps(case)


def test_complete_and_artifact_only_results_keep_distinct_gates(tmp_path):
    complete = build_diagnosis([source(tmp_path, status="completed", warnings=[])])["cases"][0]
    inspected = build_diagnosis([source(tmp_path, name="artifacts", status="inspected", warnings=[])])["cases"][0]
    assert complete["completion_basis"] == "recorded_static_gates_passed"
    assert inspected["completion_basis"] == "artifacts_only"
    assert inspected["artifact_reason"] == "native_architecture_unavailable"
    assert "coverage" not in inspected


def test_artifact_only_schema_does_not_require_architecture_warnings(tmp_path):
    case = build_diagnosis([source(tmp_path, status="inspected", warnings=[],
        report_changes=lambda d: d["metadata"]["analysis_report"].pop("warnings"))])["cases"][0]
    assert case["completion_basis"] == "artifacts_only" and case["warning_count"] == 0


def test_artifact_only_capture_cannot_relabel_an_architecture_report(tmp_path):
    with pytest.raises(AuditInputError, match="inspection"):
        build_diagnosis([source(tmp_path, status="inspected", warnings=[],
            report_changes=lambda d: d["metadata"].pop("inspection"))])


def test_malformed_complete_coverage_container_uses_safe_cli_error(tmp_path, capsys):
    records, directory = source(tmp_path, status="completed", warnings=[], row_changes={"coverage": []})
    with pytest.raises(AuditInputError, match="coverage is not an object"):
        build_diagnosis([(records, directory)])
    assert main(["--source", str(records), str(directory), "--output", str(tmp_path / "out")]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_boolean_capture_ratios_cannot_masquerade_as_numeric_one(tmp_path):
    with pytest.raises(AuditInputError, match="capture coverage"):
        build_diagnosis([source(tmp_path, status="completed", warnings=[],
            row_changes={"coverage": {key: True for key in COVERAGE_METRICS}})])


def test_recheck_overlay_preserves_pin_and_actual_engine_identity(tmp_path):
    old = source(tmp_path, name="old", engine="engine-2")
    new = source(tmp_path, name="new", status="completed", warnings=[])
    result = build_diagnosis([old, new])
    assert result["case_count"] == result["verified_report_count"] == 1
    assert result["report_engine_counts"] == {"engine-3": 1}
    changed_pin = source(tmp_path, name="wrong", row_changes={"revision": "b" * 40})
    with pytest.raises(AuditInputError, match="changes the model pin"):
        build_diagnosis([old, changed_pin])


@pytest.mark.parametrize("changes", [
    {"model_id": "owner/other"},
    {"revision": "b" * 40},
    {"engine_version": "engine-2"},
    {"warning_count": 7},
    {"coverage": {key: 0.5 for key in COVERAGE_METRICS}},
])
def test_capture_and_report_identity_or_metric_mismatches_fail(tmp_path, changes):
    with pytest.raises(AuditInputError):
        build_diagnosis([source(tmp_path, row_changes=changes)])


def test_digest_tampering_is_rejected(tmp_path):
    records, directory = source(tmp_path)
    row = json.loads(records.read_text())
    (directory / f"{row['report_digest']}.json").write_text("{}")
    with pytest.raises(AuditInputError, match="digest"):
        build_diagnosis([(records, directory)])


@pytest.mark.parametrize("changes", [
    lambda d: d["metadata"]["analysis_report"].update(unknown_nodes=True),
    lambda d: d["metadata"]["analysis_report"].update(unmapped_checkpoint_tensors="unsafe raw prose"),
    lambda d: d["metadata"]["analysis_report"].update(warnings="unsafe raw prose"),
])
def test_malformed_analysis_values_fail_closed(tmp_path, changes):
    with pytest.raises(AuditInputError):
        build_diagnosis([source(tmp_path, report_changes=changes)])


def test_completion_label_cannot_hide_opaque_warnings(tmp_path):
    with pytest.raises(AuditInputError, match="completion gates"):
        build_diagnosis([source(tmp_path, status="completed")])


def test_failure_projection_and_cli_error_do_not_echo_diagnostics(tmp_path, capsys):
    records = tmp_path / "failures.ndjson"
    records.write_text(json.dumps({"model_id": "owner/model", "revision": "a" * 40,
        "status": "http_access", "stage": "reconstruction", "cause_type": "HTTPError",
        "cause_message": "secret-provider-details"}) + "\n")
    result = build_diagnosis([(records, tmp_path)])
    assert result["verified_report_count"] == 0
    assert "secret-provider-details" not in json.dumps(result)
    assert main(["--source", str(tmp_path / "missing"), str(tmp_path), "--output", str(tmp_path / "out")]) == 2
    assert "secret-provider-details" not in capsys.readouterr().err
