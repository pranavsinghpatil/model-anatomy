"""Offline tests for the model coverage audit CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.audit_model_coverage import AuditInputError, build_audit, main

PIN_A = "a" * 40
PIN_B = "b" * 40
DIGEST_A = "c" * 64


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def partial_row(model_id: str, **updates) -> dict:
    row = {
        "model_id": model_id,
        "revision": PIN_A,
        "status": "partial",
        "coverage": {
            "semantic_coverage": 1.0,
            "module_semantic_coverage": 0.5,
            "checkpoint_parameter_coverage": 1.0,
            "tensor_shape_coverage": None,
        },
        "warnings": [
            "Conditional source path is unresolved: mask is dynamic",
            "Opaque call torch.where: dependencies retained",
        ],
    }
    row.update(updates)
    return row


def test_last_record_per_model_wins_and_failures_are_minimally_projected(tmp_path: Path):
    records = tmp_path / "capture.ndjson"
    write_rows(
        records,
        [
            partial_row("owner/model", revision=PIN_A),
            {
                "model_id": "owner/model",
                "revision": PIN_B,
                "status": "http_access",
                "stage": "reconstruction",
                "cause_type": "HTTPError",
                "cause_message": "must never be copied",
                "url": "https://private.example/token",
            },
        ],
    )

    audit = build_audit(records)

    assert audit["record_count"] == 2
    assert audit["unique_model_count"] == 1
    assert audit["status_counts"] == {"completed": 0, "partial": 0, "inspected": 0, "failed": 1}
    assert audit["failures"] == [
        {
            "model_id": "owner/model",
            "pin": PIN_B,
            "stage": "reconstruction",
            "cause_type": "HTTPError",
        }
    ]
    assert "cause_message" not in json.dumps(audit)
    assert "private.example" not in json.dumps(audit)


def test_artifact_inspection_cohorts_are_mutually_exclusive(tmp_path: Path):
    records = tmp_path / "capture.ndjson"
    codes = [
        ["pipeline_components_require_selection", "coverage_not_complete"],
        ["checkpoint_selection_required"],
        ["safetensors_unavailable"],
        ["native_architecture_unavailable"],
        ["adapter_requires_base_model"],
    ]
    rows = [
        {
            "model_id": f"owner/model-{index}",
            "revision": PIN_A,
            "status": "inspected",
            "reason_codes": reason_codes,
        }
        for index, reason_codes in enumerate(codes)
    ]
    write_rows(records, rows)

    cohorts = build_audit(records)["artifact_inspection"]["mutually_exclusive_reason_cohorts"]

    assert sum(cohorts.values()) == len(rows)
    assert cohorts["pipeline_components_require_selection"] == 1
    assert cohorts["checkpoint_selection_required"] == 1
    assert cohorts["safetensors_unavailable"] == 1
    assert cohorts["native_architecture_unavailable"] == 1
    assert cohorts["adapter_requires_base_model"] == 1
    assert cohorts["other"] == 0


def test_partial_warning_and_metric_cohorts_overlap(tmp_path: Path):
    records = tmp_path / "capture.ndjson"
    write_rows(
        records,
        [
            partial_row("owner/a"),
            partial_row(
                "owner/b",
                warnings=["Opaque call torch.clamp: unverified"],
                coverage={
                    "semantic_coverage": 0.9,
                    "module_semantic_coverage": 1.0,
                    "checkpoint_parameter_coverage": 1.0,
                    "tensor_shape_coverage": 1.0,
                },
            ),
        ],
    )

    partial = build_audit(records)["partial"]

    assert partial["model_count"] == 2
    assert partial["overlapping_warning_cohorts"] == {
        "unresolved_source_branches": 1,
        "opaque_calls": 2,
    }
    assert partial["incomplete_coverage_metric_counts"] == {
        "semantic_coverage": 1,
        "module_semantic_coverage": 1,
        "checkpoint_parameter_coverage": 0,
        "tensor_shape_coverage": 1,
    }


def test_report_reason_code_is_extracted_without_copying_report_strings(tmp_path: Path):
    records = tmp_path / "capture.ndjson"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    data = json.dumps({
        "model_id": "owner/model", "metadata": {"revision": PIN_A, "inspection": {
            "reason_code": "native_architecture_unavailable", "message": "secret text must not appear",
            "files": ["private/path/model.safetensors"],
        }},
    }).encode()
    digest = hashlib.sha256(data).hexdigest()
    write_rows(
        records,
        [
            {
                "model_id": "owner/model",
                "revision": PIN_A,
                "status": "inspected",
                "report_digest": digest,
                "reason_codes": [],
            }
        ],
    )
    (report_dir / f"{digest}.json").write_bytes(data)

    audit = build_audit(records, report_dir)
    rendered = json.dumps(audit)

    assert audit["artifact_inspection"]["mutually_exclusive_reason_cohorts"][
        "native_architecture_unavailable"
    ] == 1
    assert audit["artifact_inspection"]["report_reason_code_counts"] == {
        "native_architecture_unavailable": 1
    }
    assert "secret text" not in rendered
    assert "private/path" not in rendered


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"status": "partial", "revision": PIN_A}, "missing model_id"),
        ({"model_id": "owner/model", "status": "partial"}, "invalid revision pin"),
        (
            {"model_id": "owner/model", "revision": PIN_A, "status": "partial"},
            "missing coverage",
        ),
        (
            {
                "model_id": "owner/model",
                "revision": PIN_A,
                "status": "http_access",
                "cause_type": "HTTPError",
            },
            "missing stage",
        ),
    ],
)
def test_incomplete_capture_records_fail_honestly(tmp_path: Path, row: dict, message: str):
    records = tmp_path / "capture.ndjson"
    write_rows(records, [row])

    with pytest.raises(AuditInputError, match=message):
        build_audit(records)


def test_malformed_ndjson_fails_honestly(tmp_path: Path):
    records = tmp_path / "capture.ndjson"
    records.write_text('{"model_id":\n', encoding="utf-8")

    with pytest.raises(AuditInputError, match="line 1: invalid JSON"):
        build_audit(records)


def test_cli_writes_structured_json(tmp_path: Path):
    records = tmp_path / "capture.ndjson"
    output = tmp_path / "audit.json"
    write_rows(records, [partial_row("owner/model")])

    assert main(["--records", str(records), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status_counts"]["partial"] == 1


def test_audit_rejects_a_report_whose_content_disagrees_with_its_digest(tmp_path):
    records = tmp_path / "capture.ndjson"
    write_rows(records, [{"model_id": "owner/model", "revision": PIN_A, "status": "inspected",
                          "report_digest": DIGEST_A, "reason_codes": []}])
    (tmp_path / f"{DIGEST_A}.json").write_bytes(b"{}")
    with pytest.raises(AuditInputError, match="digest verification"):
        build_audit(records, tmp_path)


def test_verified_full_reports_count_warnings_beyond_the_capture_sample(tmp_path):
    data = json.dumps({"model_id": "owner/model", "metadata": {"revision": PIN_A,
        "analysis_report": {"warnings": ["Opaque call native.helper: dependencies retained"]}}}).encode()
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / f"{digest}.json").write_bytes(data)
    records = tmp_path / "capture.ndjson"
    write_rows(records, [partial_row("owner/model", warnings=[], report_digest=digest)])
    audit = build_audit(records, tmp_path)
    assert audit["partial"]["warning_evidence"] == "verified_full_reports"
    assert audit["partial"]["overlapping_warning_cohorts"]["opaque_calls"] == 1


@pytest.mark.parametrize("field,value", [("model_id", "https://private/token"), ("stage", "C:/private"),
                                         ("cause_type", "private secret")])
def test_arbitrary_failure_text_cannot_enter_the_public_audit(tmp_path, field, value):
    row = {"model_id": "owner/model", "revision": PIN_A, "status": "http_access",
           "stage": "reconstruction", "cause_type": "HTTPError", field: value}
    records = tmp_path / "capture.ndjson"
    write_rows(records, [row])
    with pytest.raises(AuditInputError):
        build_audit(records)
