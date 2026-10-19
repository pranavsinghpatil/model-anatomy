"""Offline contract checks for the real-model matrix runner."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests.exceptions import ConnectionError, HTTPError
from scripts.model_matrix import (
    CaseStageError,
    case_identity,
    classify_failure,
    read_existing,
    run_case,
    source_tree_sha256,
    summarize_observation,
    validate_manifest,
)

from model_anatomy.src.errors import AnatomyError, ArtifactError, UnsupportedModelError

SHA = "a" * 40
CASE = {"model_id": "owner/model", "category": "text", "revision": SHA}


def idle_worker(case, connection):
    """Synthetic worker used to exercise parent timeout and RSS handling."""
    time.sleep(2)


def test_manifest_requires_pinned_unique_repositories_and_production_size():
    with pytest.raises(ValueError, match="at least 100"):
        validate_manifest({"schema_version": 1, "cases": [CASE]})
    assert validate_manifest({"schema_version": 1, "cases": [CASE]}, allow_small=True) == [CASE]
    with pytest.raises(ValueError, match="duplicate model_id"):
        validate_manifest({"schema_version": 1, "cases": [CASE, dict(CASE)]}, allow_small=True)
    with pytest.raises(ValueError, match="pinned lowercase 40-character"):
        validate_manifest({"schema_version": 1, "cases": [{**CASE, "revision": "main"}]}, allow_small=True)
    with pytest.raises(ValueError, match="pinned HTTPS"):
        validate_manifest(
            {"schema_version": 1, "cases": [{**CASE, "reference": "https://example.com/model.safetensors"}]},
            allow_small=True,
        )
    reference = f"https://huggingface.co/{CASE['model_id']}/resolve/{SHA}/weights/model.safetensors"
    assert validate_manifest(
        {"schema_version": 1, "cases": [{**CASE, "reference": reference}]}, allow_small=True
    )
    with pytest.raises(ValueError, match="repository and revision must match"):
        validate_manifest(
            {"schema_version": 1, "cases": [{**CASE, "reference": reference.replace(SHA, "b" * 40)}]},
            allow_small=True,
        )


def test_resume_identity_includes_revision_reference_and_engine_version():
    baseline = case_identity(CASE, "engine-a")
    assert case_identity({**CASE, "revision": "b" * 40}, "engine-a") != baseline
    assert (
        case_identity(
            {
                **CASE,
                "reference": "https://huggingface.co/owner/model/resolve/"
                + SHA
                + "/weights/model.safetensors",
            },
            "engine-a",
        )
        != baseline
    )
    assert case_identity(CASE, "engine-b") != baseline
    assert case_identity(CASE, "engine-a", "runner-a") != case_identity(CASE, "engine-a", "runner-b")
    assert case_identity(CASE, "engine-a", "runner-a", "source-a") != case_identity(
        CASE, "engine-a", "runner-a", "source-b"
    )
    environment = {
        "timeout_seconds": 120,
        "memory_bytes": 4 * 1024**3,
        "dependencies": {"torch": "2.0"},
    }
    identity = case_identity(CASE, "engine-a", "runner-a", "source-a", environment)
    assert (
        case_identity(CASE, "engine-a", "runner-a", "source-a", {**environment, "timeout_seconds": 60})
        != identity
    )
    assert (
        case_identity(CASE, "engine-a", "runner-a", "source-a", {**environment, "memory_bytes": 1024})
        != identity
    )
    assert (
        case_identity(
            CASE, "engine-a", "runner-a", "source-a", {**environment, "dependencies": {"torch": "2.1"}}
        )
        != identity
    )


def test_failure_classification_uses_types_and_preserves_stage_and_cause():
    http_error = HTTPError("forbidden", response=SimpleNamespace(status_code=403))
    connection_error = ConnectionError("connection reset")
    assert classify_failure(http_error)["status"] == "http_access"
    assert classify_failure(connection_error)["status"] == "network"
    assert classify_failure(UnsupportedModelError("unknown architecture"))["status"] == "unsupported"
    assert classify_failure(ArtifactError("invalid safetensors header"))["status"] == "artifact_error"
    assert classify_failure(MemoryError("allocation failed"))["status"] == "memorylimit"

    reconstruct_error = classify_failure(
        CaseStageError("reconstruction", AnatomyError("internal graph issue"))
    )
    assert reconstruct_error == {
        "status": "failure",
        "stage": "reconstruction",
        "cause_type": "AnatomyError",
        "cause_message": "internal graph issue",
    }
    report_error = classify_failure(CaseStageError("report_validation", AnatomyError("dangling edge")))
    assert report_error["status"] == "invalid_report"
    assert report_error["stage"] == "report_validation"


def test_run_invariant_observations_become_unknown_when_case_has_no_evidence():
    records = [
        {"weight_payload_bytes_read": 0, "forward_executed": False},
        {"status": "timeout"},
    ]
    assert summarize_observation(records, "weight_payload_bytes_read", 0) == {
        "value": None,
        "observed_cases": 1,
        "total_cases": 2,
    }
    assert summarize_observation(records, "forward_executed", False)["value"] is None
    assert summarize_observation(records[:1], "weight_payload_bytes_read", 0)["value"] is True
    assert (
        summarize_observation(
            [{"weight_payload_bytes_read": 16}, {"status": "timeout"}], "weight_payload_bytes_read", 0
        )["value"]
        is False
    )


def test_source_tree_hash_tracks_python_source_changes(tmp_path: Path):
    root = tmp_path / "model_anatomy"
    root.mkdir()
    source = root / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(root)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(root) != first


def test_resume_reads_ndjson_by_identity_and_rejects_corrupt_rows(tmp_path: Path):
    path = tmp_path / "runs.ndjson"
    path.write_text(json.dumps({"case_identity": "one", "status": "partial"}) + "\n", encoding="utf-8")
    assert read_existing(path)["one"]["status"] == "partial"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid NDJSON"):
        read_existing(path)


def test_timeout_and_memory_limit_are_distinct_process_outcomes():
    timeout = run_case(CASE, timeout=0.2, worker_target=idle_worker)
    assert timeout["status"] == "timeout"
    memory = run_case(CASE, timeout=5, memory_bytes=1, worker_target=idle_worker)
    assert memory["status"] == "memorylimit"
    assert memory["peak_process_tree_rss_bytes"] > 1


def test_nonfinite_resource_budgets_are_rejected_before_starting_a_child():
    with pytest.raises(ValueError, match="timeout"):
        run_case(CASE, timeout=float("nan"))
    with pytest.raises(ValueError, match="memory budget"):
        run_case(CASE, memory_bytes=float("nan"))
