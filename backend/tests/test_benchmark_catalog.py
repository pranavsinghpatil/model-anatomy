"""Evidence boundary tests for the public benchmark catalog."""

import copy
import json
from collections import Counter

import pytest
from scripts.model_matrix import COVERAGE_KEYS, case_identity
from scripts.publish_benchmark_catalog import build_catalog


@pytest.fixture
def measured_run():
    cases = [{"model_id": f"owner/model-{i}", "category": "text", "revision": "a" * 40} for i in range(250)]
    env = {
        "engine_version": "test-engine",
        "runner_sha256": "r" * 64,
        "engine_source_sha256": "s" * 64,
        "wall_timeout_seconds": 120,
        "process_tree_rss_limit_bytes": 4 * 1024**3,
        "python": "3.11",
        "platform": "test",
        "dependencies": {},
    }
    identity_env = {
        "timeout_seconds": 120,
        "memory_bytes": 4 * 1024**3,
        "python": "3.11",
        "platform": "test",
        "dependencies": {},
    }
    rows = []
    for case in cases:
        rows.append(
            {
                **case,
                "status": "completed",
                "engine_version": "test-engine",
                "architecture": "TestModel",
                "case_identity": case_identity(case, "test-engine", "r" * 64, "s" * 64, identity_env),
                "runner_sha256": "r" * 64,
                "engine_source_sha256": "s" * 64,
                "started_at": "2026-10-20T00:00:00Z",
                "finished_at": "2026-10-20T00:00:01Z",
                "graph_count": 2,
                "coverage": {key: 1.0 for key in COVERAGE_KEYS},
                "warning_count": 0,
                "warnings": [],
                "service_contract_valid": True,
                "storage_contract_valid": True,
                "report_roundtrip": {"valid": True},
                "weight_payload_bytes_read": 0,
                "forward_executed": False,
            }
        )
    return json.dumps({"schema_version": 1, "cases": cases}).encode(), {
        "cases": 250,
        "results": rows,
        "environment": env,
        "status_counts": {"completed": 250},
        "generated_at": "2026-10-20T00:01:00Z",
    }


def recount(summary):
    summary["status_counts"] = dict(Counter(row["status"] for row in summary["results"]))


def test_catalog_preserves_failed_attempts_without_publishing_private_diagnostics(measured_run):
    manifest, summary = measured_run
    row = summary["results"][0]
    row.update(
        status="artifact_error",
        stage="reconstruction",
        cause_type="ArtifactError",
        message="C:/private/user/token",
        traceback="private trace",
    )
    recount(summary)
    catalog = build_catalog(manifest, summary)
    assert len(catalog["results"]) == 250
    failed = catalog["results"][0]
    assert failed["status"] == "artifact_error"
    assert "architecture" not in failed and "coverage" not in failed
    assert failed["error"] == {
        "message": "artifact_error during reconstruction",
        "cause_type": "ArtifactError",
    }
    assert "private" not in json.dumps(catalog)
    assert catalog["status_counts"] == {"artifact_error": 1, "completed": 249}


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"revision": "b" * 40}, "revision differs"),
        ({"case_identity": "forged"}, "identity mismatch"),
        ({"engine_source_sha256": "old"}, "mixed runner or engine"),
        ({"service_contract_valid": False}, "service/storage contract"),
        ({"forward_executed": True}, "static-inspection invariants"),
        ({"coverage": {}}, "complete report"),
        ({"status": "inspected"}, "artifact-only"),
    ],
)
def test_catalog_rejects_misattributed_or_overstated_results(measured_run, mutation, match):
    manifest, summary = measured_run
    summary["results"][0].update(mutation)
    recount(summary)
    with pytest.raises(ValueError, match=match):
        build_catalog(manifest, summary)


def test_catalog_requires_all_250_attempts_and_correct_counts(measured_run):
    manifest, summary = measured_run
    shortened = copy.deepcopy(summary)
    shortened["results"].pop()
    with pytest.raises(ValueError, match="250 measured"):
        build_catalog(manifest, shortened)
    summary["status_counts"] = {"completed": 249}
    with pytest.raises(ValueError, match="counts disagree"):
        build_catalog(manifest, summary)


def test_committed_matrix_is_250_distinct_and_preserves_baseline():
    from pathlib import Path

    from scripts.model_matrix import validate_manifest

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "benchmarks/model-matrix-250.json").read_text())
    cases = validate_manifest(manifest)
    assert len(cases) == len({case["model_id"].casefold() for case in cases}) == 250
    baseline = json.loads((root / "benchmarks/model-matrix-100.json").read_text())["cases"]
    assert cases[:120] == baseline
    assert len({case["category"] for case in cases}) >= 40
