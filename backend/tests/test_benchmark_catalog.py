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


@pytest.mark.parametrize("status", ["partial", "inspected"])
def test_successful_rows_never_publish_raw_native_exception_diagnostics(measured_run, status):
    manifest, summary = measured_run
    row = summary["results"][0]
    private = "Constructor failed at C:/private/person/cache/model.py with private-token"
    row.update(
        status=status,
        warning_count=1,
        warnings=[private],
        reason_codes=["native_architecture_unavailable", private],
    )
    if status == "inspected":
        row.update(architecture=None, graph_count=0)
    recount(summary)
    result = build_catalog(manifest, summary)
    assert "private" not in json.dumps(result)
    item = result["results"][0]
    assert item["warnings"] == [] and item["warning_count"] == 1
    assert item["reason_codes"] == ["native_architecture_unavailable"]
    assert item["diagnostic_notes"]


def test_publisher_equivalence_warning_uses_fixed_public_summary(measured_run):
    manifest, summary = measured_run
    row = summary["results"][0]
    row.update(
        status="partial",
        warning_count=1,
        warnings=["C:/private: equivalence to publisher custom code is unverified; raw-secret"],
    )
    recount(summary)
    result = build_catalog(manifest, summary)
    assert "raw-secret" not in json.dumps(result) and "private" not in json.dumps(result)
    assert any("equivalence is unverified" in note for note in result["results"][0]["diagnostic_notes"])


def test_failed_public_stage_cannot_carry_arbitrary_diagnostics(measured_run):
    manifest, summary = measured_run
    summary["results"][0].update(status="failure", stage="C:/private/cache", cause_type="private-token")
    recount(summary)
    catalog = build_catalog(manifest, summary)
    assert catalog["results"][0]["error"] == {"message": "failure during case"}
    assert "private" not in json.dumps(catalog)


def test_manifest_fingerprint_survives_cross_platform_json_checkout(measured_run):
    manifest, summary = measured_run
    windows = json.dumps(json.loads(manifest), indent=2).replace("\n", "\r\n").encode()
    linux = json.dumps(json.loads(manifest), sort_keys=True, separators=(",", ":")).encode()
    assert windows != linux
    assert (
        build_catalog(windows, summary)["manifest_sha256"] == build_catalog(linux, summary)["manifest_sha256"]
    )


def targeted_recheck(manifest, baseline):
    case = json.loads(manifest)["cases"][0]
    environment = {
        **baseline["environment"],
        "engine_version": "fixed-engine",
        "engine_source_sha256": "f" * 64,
    }
    identity_environment = {
        "timeout_seconds": environment["wall_timeout_seconds"],
        "memory_bytes": environment["process_tree_rss_limit_bytes"],
        "python": environment["python"],
        "platform": environment["platform"],
        "dependencies": environment["dependencies"],
    }
    row = {
        **copy.deepcopy(baseline["results"][0]),
        "engine_version": "fixed-engine",
        "engine_source_sha256": "f" * 64,
        "status": "partial",
        "finished_at": "2026-10-20T00:02:00Z",
        "warning_count": 1,
        "coverage": {key: 0.8 for key in COVERAGE_KEYS},
    }
    row["case_identity"] = case_identity(
        case, "fixed-engine", environment["runner_sha256"], "f" * 64, identity_environment
    )
    return json.dumps({"schema_version": 1, "cases": [case]}).encode(), {
        "environment": environment,
        "cases": 1,
        "results": [row],
        "status_counts": {"partial": 1},
        "generated_at": "2026-10-20T00:03:00Z",
    }


def test_targeted_recheck_keeps_250_rows_and_attributes_the_repair(measured_run):
    manifest, baseline = measured_run
    recheck = targeted_recheck(manifest, baseline)
    baseline["results"][0]["status"] = "invalid_report"
    recount(baseline)
    result = build_catalog(manifest, baseline, rechecks=[recheck])
    repaired = result["results"][0]
    assert len(result["results"]) == 250 and result["recheck_attempts"] == 1
    assert repaired["status"] == "partial" and repaired["engine_version"] == "fixed-engine"
    assert repaired["previous_observations"][0]["status"] == "invalid_report"
    assert result["results"][1]["engine_version"] == "test-engine"
    assert result["engine_versions"] == ["test-engine", "fixed-engine"]
    assert result["status_counts"] == {"partial": 1, "completed": 249}
    assert [run["cases"] for run in result["runs"]] == [250, 1]
    assert "engine_source_sha256" not in result


def test_rechecks_cannot_change_the_repository_pin_or_replace_with_older_data(measured_run):
    manifest, baseline = measured_run
    recheck_manifest, recheck_summary = targeted_recheck(manifest, baseline)
    altered = json.loads(recheck_manifest)
    altered["cases"][0]["revision"] = "b" * 40
    with pytest.raises(ValueError, match="original pinned case"):
        build_catalog(manifest, baseline, rechecks=[(json.dumps(altered).encode(), recheck_summary)])
    recheck_summary["results"][0]["finished_at"] = baseline["results"][0]["finished_at"]
    with pytest.raises(ValueError, match="must be newer"):
        build_catalog(manifest, baseline, rechecks=[(recheck_manifest, recheck_summary)])
