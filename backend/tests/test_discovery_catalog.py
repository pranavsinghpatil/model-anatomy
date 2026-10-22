import hashlib
import json

from model_anatomy.api.app import create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.database import BenchmarkCase, BenchmarkPublication, InspectionFailure, ModelReport
from model_anatomy.service.discovery_catalog import DiscoveryCatalogService
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, cache_key


def test_latest_verified_inspection_replaces_case_metrics_and_keeps_benchmark_provenance(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path, engine_version="engine-current"))
    jobs.initialize()
    with jobs.db.write() as session:
        session.add(BenchmarkPublication(
            id="p" * 64,
            metadata_json={"schema_version": 1},
            run_ids=[],
            created_at=1,
        ))
        session.flush()
        session.add(BenchmarkCase(
            publication_id="p" * 64,
            model_id="owner/model",
            ordinal=0,
            data={
                "model_id": "owner/model", "category": "vision", "revision": "a" * 40,
                "status": "artifact_error", "engine_version": "engine-old",
                "started_at": "2026-10-20T00:00:00Z", "finished_at": "2026-10-20T00:00:10Z",
                "case_identity": "c" * 64, "reference": "https://huggingface.co/owner/model/tree/" + "a" * 40,
                "diagnostic_notes": ["This attempt did not produce a validated report."],
            },
        ))
        session.add(ModelReport(
            id="r" * 64,
            cache_identity=cache_key("owner/model", "a" * 40, "engine-current", "1.1"),
            model_id="owner/model", request_revision="main", revision="a" * 40,
            engine_version="engine-current", schema_version="1.1", artifact="model.safetensors",
            explicit_artifact=False, blob_digest="b" * 64,
            summary={
                "result_status": "partial", "architecture": "ActualModel", "parameter_count": 123,
                "dimensions": {"hidden": 16}, "warning_count": 2, "graph_count": 1, "tensor_count": 7,
                "quality": {"semantic_coverage": 0.5, "module_semantic_coverage": 1,
                            "checkpoint_parameter_coverage": 1, "tensor_shape_coverage": 1},
                "category": "unclassified", "declared_task": "text-generation",
            },
            inspected_at=1_800_000_000,
        ))

    result = DiscoveryCatalogService(jobs, clock=lambda: 1_900_000_000).get_catalog()
    assert result["total"] == 1
    row = result["results"][0]
    assert row["source"] == "inspection"
    assert row["status"] == "partial"
    assert row["stored_report_id"] == "r" * 64
    assert row["graph_count"] == 1 and row["tensor_count"] == 7
    assert row["artifact_verified"] is None  # implicit selection is not a verified filename
    assert row["category"] == "vision" and row["category_source"] == "benchmark"
    assert row["declared_task"] == "text-generation" and row["declared_task_source"] == "inspection"
    assert row["benchmark_observations"][0]["status"] == "artifact_error"
    assert row["benchmark_observations"][0]["publication_id"] == "p" * 64
    jobs.db.engine.dispose()


def test_terminal_failure_is_durable_and_public_view_excludes_private_job_identity(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path), clock=lambda: 1000)
    jobs.initialize()
    jobs.acquire_worker("worker")
    guest_id, visitor_token = jobs.guest(None)
    job = jobs.create(guest_id, "https://huggingface.co/owner/model/blob/main/model.safetensors")
    jobs.claim("worker")
    jobs.fail(job["id"], "worker", "artifact_network", "provider said private thing")
    with jobs.db.read() as session:
        failures = session.query(InspectionFailure).all()
        assert len(failures) == 1
        assert failures[0].revision is None
        assert failures[0].status == "network"
    public = DiscoveryCatalogService(jobs, clock=lambda: 2000).get_catalog()
    encoded = json.dumps(public)
    assert public["results"][0]["status"] == "network"
    assert public["results"][0]["source"] == "inspection"
    assert public["results"][0]["artifact_verified"] is False
    assert visitor_token not in encoded and guest_id not in encoded and "provider said" not in encoded
    assert job["id"] not in encoded
    jobs.db.engine.dispose()


def test_failed_explicit_artifact_paths_are_redacted_but_keep_opaque_identity(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path, guest_jobs_per_hour=10))
    jobs.initialize()
    jobs.acquire_worker("worker")
    guest_id, _ = jobs.guest(None)
    revision = "c" * 40
    private_paths = (
        "nested/private-secret-marker-one.safetensors",
        "nested/private-secret-marker-two.safetensors",
    )
    for filename in private_paths:
        submitted = f"https://huggingface.co/owner/model/resolve/{revision}/{filename}"
        job = jobs.create(guest_id, submitted)
        assert jobs.claim("worker") == job["id"]
        jobs.fail(job["id"], "worker", "artifact_network", "private provider response")

    catalog = DiscoveryCatalogService(jobs).get_catalog()
    rows = catalog["results"]
    assert len(rows) == 2
    assert all(row["artifact"] is None and row["artifact_verified"] is False for row in rows)
    assert all(row["reference"] == f"https://huggingface.co/owner/model/tree/{revision}" for row in rows)
    selection_ids = {row["selection_id"] for row in rows}
    assert len(selection_ids) == 2 and all(len(value) == 64 for value in selection_ids)
    public = json.dumps(catalog)
    assert all(filename not in public for filename in private_paths)
    assert "private-secret-marker" not in public and "private provider response" not in public
    jobs.db.engine.dispose()


def test_newer_benchmark_metrics_are_not_combined_with_older_stored_report(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    with jobs.db.write() as session:
        session.add(BenchmarkPublication(id="q" * 64, metadata_json={}, run_ids=[], created_at=1))
        session.flush()
        session.add(BenchmarkCase(
            publication_id="q" * 64, model_id="owner/model", ordinal=0,
            data={"model_id": "owner/model", "category": "text", "revision": "a" * 40,
                  "status": "completed", "engine_version": "engine-new",
                  "started_at": "2026-10-20T00:00:00Z", "finished_at": "2028-01-01T00:00:00Z",
                      "case_identity": "d" * 64, "graph_count": 9, "tensor_count": 90},
        ))
        session.add(ModelReport(
            id="s" * 64, cache_identity="x" * 64, model_id="owner/model", request_revision="main",
            revision="a" * 40, engine_version="engine-old", schema_version="1.1",
            artifact="model.safetensors", explicit_artifact=False, blob_digest="z" * 64,
            summary={"result_status": "partial", "graph_count": 1, "tensor_count": 2,
                     "parameter_count": 3, "warning_count": 4}, inspected_at=1_700_000_000,
        ))
    row = DiscoveryCatalogService(jobs).get_catalog()["results"][0]
    assert row["source"] == "benchmark" and row["stored_report_id"] is None
    assert (row["status"], row["graph_count"], row["tensor_count"]) == ("completed", 9, 90)
    assert row["warning_count"] is None and row["parameter_counts"]["unique"] is None
    assert row["previous_observations"][0]["stored_report_id"] == "s" * 64
    jobs.db.engine.dispose()


def test_distinct_revisions_and_artifact_selections_remain_separate(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    with jobs.db.write() as session:
        for index, revision, artifact, explicit in (
            ("a", "a" * 40, "model.safetensors", False),
            ("b", "b" * 40, "model.safetensors", False),
            ("c", "a" * 40, "custom.safetensors", True),
        ):
            session.add(ModelReport(
                id=index * 64, cache_identity=index * 64, model_id="owner/model", request_revision=revision,
                revision=revision, engine_version=f"engine-{index}", schema_version="1.1", artifact=artifact,
                explicit_artifact=explicit, blob_digest=(index.upper() * 64).lower(),
                summary={"result_status": "completed", "tensor_count": 1}, inspected_at=1_700_000_000 + ord(index),
            ))
    rows = DiscoveryCatalogService(jobs).get_catalog()["results"]
    assert len(rows) == 3
    assert {(row["revision"], row["artifact"], row["explicit_artifact"]) for row in rows} == {
        ("a" * 40, "model.safetensors", False), ("b" * 40, "model.safetensors", False),
        ("a" * 40, "custom.safetensors", True),
    }
    jobs.db.engine.dispose()


def test_timestamp_sorting_compares_instants_across_timezone_spellings(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    with jobs.db.write() as session:
        session.add(BenchmarkPublication(id="t" * 64, metadata_json={}, run_ids=[], created_at=1))
        session.flush()
        session.add(BenchmarkCase(
            publication_id="t" * 64, model_id="owner/model", ordinal=0,
            data={"model_id": "owner/model", "category": "text", "revision": "a" * 40,
                  "status": "completed", "engine_version": "engine-benchmark",
                  "started_at": "2028-01-01T00:00:00Z", "finished_at": "2028-01-01T01:00:00+01:00",
                  "case_identity": "e" * 64, "graph_count": 9},
        ))
        session.add(ModelReport(
            id="u" * 64, cache_identity="v" * 64, model_id="owner/model", request_revision="main",
            revision="a" * 40, engine_version="engine-inspection", schema_version="1.1",
            artifact="model.safetensors", explicit_artifact=False, blob_digest="w" * 64,
            summary={"result_status": "partial", "graph_count": 1}, inspected_at=1_830_297_600,
        ))
    row = DiscoveryCatalogService(jobs).get_catalog()["results"][0]
    assert row["source"] == "inspection"
    assert row["graph_count"] == 1
    assert row["benchmark_observations"][0]["finished_at"] == "2028-01-01T01:00:00+01:00"
    jobs.db.engine.dispose()


def test_later_failure_keeps_prior_saved_report_link_in_history(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    with jobs.db.write() as session:
        session.add(ModelReport(
            id="m" * 64, cache_identity="n" * 64, model_id="owner/model", request_revision="main",
            revision="a" * 40, engine_version="engine-old", schema_version="1.1",
            artifact="model.safetensors", explicit_artifact=False, blob_digest="o" * 64,
            summary={"result_status": "completed", "graph_count": 1}, inspected_at=1_700_000_000,
        ))
        session.add(InspectionFailure(
            id="f" * 64, model_id="owner/model", revision="a" * 40, revision_source="resolved",
            artifact="model.safetensors", explicit_artifact=False, status="network",
            engine_version="engine-new", started_at=1_800_000_000, finished_at=1_800_000_100,
        ))
    row = DiscoveryCatalogService(jobs).get_catalog()["results"][0]
    assert row["status"] == "network" and row["stored_report_id"] is None
    assert any(item["stored_report_id"] == "m" * 64 for item in row["previous_observations"])
    jobs.db.engine.dispose()


def test_failed_retry_is_saved_only_when_terminal_and_cancellation_is_not_public(tmp_path):
    now = [1000.0]
    jobs = JobService(Settings(data_dir=tmp_path), clock=lambda: now[0])
    jobs.initialize()
    jobs.acquire_worker("worker")
    guest_id, _ = jobs.guest(None)
    first = jobs.create(guest_id, "owner/model")
    jobs.claim("worker")
    jobs.fail(first["id"], "worker", "artifact_network", "temporary", transient=True)
    with jobs.db.read() as session:
        assert session.query(InspectionFailure).count() == 0
    now[0] += 5
    jobs.claim("worker")
    jobs.fail(first["id"], "worker", "artifact_network", "temporary", transient=True)
    with jobs.db.read() as session:
        assert session.query(InspectionFailure).count() == 1

    second = jobs.create(guest_id, "owner/other")
    jobs.claim("worker")
    jobs.cancel(second["id"], guest_id)
    jobs.fail(second["id"], "worker", "artifact_network", "cancelled failure text")
    with jobs.db.read() as session:
        assert session.query(InspectionFailure).count() == 1
    jobs.db.engine.dispose()


def test_failed_requested_pins_and_recovery_exhaustion_are_durable(tmp_path):
    now = [1000.0]
    settings = Settings(data_dir=tmp_path, max_attempts=1, lease_seconds=5, guest_jobs_per_hour=10)
    jobs = JobService(settings, clock=lambda: now[0])
    jobs.initialize()
    jobs.acquire_worker("worker-a")
    guest_id, _ = jobs.guest(None)
    for revision in ("a" * 40, "b" * 40):
        job = jobs.create(guest_id, f"https://huggingface.co/owner/model/tree/{revision}")
        assert jobs.claim("worker-a") == job["id"]
        jobs.fail(job["id"], "worker-a", "artifact_network", "provider failure")
    with jobs.db.read() as session:
        failures = session.query(InspectionFailure).order_by(InspectionFailure.revision).all()
        assert {row.revision for row in failures} == {"a" * 40, "b" * 40}
        assert all(row.revision_source == "requested" for row in failures)

    stuck = jobs.create(guest_id, "owner/stuck")
    assert jobs.claim("worker-a") == stuck["id"]
    now[0] += settings.lease_seconds + 1
    jobs.acquire_worker("worker-b")
    assert jobs.claim("worker-b") is None
    with jobs.db.read() as session:
        rows = session.query(InspectionFailure).all()
        assert len(rows) == 3
        assert any(row.status == "failure" and row.revision is None for row in rows)
    jobs.db.engine.dispose()


def test_discovery_endpoint_is_available_alongside_legacy_catalog_routes(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    paths = {route.path for route in app.routes}
    assert "/api/v1/catalog/discovery" in paths
    assert "/api/v1/catalog" in paths and "/api/v1/benchmarks/catalog" in paths
    endpoint = next(route.endpoint for route in app.routes if route.path == "/api/v1/catalog/discovery")
    body = endpoint()
    assert set(body) == {
        "results", "total", "status_counts", "generated_at", "benchmark_publication_id", "external_publications",
    }
    benchmark_endpoint = next(route.endpoint for route in app.routes if route.path == "/api/v1/benchmarks/catalog")
    try:
        benchmark_endpoint()
    except ServiceError as error:
        assert error.status == 404 and error.code == "benchmark_catalog_not_found"
    else:
        raise AssertionError("legacy benchmark catalog should keep its not-published response")
    app.state.jobs.db.engine.dispose()


def test_bounded_backfill_reads_and_validates_private_blob_without_changing_report_identity(tmp_path):
    from scripts.backfill_catalog_metrics import backfill

    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    revision = "a" * 40
    document = {
        "model_id": "owner/model", "schema_version": "1.1",
        "metadata": {
            "revision": revision, "engine_version": jobs.settings.engine_version,
            "inspection": {"kind": "artifact_inspection", "reason_code": "unsupported",
                            "message": "not supported", "files": [], "checkpoint_available": False},
            "model_profile": {"declared_task": "text-generation"},
        },
        "tensors": {"weight": {"shape": [2]}},
    }
    raw = json.dumps(document).encode()
    digest = hashlib.sha256(raw).hexdigest()
    report_id = "r" * 64
    with jobs.db.write() as session:
        session.add(ModelReport(
            id=report_id, cache_identity=cache_key("owner/model", revision, jobs.settings.engine_version, "1.1"),
            model_id="owner/model", request_revision="main", revision=revision,
            engine_version=jobs.settings.engine_version, schema_version="1.1", artifact="model.safetensors",
            explicit_artifact=False, blob_digest=digest,
            summary={"legacy_marker": "preserved"}, inspected_at=1,
        ))

    class Storage:
        def get(self, _digest):
            return raw

    assert backfill(jobs.db, Storage(), limit=1) == {
        "considered": 1, "updated": 1, "failed": 0, "last_id": report_id,
    }
    with jobs.db.read() as session:
        updated = session.get(ModelReport, report_id)
        assert updated.id == report_id and updated.revision == revision and updated.blob_digest == digest
        assert updated.summary["tensor_count"] == 1
        assert updated.summary["declared_task"] == "text-generation"
        assert updated.summary == {
            "legacy_marker": "preserved", "warning_count": 0, "graph_count": 0,
            "tensor_count": 1, "declared_task": "text-generation", "category": "unclassified",
        }
    assert backfill(jobs.db, Storage(), limit=1) == {
        "considered": 0, "updated": 0, "failed": 0, "last_id": None,
    }
    jobs.db.engine.dispose()
