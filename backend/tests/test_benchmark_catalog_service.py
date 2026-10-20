import json
from collections import Counter
from pathlib import Path

import pytest

from model_anatomy.api.app import create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.benchmark_catalog import BenchmarkCatalogService
from model_anatomy.service.database import ModelReport
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService


def sample_catalog():
    rows = []
    for index in range(250):
        rows.append({
            "model_id": f"owner/model-{index}", "category": "text", "revision": f"{index:040x}",
            "status": "completed", "architecture": "ExampleModel", "engine_version": "engine-1",
            "started_at": "2026-10-20T00:00:00Z", "finished_at": "2026-10-20T00:00:01Z",
            "warning_count": 0, "graph_count": 1, "node_count": 2, "tensor_count": 3,
            "parameter_counts": {"unique": 100}, "coverage": {"semantic_coverage": 1.0},
            "wall_seconds": 1.0, "service_contract_valid": True, "case_identity": f"{index:064x}",
            "warnings": [], "reason_codes": [], "diagnostic_notes": [],
        })
    return {
        "schema_version": 1, "generated_at": "2026-10-20T00:00:00Z", "engine_version": "engine-1",
        "manifest_sha256": "a" * 64, "manifest_hash_encoding": "canonical-json-v1",
        "method": "Pinned metadata and bounded static reconstruction; no inference quality or speed measurement.",
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "runs": [{
            "kind": "baseline", "cases": 250, "generated_at": "2026-10-20T00:00:00Z",
            "engine_version": "engine-1", "runner_sha256": "b" * 64,
            "engine_source_sha256": "c" * 64, "status_counts": {"completed": 250},
        }], "results": rows,
    }


def sample_runs():
    catalog = sample_catalog()
    history = catalog["runs"][0]
    return [{
        "cases": 250, "generated_at": history["generated_at"], "status_counts": history["status_counts"],
        "environment": {
            "engine_version": history["engine_version"], "runner_sha256": history["runner_sha256"],
            "engine_source_sha256": history["engine_source_sha256"],
        },
        "results": [{"model_id": row["model_id"], "status": row["status"]} for row in catalog["results"]],
    }]


def test_publish_is_content_addressed_and_returns_safe_ordered_catalog(tmp_path):
    service = JobService(Settings(data_dir=tmp_path))
    service.initialize()
    catalog_service = BenchmarkCatalogService(service, clock=lambda: 10.0)
    catalog, runs = sample_catalog(), sample_runs()

    first = catalog_service.publish(catalog, runs)
    second = catalog_service.publish(catalog, runs)

    assert first == second
    published = catalog_service.get_catalog()
    assert published["results"][0]["model_id"] == "owner/model-0"
    assert published["runs"] == catalog["runs"]
    assert len(published["results"]) == 250
    assert all("stored_report_id" in row for row in published["results"])
    service.db.engine.dispose()


def test_publish_matches_only_compatible_stored_report(tmp_path):
    service = JobService(Settings(data_dir=tmp_path, engine_version="engine-1", schema_version="1.1"))
    service.initialize()
    catalog_service = BenchmarkCatalogService(service, clock=lambda: 10.0)
    catalog = sample_catalog()
    row = catalog["results"][0]
    with service.db.write() as session:
        session.add(ModelReport(
            id="r" * 64, cache_identity="c" * 64, model_id=row["model_id"], request_revision=row["revision"],
            revision=row["revision"], engine_version="engine-1", schema_version="1.1",
            artifact="model.safetensors", explicit_artifact=False, blob_digest="b" * 64,
            summary={}, inspected_at=10.0,
        ))
    catalog_service.publish(catalog, sample_runs())
    assert catalog_service.get_catalog()["results"][0]["stored_report_id"] == "r" * 64
    service.db.engine.dispose()


def test_publication_retains_external_failures_and_baseline_recheck_history(tmp_path):
    service = JobService(Settings(data_dir=tmp_path))
    service.initialize()
    catalog_service = BenchmarkCatalogService(service, clock=lambda: 10.0)
    catalog = sample_catalog()
    baseline = sample_runs()[0]
    recheck_rows = []
    for row in catalog["results"][:5]:
        row.update(
            status="artifact_error", error={"message": "artifact_error during reconstruction"},
            diagnostic_notes=["This attempt did not produce a validated report."],
            previous_observations=[{
                "status": "completed", "engine_version": "engine-1", "finished_at": row["finished_at"],
                "case_identity": row["case_identity"],
            }],
        )
        for key in ("architecture", "graph_count", "node_count", "tensor_count", "parameter_counts", "coverage", "service_contract_valid"):
            row.pop(key, None)
        recheck_rows.append({"model_id": row["model_id"], "status": "artifact_error", "stage": "reconstruction"})
    catalog["status_counts"] = {"artifact_error": 5, "completed": 245}
    recheck = {
        "cases": 5, "generated_at": "2026-10-20T00:02:00Z", "status_counts": {"artifact_error": 5},
        "environment": baseline["environment"], "results": recheck_rows,
    }
    catalog["runs"].append({
        "kind": "recheck", "cases": 5, "generated_at": recheck["generated_at"], "engine_version": "engine-1",
        "runner_sha256": "b" * 64, "engine_source_sha256": "c" * 64, "status_counts": {"artifact_error": 5},
    })

    catalog_service.publish(catalog, [baseline, recheck])

    result = catalog_service.get_catalog()
    assert len(result["results"]) == 250
    assert sum(row["status"] == "artifact_error" for row in result["results"]) == 5
    assert result["results"][0]["previous_observations"][0]["status"] == "completed"
    assert [run["kind"] for run in result["runs"]] == ["baseline", "recheck"]
    service.db.engine.dispose()


@pytest.mark.parametrize("mutation", [
    lambda catalog: catalog["results"].pop(),
    lambda catalog: catalog["results"][1].update(model_id="OWNER/model-0"),
    lambda catalog: catalog["results"][0].update(traceback="private trace"),
    lambda catalog: catalog["results"][0].update(diagnostic_notes=["C:/private/token"]),
])
def test_publish_rejects_incomplete_duplicate_or_unsafe_rows(tmp_path, mutation):
    service = JobService(Settings(data_dir=tmp_path))
    service.initialize()
    catalog = sample_catalog()
    mutation(catalog)
    with pytest.raises(ValueError):
        BenchmarkCatalogService(service).publish(catalog, sample_runs())
    service.db.engine.dispose()


def test_api_serves_published_catalog_and_honest_404(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    endpoint = next(route.endpoint for route in app.routes if route.path == "/api/v1/benchmarks/catalog")
    with pytest.raises(ServiceError) as error:
        endpoint()
    assert error.value.status == 404
    assert error.value.code == "benchmark_catalog_not_found"
    app.state.benchmark_catalog.publish(sample_catalog(), sample_runs())
    result = endpoint()
    assert len(result["results"]) == 250
    assert "stored_report_id" in result["results"][0]


def test_original_250_publication_imports_idempotently_with_private_source_runs(tmp_path):
    from scripts.publish_benchmark_catalog import store_catalog_only
    from sqlalchemy import func, select

    from model_anatomy.service.database import BenchmarkCase, BenchmarkRun, Database

    root = Path(__file__).resolve().parents[2]
    database_url = f"sqlite:///{(tmp_path / 'catalog.sqlite3').as_posix()}"
    def make_database(_):
        database = Database(database_url)
        database.initialize(create=True)
        return database
    def load_profile(_):
        return {
            "DATABASE_URL": "postgresql://user:test@localhost:5432/postgres?sslmode=require",
            "MODEL_ANATOMY_SUPABASE_URL": "https://storage.example",
            "MODEL_ANATOMY_SUPABASE_KEY": "test-only",
            "MODEL_ANATOMY_SUPABASE_BUCKET": "private",
        }
    arguments = (
        root / "frontend/public/benchmark-catalog.json", root / "benchmarks/model-matrix-250.json",
        [root / "docs/benchmarks/model-matrix-250-baseline.json", root / "docs/benchmarks/model-matrix-250-rechecks.json"],
        root / "not-read.env",
    )
    first = store_catalog_only(*arguments, profile_loader=load_profile, database_factory=make_database)
    assert store_catalog_only(*arguments, profile_loader=load_profile, database_factory=make_database) == first
    database = make_database(None)
    with database.read() as session:
        assert session.scalar(select(func.count()).select_from(BenchmarkRun)) == 2
        assert session.scalar(select(func.count()).select_from(BenchmarkCase)) == 250
        assert "_source_case_definition" in session.scalars(select(BenchmarkCase)).first().data
    database.engine.dispose()
    service = JobService(Settings(data_dir=tmp_path, database_url=database_url))
    result = BenchmarkCatalogService(service).get_catalog()
    expected = json.loads(arguments[0].read_text(encoding="utf-8"))
    assert result["status_counts"] == expected["status_counts"]
    assert len(result["results"]) == 250
    assert all("_source_case_definition" not in row for row in result["results"])
    service.db.engine.dispose()
