import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from scripts.import_model_validation_csv import import_csv

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import Database, ExternalValidationPublication, ModelReport
from model_anatomy.service.discovery_catalog import DiscoveryCatalogService
from model_anatomy.service.external_validation import (
    MAX_CSV_BYTES,
    ExternalValidationService,
    prepare_publication,
)
from model_anatomy.service.jobs import JobService, cache_key

SOURCE_REPOSITORY = "https://github.com/pranavsinghpatil/model-anatomy"
SOURCE_COMMIT = "56ffad8f67afb83076c3067a90d846de1e88d83d"
SOURCE_PATH = "output/hf_model_parse_results.csv"


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[
        "input_row", "model_id", "status", "graph_generated", "reason_code", "architecture", "mode",
        "revision", "engine_version", "graph_count", "node_count", "started_at", "reason", "suggested_fix",
        "graph_path", "diagnostic_detail",
    ])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _source_row(input_row, model_id, status, revision, **overrides):
    return {
        "input_row": str(input_row), "model_id": model_id, "status": status,
        "graph_generated": "True" if status in {"complete", "partial"} else "False",
        "reason_code": "" if status != "fail" else "ArtifactError",
        "architecture": "DecoderModel", "mode": "static_module_architecture",
        "revision": revision, "engine_version": "source-1", "graph_count": "3", "node_count": "42",
        "started_at": "2026-10-20T00:00:00+00:00", "reason": "private diagnostic marker",
        "suggested_fix": "private proposed fix", "graph_path": "C:/private/path/report.mmd",
        "diagnostic_detail": "private exception detail", **overrides,
    }


def test_real_csv_is_pinned_deduplicated_and_public_safe():
    # Keep the exact source bytes from the attributed integration commit. The
    # repository's current output CSV is a later, distinct historical run.
    csv_path = Path(__file__).parent / "fixtures" / "vv-validation-56ffad8.csv"
    raw = csv_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "53332c2c68c8ff23b837ebb7fd299bd0d9834b056a39e3e7174d5bdf897bd9a4"
    publication_id, publication, rows = prepare_publication(
        raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH,
    )
    assert len(publication_id) == 64
    assert publication["csv_sha256"] == hashlib.sha256(raw).hexdigest()
    assert publication["metadata"]["source_counts"] == {
        "input_rows": 163, "unique_repository_ids": 162, "unique_observations": 162,
        "duplicate_input_rows": 1,
        "raw_statuses": {"complete": 48, "fail": 105, "partial": 10},
        "observation_statuses": {
            "out_of_scope": 87, "reported_complete": 48, "reported_failure": 17, "reported_partial": 10,
        },
        "raw_reason_codes": {
            "ArtifactError": 8, "architecture_config_unavailable": 1, "architecture_incomplete": 10,
            "http_401": 7, "native_architecture_unavailable": 8, "none": 48, "safetensors_unavailable": 81,
        },
        "observation_reason_codes": {
            "ArtifactError": 8, "architecture_config_unavailable": 1, "architecture_incomplete": 10,
            "http_401": 7, "native_architecture_unavailable": 8, "none": 48, "safetensors_unavailable": 80,
        },
    }
    duplicate = next(row for row in rows if row["model_id"] == "microsoft/unispeech-sat-base")
    assert duplicate["scope_reason"] == "no_safetensors" and duplicate["source_rows"] == [139, 140]
    restricted = [row for row in rows if row["scope_reason"] == "access_restricted"]
    assert len(restricted) == 7 and sum(row["revision"] is None for row in restricted) == 3
    assert all(row["status"] == "out_of_scope" for row in restricted)
    assert all(row["artifact"] == "model.safetensors" and not row["explicit_artifact"] for row in rows)
    public = json.dumps(publication) + json.dumps(rows)
    for private in ("suggested_fix", "diagnostic_detail", "graph_path", "private diagnostic marker", "private proposed fix"):
        assert private not in public
    assert not any(key in json.dumps(rows) for key in ("parameter_count", "tensor_count", "coverage"))


def test_immutable_import_is_idempotent_and_discovery_preserves_primary_report(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    raw = _csv([
        _source_row(1, "owner/model", "complete", "a" * 40),
        _source_row(2, "owner/restricted", "fail", "", reason_code="http_401", graph_count="0", node_count="0"),
    ])
    service = ExternalValidationService(jobs.db, clock=lambda: 1_900_000_000)
    first = service.publish(raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH)
    second = service.publish(raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH)
    assert first == second
    with jobs.db.read() as session:
        assert session.query(ExternalValidationPublication).count() == 1
    with jobs.db.write() as session:
        session.add(ModelReport(
            id="r" * 64, cache_identity=cache_key("owner/model", "a" * 40, jobs.settings.engine_version, "1.1"),
            model_id="owner/model", request_revision="main", revision="a" * 40,
            engine_version=jobs.settings.engine_version, schema_version="1.1", artifact="model.safetensors",
            explicit_artifact=False, blob_digest="b" * 64,
            summary={"result_status": "partial", "architecture": "VerifiedModel", "graph_count": 1,
                     "tensor_count": 7, "parameter_count": 99, "warning_count": 2}, inspected_at=1_800_000_000,
        ))
    result = DiscoveryCatalogService(jobs, clock=lambda: 2_000_000_000).get_catalog()
    trusted = next(row for row in result["results"] if row["model_id"] == "owner/model")
    assert trusted["source"] == "inspection" and trusted["status"] == "partial"
    assert trusted["stored_report_id"] == "r" * 64
    assert trusted["graph_count"] == 1 and trusted["tensor_count"] == 7
    assert trusted["external_observations"][0]["status"] == "reported_complete"
    assert trusted["external_observations"][0]["evidence_basis"] == "reported"
    assert trusted["external_observations"][0]["report_mode"] == "static_module_architecture"
    restricted = next(row for row in result["results"] if row["model_id"] == "owner/restricted")
    assert restricted["source"] == "external_validation" and restricted["status"] == "out_of_scope"
    assert restricted["revision"] is None and restricted["reference"] is None
    assert restricted["stored_report_id"] is None and restricted["parameter_counts"] == {"unique": None}
    assert restricted["scope_reason"] == "access_restricted"
    assert restricted["evidence_basis"] == "reported" and restricted["report_mode"] == "static_module_architecture"
    assert restricted["graph_count"] == 0 and restricted["node_count"] == 0
    assert restricted["source_rows"] == [2]
    assert result["external_publications"][0]["source_counts"]["input_rows"] == 2
    assert result["status_counts"]["out_of_scope"] == 1
    jobs.db.engine.dispose()


def test_import_rejects_unpinned_success_and_raw_status_mismatch():
    raw = _csv([_source_row(1, "owner/model", "complete", "")])
    try:
        prepare_publication(raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH)
    except ValueError as error:
        assert "omit a pinned revision" in str(error)
    else:
        raise AssertionError("unresolved success must be rejected")


def test_conflicting_duplicate_input_rows_are_kept_separately():
    raw = _csv([
        _source_row(1, "owner/model", "complete", "a" * 40),
        _source_row(
            2, "owner/model", "partial", "a" * 40,
            graph_generated="True", reason_code="architecture_incomplete",
        ),
    ])
    _, publication, rows = prepare_publication(
        raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH,
    )
    assert len(rows) == 2
    assert publication["metadata"]["source_counts"]["unique_observations"] == 2


def test_malformed_late_csv_row_leaves_publication_table_empty(tmp_path):
    jobs = JobService(Settings(data_dir=tmp_path))
    jobs.initialize()
    raw = _csv([
        _source_row(1, "owner/valid", "complete", "a" * 40),
        _source_row(2, "not-a-repository-id", "complete", "b" * 40),
    ])
    with pytest.raises(ValueError, match="invalid repository ID"):
        ExternalValidationService(jobs.db).publish(
            raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH,
        )
    with jobs.db.read() as session:
        assert session.query(ExternalValidationPublication).count() == 0
    jobs.db.engine.dispose()


def test_oversized_cli_file_is_rejected_before_profile_or_database_access(tmp_path):
    path = tmp_path / "oversized.csv"
    path.write_bytes(b"x" * (MAX_CSV_BYTES + 128))
    calls = []
    with pytest.raises(ValueError, match="size limit"):
        import_csv(
            path,
            source_repository=SOURCE_REPOSITORY,
            source_commit=SOURCE_COMMIT,
            source_path=SOURCE_PATH,
            env_file=tmp_path / "unused.env",
            profile_loader=lambda _path: calls.append("profile"),
            database_factory=lambda _url: calls.append("database"),
        )
    assert calls == []


def test_cli_uses_psycopg_url_and_imports_with_a_local_sqlite_factory(tmp_path):
    raw_path = tmp_path / "valid.csv"
    raw_path.write_bytes(_csv([_source_row(1, "owner/model", "complete", "a" * 40)]))
    local_database = Database(f"sqlite:///{(tmp_path / 'cli.sqlite3').as_posix()}")
    local_database.initialize(create=True)
    seen_urls = []

    def profile_loader(_path):
        return {
            "DATABASE_URL": "postgresql://catalog-user:catalog-pass@db.example.com:5432/model?sslmode=require",
            "MODEL_ANATOMY_SUPABASE_URL": "https://project.supabase.co",
            "MODEL_ANATOMY_SUPABASE_KEY": "test-service-role-key",
            "MODEL_ANATOMY_SUPABASE_BUCKET": "model-anatomy-private",
        }

    def database_factory(url):
        seen_urls.append(url)
        return local_database

    publication_id = import_csv(
        raw_path,
        source_repository=SOURCE_REPOSITORY,
        source_commit=SOURCE_COMMIT,
        source_path=SOURCE_PATH,
        env_file=tmp_path / "unused.env",
        profile_loader=profile_loader,
        database_factory=database_factory,
    )
    assert seen_urls == ["postgresql+psycopg://catalog-user:catalog-pass@db.example.com:5432/model?sslmode=require"]
    with local_database.read() as session:
        publication = session.get(ExternalValidationPublication, publication_id)
        assert publication is not None and publication.csv_sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()


def test_cli_redacts_database_factory_exception(tmp_path):
    raw_path = tmp_path / "valid.csv"
    raw_path.write_bytes(_csv([_source_row(1, "owner/model", "complete", "a" * 40)]))
    private_dsn = "postgresql://user:secret@private-db.example.com/model"

    def profile_loader(_path):
        return {
            "DATABASE_URL": "postgresql://catalog-user:catalog-pass@db.example.com:5432/model?sslmode=require",
            "MODEL_ANATOMY_SUPABASE_URL": "https://project.supabase.co",
            "MODEL_ANATOMY_SUPABASE_KEY": "test-service-role-key",
            "MODEL_ANATOMY_SUPABASE_BUCKET": "model-anatomy-private",
        }

    with pytest.raises(RuntimeError, match="inspect local service diagnostics") as error:
        import_csv(
            raw_path,
            source_repository=SOURCE_REPOSITORY,
            source_commit=SOURCE_COMMIT,
            source_path=SOURCE_PATH,
            env_file=tmp_path / "unused.env",
            profile_loader=profile_loader,
            database_factory=lambda _url: (_ for _ in ()).throw(RuntimeError(private_dsn)),
        )
    assert private_dsn not in str(error.value)


@pytest.mark.parametrize(
    ("status", "revision", "overrides", "message"),
    [
        ("complete", "a" * 40, {"mode": "artifact_inspection"}, "static module architecture"),
        ("complete", "a" * 40, {"reason_code": "architecture_incomplete"}, "complete observations"),
        ("complete", "a" * 40, {"graph_count": "0"}, "nonempty graph counts"),
        ("complete", "a" * 40, {"started_at": "2026-10-20T00:00:00"}, "timezone"),
        ("partial", "a" * 40, {"reason_code": "native_architecture_unavailable"}, "architecture_incomplete"),
        ("fail", "a" * 40, {"graph_generated": "True"}, "cannot claim a generated graph"),
        ("fail", "", {"reason_code": "safetensors_unavailable"}, "access-restricted observations"),
    ],
)
def test_validator_rejects_inconsistent_source_claims(status, revision, overrides, message):
    raw = _csv([_source_row(1, "owner/model", status, revision, **overrides)])
    with pytest.raises(ValueError, match=message):
        prepare_publication(
            raw, source_repository=SOURCE_REPOSITORY, source_commit=SOURCE_COMMIT, source_path=SOURCE_PATH,
        )
