import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest
from scripts import store_benchmark_reports as reports

from model_anatomy.src.version import ENGINE_VERSION


def test_failed_case_is_kept_as_evidence_without_report(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": [{"model_id": "owner/model", "category": "text", "revision": "a" * 40}]}),
        encoding="utf-8",
    )

    def fail_case(*args, **kwargs):
        return {"status": "network", "stage": "reconstruction", "cause_type": "ConnectionError", "model_id": "owner/model"}

    monkeypatch.setattr(reports.model_matrix, "run_case", fail_case)
    summary = reports.capture_only(manifest, tmp_path / "rows.ndjson", tmp_path / "reports", allow_small=True)
    row = summary["results"][0]
    assert row["status"] == "network"
    assert "report_digest" not in row
    assert list((tmp_path / "reports").glob("*.json")) == []


def test_capture_returns_digest_record_and_persists_report_file(monkeypatch, tmp_path):
    case = {"model_id": "owner/model", "category": "text", "revision": "a" * 40}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case]}), encoding="utf-8")
    reports_dir = tmp_path / "reports"
    report_bytes = b'{"model_id":"owner/model"}'
    digest = hashlib.sha256(report_bytes).hexdigest()

    def fake_run_case(worker_case, **kwargs):
        Path(worker_case["__report_directory"]).mkdir(parents=True, exist_ok=True)
        (Path(worker_case["__report_directory"]) / f"{digest}.json").write_bytes(report_bytes)
        return {
            "status": "partial",
            "model_id": "owner/model",
            "report_digest": digest,
            "serialized_report_bytes": len(report_bytes),
            "service_contract_valid": True,
        }

    monkeypatch.setattr(reports.model_matrix, "run_case", fake_run_case)
    summary = reports.capture_only(manifest, tmp_path / "rows.ndjson", reports_dir, allow_small=True)
    assert summary["results"][0]["report_digest"] == digest
    assert (reports_dir / f"{digest}.json").read_bytes() == report_bytes


def test_capture_resume_requires_verified_digest(monkeypatch, tmp_path):
    case = {"model_id": "owner/model", "category": "text", "revision": "a" * 40}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case]}), encoding="utf-8")
    output = tmp_path / "rows.ndjson"
    reports_dir = tmp_path / "reports"
    calls = 0

    def fake_run_case(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"status": "completed", "model_id": "owner/model", "report_digest": "b" * 64}

    monkeypatch.setattr(reports.model_matrix, "run_case", fake_run_case)
    reports.capture_only(manifest, output, reports_dir, allow_small=True)
    monkeypatch.setattr(reports, "_verified_report", lambda *_: True)
    second = reports.capture_only(manifest, output, reports_dir, allow_small=True, resume=True)
    assert calls == 1
    assert second["results"][0]["resumed"] is True


def test_corrupt_digest_and_mismatched_pin_are_rejected(monkeypatch, tmp_path):
    case = {"model_id": "owner/model", "category": "text", "revision": "a" * 40}
    digest = hashlib.sha256(b"{}").hexdigest()
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / f"{digest}.json").write_bytes(b"corrupt")
    row = {
        "status": "completed",
        "service_contract_valid": True,
        "report_digest": digest,
        "serialized_report_bytes": 2,
    }
    assert not reports._verified_report(case, row, report_dir)


    data = {"model_id": "owner/model", "schema_version": "1.1"}
    encoded = json.dumps(data).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    (report_dir / f"{digest}.json").write_bytes(encoded)

    class FakeIR:
        model_id = "owner/model"
        schema_version = "1.1"
        metadata: ClassVar[dict] = {"revision": "c" * 40, "engine_version": ENGINE_VERSION}

        class Graph:
            @staticmethod
            def validate():
                return None

        root_graph = Graph()

        @staticmethod
        def from_json(_):
            return FakeIR()

        @staticmethod
        def to_dict():
            return data

    monkeypatch.setattr("model_anatomy.src.ir.ModelIR.from_json", FakeIR.from_json)
    row.update(
        {
            "status": "completed",
            "service_contract_valid": True,
            "report_digest": digest,
            "serialized_report_bytes": len(encoded),
            "engine_version": FakeIR.metadata["engine_version"],
            "report_roundtrip": {"valid": True, "schema_version": "1.1"},
            "storage_contract_valid": True,
            "weight_payload_bytes_read": 0,
            "forward_executed": False,
        }
    )
    assert not reports._verified_report(case, row, report_dir)


@pytest.mark.parametrize("corrupt_readback", [False, True])
def test_store_commits_only_verified_blobs_and_reuses_existing_reports(monkeypatch, tmp_path, corrupt_readback):
    from sqlalchemy import func, select

    from model_anatomy.service.database import Database, ModelReport

    database = Database(f"sqlite:///{(tmp_path / 'jobs.sqlite3').as_posix()}")
    database.initialize(create=True)
    case = {"model_id": "owner/model", "category": "text", "revision": "a" * 40}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    records = tmp_path / "records.ndjson"
    records.write_text(json.dumps({**case, "status": "partial", "service_contract_valid": True}) + "\n")
    data = b'{"schema_version":"1.1"}'
    digest = hashlib.sha256(data).hexdigest()
    class Storage:
        def __init__(self):
            self.uploads = 0
        def check(self):
            pass
        def put(self, _):
            self.uploads += 1
            return digest
        def get(self, _):
            return b"corrupt" if corrupt_readback else data
    storage = Storage()
    monkeypatch.setattr("model_anatomy.service.database.Database", lambda _: database)
    monkeypatch.setattr("model_anatomy.service.storage.storage_for", lambda _: storage)
    monkeypatch.setattr("scripts.supabase_profile.load_supabase_profile", lambda _: {
        "DATABASE_URL": "postgresql://user:test@localhost:5432/postgres?sslmode=require",
        "MODEL_ANATOMY_SUPABASE_URL": "https://storage.example",
        "MODEL_ANATOMY_SUPABASE_KEY": "test-only",
        "MODEL_ANATOMY_SUPABASE_BUCKET": "private",
    })
    monkeypatch.setattr(reports, "_load_report", lambda *_: (data, json.loads(data)))
    monkeypatch.setattr("model_anatomy.service.jobs.ir_summary", lambda _: {"result_status": "partial"})
    arguments = (manifest, records, tmp_path, tmp_path / "unused.env", tmp_path / "progress.ndjson")
    first = reports.store_only(*arguments)
    with database.read() as session:
        assert session.scalar(select(func.count()).select_from(ModelReport)) == (0 if corrupt_readback else 1)
    if corrupt_readback:
        assert first["failed"] == 1
        assert json.loads(arguments[-1].read_text().splitlines()[-1])["stage"] == "storage_readback"
    else:
        assert first["stored"] == 1
        second = reports.store_only(*arguments)
        assert second["reused"] == 1
        assert storage.uploads == 1
    database.engine.dispose()
