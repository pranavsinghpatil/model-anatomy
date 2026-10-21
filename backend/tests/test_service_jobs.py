import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import select

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import Blob, CacheEntry, Guest, Job, Tombstone
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, cache_key, ir_summary
from model_anatomy.service.storage import LocalStorage


@pytest.fixture
def service(tmp_path):
    now = [10000.0]
    service = JobService(Settings(data_dir=tmp_path, guest_jobs_per_hour=20), clock=lambda: now[0])
    service.initialize()
    service.acquire_worker("worker-a")
    service.test_now = now
    yield service
    service.db.engine.dispose()


def visitor(service):
    return service.guest(None)[0]


def document():
    return json.loads((Path(__file__).resolve().parents[2] / "output/gpt2/model.json").read_text())


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.get_count = 0

    def put(self, data):
        import hashlib
        digest = hashlib.sha256(data).hexdigest()
        self.objects[digest] = data
        return digest

    def get(self, digest):
        self.get_count += 1
        return self.objects[digest]

    def delete(self, digest):
        self.objects.pop(digest, None)

    def check(self):
        pass


def finish(service, job_id):
    data = document()
    service.resolved(job_id, "worker-a", data["model_id"], data["metadata"]["revision"])
    key = cache_key(data["model_id"], data["metadata"]["revision"], service.settings.engine_version, "1.1")
    service.finish(job_id, "worker-a", json.dumps(data).encode(), key, cached=False)
    return key


def test_guests_are_persistent_unforgeable_and_isolated(service):
    first, token = service.guest(None)
    assert service.guest(token) == (first, None)
    assert service.guest("a" * 43)[0] != first
    detail = service.create(first, "gpt2")
    second = visitor(service)
    assert service.recent(second) == []
    for unknown in (detail["id"], "missing"):
        with pytest.raises(ServiceError) as exc:
            service.get(unknown, second)
        assert exc.value.status == 404
    reopened = JobService(service.settings, clock=service.clock)
    reopened.initialize()
    assert reopened.guest(token) == (first, None)
    assert reopened.get(detail["id"], first).status == "queued"
    reopened.db.engine.dispose()


def test_durable_registry_serves_private_ready_job_without_worker_or_cache_ttl(service):
    from model_anatomy.service.database import ModelReport

    service.storage = MemoryStorage()
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    key = finish(service, job_id)
    service.test_now[0] += service.settings.cache_ttl_seconds + 1
    service.cleanup()
    service.release_worker("worker-a")
    result = service.create(guest, "gpt2")
    assert result["status"] == "partial"
    assert result["cached"] is True
    assert result["revision"] == document()["metadata"]["revision"]
    with service.db.read() as session:
        assert session.scalar(select(ModelReport).where(ModelReport.cache_identity == key))
    with pytest.raises(ServiceError) as exc:
        service.get(result["id"], visitor(service))
    assert exc.value.status == 404


def test_finished_models_automatically_enter_recent_catalog_and_remain_revisitable(service):
    service.storage = MemoryStorage()
    guest = visitor(service)
    for name, inspected_at in [
        ("owner/aa-older", 10000.0),
        ("owner/zz-newer", 10001.0),
        ("owner/mm-newer", 10001.0),
    ]:
        service.test_now[0] = inspected_at
        data = document()
        data["model_id"] = name
        data["metadata"]["engine_version"] = service.settings.engine_version
        job_id = service.create(guest, name)["id"]
        assert service.claim("worker-a") == job_id
        revision = data["metadata"]["revision"]
        service.resolved(job_id, "worker-a", name, revision)
        key = cache_key(name, revision, service.settings.engine_version, service.settings.schema_version)
        service.finish(job_id, "worker-a", json.dumps(data).encode(), key, cached=False)

    pages = [service.catalog(offset=offset, limit=1) for offset in range(3)]
    rows = [page["models"][0] for page in pages]
    assert all(page["total"] == 3 for page in pages)
    assert {row["model_id"] for row in rows[:2]} == {"owner/zz-newer", "owner/mm-newer"}
    assert rows[2]["model_id"] == "owner/aa-older"
    assert len({row["id"] for row in rows}) == 3
    assert [service.catalog(offset=offset, limit=1)["models"][0]["id"] for offset in range(3)] == [
        row["id"] for row in rows
    ]
    assert service.catalog("aa-older")["total"] == 1

    service.test_now[0] += service.settings.job_ttl_seconds + service.settings.cache_ttl_seconds + 1
    service.cleanup()
    service.release_worker("worker-a")
    reopened = JobService(service.settings, clock=service.clock)
    reopened.storage = service.storage
    reopened.initialize()
    try:
        assert reopened.catalog()["total"] == 3
        saved = reopened.inspect_registry(visitor(reopened), rows[0]["id"])
        assert saved["cached"] is True
        assert saved["model_id"] == rows[0]["model_id"]
        assert saved["revision"] == rows[0]["revision"]
    finally:
        reopened.db.engine.dispose()


def test_default_main_does_not_reuse_a_report_requested_from_named_branch(service):
    service.storage = MemoryStorage()
    guest = visitor(service)
    reference = "https://huggingface.co/openai-community/gpt2/resolve/feature-branch/model.safetensors"
    job_id = service.create(guest, reference)["id"]
    service.claim("worker-a")
    finish(service, job_id)
    service.release_worker("worker-a")
    with pytest.raises(ServiceError) as exc:
        service.create(guest, "gpt2")
    assert exc.value.code == "worker_unavailable"


def test_exhausted_visitor_is_rejected_before_private_blob_read(service):
    from model_anatomy.service.database import RateBucket

    memory = MemoryStorage()
    service.storage = memory
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    finish(service, job_id)
    memory.get_count = 0
    with service.db.write() as session:
        bucket = session.get(RateBucket, (f"guest:{guest}:jobs:hour", int(service.clock() // 3600)))
        bucket.count = 20
    with pytest.raises(ServiceError) as exc:
        service.create(guest, "gpt2")
    assert exc.value.status == 429
    assert memory.get_count == 0


def test_registry_cache_identity_mismatch_is_invalidated(service):
    from model_anatomy.service.database import ModelReport

    service.storage = MemoryStorage()
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    key = finish(service, job_id)
    with service.db.write() as session:
        report = session.scalar(select(ModelReport).where(ModelReport.cache_identity == key))
        report.cache_identity = "f" * 64
    candidates = service._registry_candidates("openai-community/gpt2", "main", "model.safetensors", False)
    assert service._valid_registry(candidates) is None
    with service.db.read() as session:
        assert session.get(ModelReport, report.id) is None


def test_historical_registry_report_is_openable_but_not_an_automatic_cache_hit(service):
    from model_anatomy.service.database import Blob, ModelReport

    service.settings = service.settings.model_copy(
        update={"engine_version": "engine-3", "schema_version": "1.1"}
    )
    service.storage = MemoryStorage()
    guest = visitor(service)
    data = document()
    data["metadata"]["engine_version"] = "engine-2"
    payload = json.dumps(data).encode()
    digest = service.storage.put(payload)
    report_id = "a" * 64
    revision = data["metadata"]["revision"]
    summary = ir_summary(data)
    with service.db.write() as session:
        session.add(Blob(digest=digest, created_at=service.clock()))
        session.add(
            ModelReport(
                id=report_id,
                cache_identity=cache_key(data["model_id"], revision, "engine-2", "1.1"),
                model_id=data["model_id"],
                request_revision="main",
                revision=revision,
                engine_version="engine-2",
                schema_version="1.1",
                artifact="model.safetensors",
                explicit_artifact=False,
                blob_digest=digest,
                summary=summary,
                inspected_at=service.clock(),
            )
        )
        # Historical engine versions are supported for opening only while the report schema is current.
        session.add(
            ModelReport(
                id="b" * 64,
                cache_identity=cache_key(data["model_id"], revision, "engine-1", "1.0"),
                model_id=data["model_id"],
                request_revision="main",
                revision=revision,
                engine_version="engine-1",
                schema_version="1.0",
                artifact="model.safetensors",
                explicit_artifact=False,
                blob_digest=digest,
                summary=summary,
                inspected_at=service.clock(),
            )
        )

    opened = service.inspect_registry(guest, report_id)
    stored_job = service.get(opened["id"], guest)
    assert opened["cached"] is True
    assert opened["engine_version"] == "engine-2"
    assert stored_job.engine_version == "engine-2"
    assert [(row["id"], row["engine_version"]) for row in service.catalog()["models"]] == [
        (report_id, "engine-2")
    ]

    # The current engine cache lookup remains strict and queues a fresh reconstruction.
    queued = service.create(guest, "gpt2")
    assert queued["status"] == "queued"
    assert queued["cached"] is False
    assert queued["engine_version"] == "engine-3"
    with pytest.raises(ServiceError) as exc:
        service.inspect_registry(guest, "b" * 64)
    assert exc.value.status == 404


def test_existing_local_sqlite_jobs_gain_refresh_flag_without_data_loss(tmp_path):
    service = JobService(Settings(data_dir=tmp_path))
    service.initialize()
    service.acquire_worker("worker")
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    with service.db.engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE analysis_jobs DROP COLUMN refresh_requested")
    service.db.initialize(create=True)
    with service.db.read() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.model_id == "openai-community/gpt2"
        assert job.refresh_requested is False
    service.db.engine.dispose()


def test_legacy_summary_is_reprojected_without_rewriting_saved_job_or_artifact(service, monkeypatch):
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    finish(service, job_id)
    with service.db.write() as session:
        job = session.get(Job, job_id)
        old_summary = dict(job.summary)
        for key in ("quality", "limitations", "partial_reasons", "engine_version"):
            old_summary.pop(key, None)
        old_summary["warnings"] += document()["metadata"]["limitations"]
        job.summary = old_summary
        digest = job.blob_digest
    original = service.storage.get(digest)
    calls = []
    get = service.storage.get
    monkeypatch.setattr(service.storage, "get", lambda digest: calls.append(digest) or get(digest))
    detail = service.present(service.get(job_id, guest))
    recent = service.recent(guest)[0]
    assert detail == recent
    assert len(calls) == 1
    assert detail["warnings"] == document()["metadata"]["analysis_report"]["warnings"]
    assert detail["limitations"] == document()["metadata"]["limitations"]
    assert detail["quality"]["checkpoint_parameter_coverage"] == 1
    assert service.get(job_id, guest).summary == old_summary
    assert get(digest) == original
    with pytest.raises(ServiceError) as exc:
        service.get(job_id, visitor(service))
    assert exc.value.status == 404
    assert len(calls) == 1


def test_legacy_terminal_status_matches_artifact_assessment_without_database_rewrite(service):
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    finish(service, job_id)
    with service.db.write() as session:
        job = session.get(Job, job_id)
        job.status = job.stage = "completed"
        job.summary = {"warnings": []}
    detail = service.present(service.get(job_id, guest))
    assert detail["status"] == detail["stage"] == "partial"
    assert detail["warnings"] == document()["metadata"]["analysis_report"]["warnings"]
    assert service.cancel(job_id, guest) == detail
    persisted = service.get(job_id, guest)
    assert persisted.status == persisted.stage == "completed"
    assert persisted.summary == {"warnings": []}


def test_summary_predating_input_applicability_is_read_from_immutable_report(service):
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    finish(service, job_id)
    with service.db.write() as session:
        job = session.get(Job, job_id)
        summary = dict(job.summary)
        summary.pop("input_names")
        job.summary = summary
    expected = sorted(document()["metadata"]["inputs"]["input_specs"])
    assert service.present(service.get(job_id, guest))["input_names"] == expected
    assert "input_names" not in service.get(job_id, guest).summary


def test_global_worker_lease_and_claim_serialize_independent_connections(service):
    owner = visitor(service)
    jobs = [service.create(owner, "gpt2")["id"] for _ in range(2)]
    other = JobService(service.settings, clock=service.clock)
    other.initialize()
    assert other.acquire_worker("worker-b") is False
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda s: s.claim("worker-a"), [service, other]))
    assert sorted(x for x in claimed if x) == jobs[:1]
    assert service.claim("worker-a") is None
    service.fail(jobs[0], "worker-a", "unsupported", "unsupported", status="unsupported")
    assert other.claim("worker-a") == jobs[1]
    other.db.engine.dispose()


def test_lease_recovery_fences_old_children_and_bounds_restarts(service):
    owner = visitor(service)
    job_id = service.create(owner, "gpt2")["id"]
    assert service.claim("worker-a") == job_id
    service.test_now[0] += service.settings.lease_seconds + 1
    assert service.acquire_worker("worker-b")
    assert service.claim("worker-b") == job_id
    with pytest.raises(ServiceError, match="lease"):
        service.stage(job_id, "worker-a", "storing")
    service.test_now[0] += service.settings.lease_seconds + 1
    assert service.acquire_worker("worker-c")
    assert service.claim("worker-c") is None
    assert service.get(job_id, owner).error["code"] == "worker_interrupted"


def test_retry_only_transient_is_bounded_and_delayed(service):
    owner = visitor(service)
    job_id = service.create(owner, "gpt2")["id"]
    service.claim("worker-a")
    service.fail(job_id, "worker-a", "artifact_network", "unavailable", transient=True)
    assert service.get(job_id, owner).status == "queued"
    assert service.claim("worker-a") is None
    service.test_now[0] += 3
    assert service.claim("worker-a") == job_id
    service.fail(job_id, "worker-a", "artifact_network", "unavailable", transient=True)
    assert service.get(job_id, owner).status == "failed"
    assert service.claim("worker-a") is None


def test_quota_is_atomic_global_guest_and_persists_restarts(tmp_path):
    settings = Settings(data_dir=tmp_path, guest_jobs_per_hour=1, global_jobs_per_day=2)
    service = JobService(settings)
    service.initialize()
    service.acquire_worker("w")
    first = visitor(service)
    service.create(first, "gpt2")
    with pytest.raises(ServiceError) as exc:
        service.create(first, "gpt2")
    assert exc.value.status == 429
    service.create(visitor(service), "gpt2")
    reopened = JobService(settings)
    reopened.initialize()
    with pytest.raises(ServiceError) as exc:
        reopened.create(visitor(service), "gpt2")
    assert exc.value.status == 429
    assert len(service.recent(first)) == 1
    service.db.engine.dispose()
    reopened.db.engine.dispose()


def test_queue_full_does_not_spend_quota_and_worker_offline(service):
    service.settings = service.settings.model_copy(update={"queue_capacity": 1})
    guest = visitor(service)
    first = service.create(guest, "gpt2")
    with pytest.raises(ServiceError) as exc:
        service.create(guest, "gpt2")
    assert exc.value.code == "queue_full"
    service.cancel(first["id"], guest)
    assert service.create(guest, "gpt2")["status"] == "queued"
    service.release_worker("worker-a")
    with pytest.raises(ServiceError) as exc:
        service.create(guest, "gpt2")
    assert exc.value.code == "worker_unavailable"


def test_cancel_queued_running_and_completed(service):
    guest = visitor(service)
    first = service.create(guest, "gpt2")["id"]
    assert service.cancel(first, guest)["status"] == "cancelled"
    assert service.claim("worker-a") is None
    running = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    assert service.cancel(running, guest)["stage"] == "cancelling"
    assert service.heartbeat("worker-a", running)
    finish(service, running)
    assert service.get(running, guest).status == "cancelled"
    complete = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    finish(service, complete)
    assert service.cancel(complete, guest)["status"] in {"completed", "partial"}


def test_expiry_is_410_for_owner_404_for_other_and_retention_is_seven_days(service):
    from model_anatomy.service.database import ModelReport

    memory = MemoryStorage()
    service.storage = memory
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    key = finish(service, job_id)
    digest = service.get(job_id, guest).blob_digest
    service.test_now[0] += 86401
    with pytest.raises(ServiceError) as exc:
        service.get(job_id, guest)
    assert exc.value.status == 410
    with pytest.raises(ServiceError) as exc:
        service.get(job_id, visitor(service))
    assert exc.value.status == 404
    assert service.recent(guest) == []
    service.cleanup()
    assert service.cached_result(key) is not None
    assert digest in memory.objects
    service.test_now[0] += 604800
    service.cleanup()
    assert service.cached_result(key) is None
    assert digest in memory.objects
    with service.db.read() as session:
        report = session.scalar(select(ModelReport).where(ModelReport.cache_identity == key))
        assert report is not None
        assert list(session.scalars(select(CacheEntry))) == []
        report_id = report.id
    with service.db.write() as session:
        session.delete(session.get(ModelReport, report_id))
    service.cleanup()
    assert digest not in memory.objects
    with service.db.read() as session:
        assert list(session.scalars(select(Blob))) == []
        assert list(session.scalars(select(ModelReport))) == []


def test_revision_and_artifact_and_engine_and_schema_are_cache_identity():
    args = ["namespace/model", "a" * 40, "engine-1", "1.1"]
    original = cache_key(*args)
    for index, value in enumerate(["namespace/other", "b" * 40, "engine-2", "1.2"]):
        changed = args.copy()
        changed[index] = value
        assert cache_key(*changed) != original
    assert cache_key(*args, "shard-0001.safetensors", True) != original
    assert cache_key(*args, "model.safetensors", True) != original
    with pytest.raises(ValueError):
        cache_key("namespace/model", "main", "engine", "1.1")


def test_pinned_retry_cannot_switch_revision(service):
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    service.resolved(job_id, "worker-a", "openai-community/gpt2", "a" * 40)
    with pytest.raises(ServiceError) as exc:
        service.resolved(job_id, "worker-a", "openai-community/gpt2", "b" * 40)
    assert exc.value.code == "revision_changed"


def test_store_atomicity_content_addressing_and_corruption(tmp_path):
    storage = LocalStorage(tmp_path, max_bytes=1024)
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = list(pool.map(storage.put, [b'{"fact":1}'] * 8))
    assert len(set(digests)) == 1
    assert storage.get(digests[0]) == b'{"fact":1}'
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(ValueError):
        storage.get("../../escape")
    storage.path(digests[0]).write_bytes(b"corrupted")
    with pytest.raises(ServiceError) as exc:
        storage.get(digests[0])
    assert exc.value.code == "artifact_corrupt"
    with pytest.raises(ServiceError):
        storage.put(b"a" * 1025)


def test_partial_when_coverage_is_incomplete_without_fabricated_warning():
    data = document()
    data["metadata"]["analysis_report"] = {"tensor_shape_coverage": 0.95}
    assert ir_summary(data)["result_status"] == "partial"


def test_waiting_capacity_excludes_running_job(service):
    service.settings = service.settings.model_copy(update={"queue_capacity": 1})
    guest = visitor(service)
    running = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    assert service.get(running, guest).status == "running"
    assert service.create(guest, "gpt2")["status"] == "queued"
    with pytest.raises(ServiceError) as exc:
        service.create(guest, "gpt2")
    assert exc.value.code == "queue_full"


def test_global_daily_quota_does_not_reset_with_guest_hour(tmp_path):
    now = [10 * 86400 + 100]
    service = JobService(
        Settings(data_dir=tmp_path, global_jobs_per_day=2, guest_jobs_per_hour=1), clock=lambda: now[0]
    )
    service.initialize()
    service.acquire_worker("w")
    guest = visitor(service)
    service.create(guest, "gpt2")
    now[0] += 3600
    service.acquire_worker("w")
    service.cleanup()
    service.create(guest, "gpt2")
    now[0] += 3600
    service.acquire_worker("w")
    service.cleanup()
    with pytest.raises(ServiceError) as exc:
        service.create(visitor(service), "gpt2")
    assert exc.value.status == 429
    now[0] += 86400
    service.acquire_worker("w")
    service.cleanup()
    assert service.create(visitor(service), "gpt2")["status"] == "queued"
    service.db.engine.dispose()


def test_retention_deletes_jobs_and_guests_and_bounded_tombstones(service):
    guest, token = service.guest(None)
    unused = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.test_now[0] += 86401
    service.cleanup()
    with service.db.read() as session:
        assert session.get(Job, job_id) is None
        assert session.get(Guest, unused) is None
        assert session.get(Tombstone, job_id) is not None
    assert service.guest(token) == (guest, None)
    with pytest.raises(ServiceError) as exc:
        service.get(job_id, guest)
    assert exc.value.status == 410
    service.test_now[0] += 86401
    service.cleanup()
    with service.db.read() as session:
        assert session.get(Tombstone, job_id) is None
        assert session.get(Guest, guest) is None


def test_storage_io_does_not_hold_database_mutex_and_cancellation_fences_result(service):
    guest = visitor(service)
    job_id = service.create(guest, "gpt2")["id"]
    service.claim("worker-a")
    original_put = service.storage.put

    def cancel_during_upload(data):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(service.cancel, job_id, guest).result(timeout=2)["stage"] == "cancelling"
        return original_put(data)

    service.storage.put = cancel_during_upload
    finish(service, job_id)
    assert service.get(job_id, guest).status == "cancelled"
    service.test_now[0] += service.settings.job_timeout_seconds + service.settings.lease_seconds + 1
    original_delete = service.storage.delete

    def heartbeat_during_delete(digest):
        assert service.acquire_worker("worker-a")
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(service.heartbeat, "worker-a").result(timeout=2) is False
        original_delete(digest)

    service.storage.delete = heartbeat_during_delete
    service.cleanup()
    with service.db.read() as session:
        assert list(session.scalars(select(Blob))) == []
