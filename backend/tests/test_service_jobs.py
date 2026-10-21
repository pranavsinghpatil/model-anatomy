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
    assert service.storage.path(digest).is_file()
    service.test_now[0] += 604800
    service.cleanup()
    assert service.cached_result(key) is None
    assert not service.storage.path(digest).exists()
    with service.db.read() as session:
        assert list(session.scalars(select(Blob))) == []
        assert list(session.scalars(select(CacheEntry))) == []


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
