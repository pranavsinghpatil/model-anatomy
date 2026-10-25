import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from test_service_views import contract_report

from model_anatomy.api.settings import Settings
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, cache_key
from model_anatomy.service.views import load_ir
from model_anatomy.service.worker import (
    Worker,
    classify_failure,
    database_connection_lost,
    enforce_limits,
    execute_job,
)


@dataclass
class Reference:
    model_id: str
    revision: str = "main"
    filename: str = "model.safetensors"
    explicit_filename: bool = False


class MetadataMock:
    def __init__(self, revision):
        self.revision, self.resolved = revision, []

    def resolve(self, reference):
        self.resolved.append(reference)
        return self.revision


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def put(self, data):
        import hashlib
        digest = hashlib.sha256(data).hexdigest()
        self.objects[digest] = data
        return digest

    def get(self, digest):
        return self.objects[digest]

    def delete(self, digest):
        self.objects.pop(digest, None)

    def check(self):
        pass


def test_resolve_once_before_extraction_and_cache_hit_skips_engine(tmp_path):
    settings = Settings(data_dir=tmp_path, guest_jobs_per_hour=5)
    service = JobService(settings)
    service.initialize()
    service.storage = MemoryStorage()
    service.acquire_worker("w")
    guest, _ = service.guest(None)
    ir = load_ir(contract_report().to_json().encode())
    metadata = MetadataMock(ir.metadata["revision"])
    calls = []

    def reconstruct(value, *, client, revision, on_progress):
        calls.append(value)
        assert client is metadata and revision == ir.metadata["revision"]
        on_progress("building_structure")
        return ir

    def parse(_):
        return Reference(ir.model_id)

    first = service.create(guest, "gpt2")["id"]
    service.claim("w")
    execute_job(service, first, "w", metadata, reconstruct, parse)
    second = service.create(guest, "gpt2")["id"]
    assert service.get(second, guest).status == "completed"
    assert service.get(second, guest).attempts == 0
    assert len(calls) == 1
    assert len(metadata.resolved) == 1
    assert service.get(second, guest).cached is True
    assert service.get(first, guest).blob_digest == service.get(second, guest).blob_digest
    third = service.create(guest, "gpt2", refresh=True)["id"]
    service.claim("w")
    execute_job(service, third, "w", metadata, reconstruct, parse)
    assert len(calls) == 2
    assert len(metadata.resolved) == 2
    assert metadata.resolved[-1].revision == "main"
    assert service.get(third, guest).cached is False
    service.db.engine.dispose()


def test_worker_requests_ignore_global_netrc_credentials(tmp_path, monkeypatch):
    import requests

    netrc = tmp_path / "credentials"
    netrc.parent.mkdir(parents=True, exist_ok=True)
    netrc.write_text("machine huggingface.co login private-user password private-token\n")
    monkeypatch.setenv("NETRC", str(netrc))
    session = requests.Session()
    session.trust_env = False
    prepared = session.prepare_request(requests.Request("GET", "https://huggingface.co/api/models/acme/public"))
    assert "Authorization" not in prepared.headers


def sleeping_child(settings, job_id, owner):
    time.sleep(20)


def successful_child(settings, job_id, owner):
    service = JobService(settings)
    data = contract_report().to_json().encode()
    document = json.loads(data)
    service.resolved(job_id, owner, document["model_id"], document["metadata"]["revision"])
    key = cache_key(document["model_id"], document["metadata"]["revision"], settings.engine_version, "1.1")
    service.finish(job_id, owner, data, key, cached=False)
    service.db.engine.dispose()


@pytest.mark.parametrize(
    "target,timeout,status", [(successful_child, 10, "completed"), (sleeping_child, 1, "timed_out")]
)
def test_real_spawn_child_finishes_or_is_killed(tmp_path, monkeypatch, target, timeout, status):
    import model_anatomy.service.worker as worker_module

    monkeypatch.setattr(worker_module, "child_main", target)
    settings = Settings(data_dir=tmp_path, job_timeout_seconds=timeout)
    service = JobService(settings)
    service.initialize()
    worker = Worker(settings, service=service)
    service.acquire_worker(worker.owner)
    guest, _ = service.guest(None)
    job_id = service.create(guest, "gpt2")["id"]
    assert service.claim(worker.owner) == job_id
    worker.run_job(job_id)
    assert service.get(job_id, guest).status == status
    assert worker.process is None
    service.db.engine.dispose()


def test_linux_limits_are_set_before_model_imports(monkeypatch):
    import model_anatomy.service.worker as worker_module

    calls = []
    resource = SimpleNamespace(
        RLIMIT_AS=1, RLIMIT_CPU=2, RLIMIT_CORE=3, setrlimit=lambda kind, limits: calls.append((kind, limits))
    )
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(worker_module.sys, "platform", "linux")
    monkeypatch.setattr(worker_module.os, "setsid", lambda: calls.append("setsid"), raising=False)
    enforce_limits(Settings())
    assert calls == ["setsid", (1, (4 * 1024**3, 4 * 1024**3)), (2, (121, 121)), (3, (0, 0))]


def test_production_worker_cannot_silently_ignore_memory_enforcement(monkeypatch):
    import model_anatomy.service.worker as worker_module

    monkeypatch.setattr(worker_module.sys, "platform", "win32")
    settings = Settings().model_copy(update={"environment": "production"})
    with pytest.raises(RuntimeError, match="Linux"):
        Worker(settings)


def test_failure_classifier_does_not_retry_unsupported_or_credentials():
    import requests

    assert classify_failure(requests.Timeout())[3] is True
    for status in (401, 403, 404):
        response = requests.Response()
        response.status_code = status
        assert classify_failure(requests.HTTPError(response=response))[3] is False
    assert classify_failure(MemoryError())[0] == "memory_limit"
    assert classify_failure(ServiceError(503, "artifact_corrupt", "corrupt"))[3] is False


def test_alembic_upgrade_schema_matches_models_and_downgrade(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "MODEL_ANATOMY_DATA_DIR": str(tmp_path),
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'migration.sqlite3').as_posix()}",
    }
    command = [sys.executable, "-m", "alembic", "-c", str(root / "alembic.ini")]
    result = subprocess.run(
        [*command, "upgrade", "head"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from model_anatomy.service.database import Base, Database

    database = Database(env["DATABASE_URL"])
    with database.engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
    database.engine.dispose()
    result = subprocess.run(
        [*command, "downgrade", "base"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "analysis_jobs" not in inspect(database.engine).get_table_names()
    database.engine.dispose()


def test_lifetime_worker_lock_excludes_other_connections(tmp_path):
    settings = Settings(data_dir=tmp_path)
    first, second = JobService(settings), JobService(settings)
    first.initialize()
    second.initialize()
    with first.db.worker_lock():
        with pytest.raises(ServiceError) as exc, second.db.worker_lock():
            pass
        assert exc.value.code == "worker_busy"
    with second.db.worker_lock():
        assert second.acquire_worker("second")
    first.db.engine.dispose()
    second.db.engine.dispose()


def test_idle_worker_recovers_expired_lease_after_computer_suspend(tmp_path, monkeypatch):
    now = [1000.0]
    service = JobService(Settings(data_dir=tmp_path), clock=lambda: now[0])
    worker = Worker(service.settings, service=service)
    heartbeat = service.heartbeat
    calls = []

    def suspend_once(owner, job_id=None):
        calls.append(owner)
        if len(calls) == 1:
            now[0] += service.settings.lease_seconds + 1
        return heartbeat(owner, job_id)

    monkeypatch.setattr(service, "heartbeat", suspend_once)
    worker.run(once=True)
    assert calls == [worker.owner]
    assert not service.worker_available()


@pytest.mark.parametrize("error", [ServiceError(503, "storage_unavailable", "SECRET"), OSError("SECRET")])
def test_cleanup_outage_does_not_stop_queued_work(tmp_path, monkeypatch, caplog, error):
    service = JobService(Settings(data_dir=tmp_path))
    service.initialize()
    worker = Worker(service.settings, service=service)
    assert service.acquire_worker(worker.owner)
    guest, _ = service.guest(None)
    job_id = service.create(guest, "gpt2")["id"]
    processed = []

    def unavailable():
        raise error

    def process(claimed):
        assert service.worker_available()
        processed.append(claimed)
        service.fail(claimed, worker.owner, "test_result", "Test result", status="unsupported")

    monkeypatch.setattr(service, "cleanup", unavailable)
    monkeypatch.setattr(worker, "run_job", process)
    worker.run(once=True)
    assert processed == [job_id]
    assert service.get(job_id, guest).status == "unsupported"
    assert "retention cleanup deferred" in caplog.text
    assert "SECRET" not in caplog.text
    service.db.engine.dispose()


def test_cleanup_retries_back_off_and_recover_without_worker_restart(tmp_path, monkeypatch):
    import model_anatomy.service.worker as worker_module

    service = JobService(Settings(data_dir=tmp_path))
    worker = Worker(service.settings, service=service)
    now = [0.0]
    attempts = []
    cleanup = service.cleanup

    def retry_cleanup():
        attempts.append(now[0])
        if len(attempts) <= 7:
            raise ServiceError(503, "storage_unavailable", "Temporary outage")
        cleanup()
        worker.request_stop()

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(worker_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(worker_module.time, "sleep", advance)
    monkeypatch.setattr(service, "cleanup", retry_cleanup)
    worker.run()
    assert attempts == [0, 1, 3, 7, 15, 31, 61, 91]
    assert not service.worker_available()


def test_idle_cleanup_can_outlive_lease_without_stopping_worker(tmp_path, monkeypatch):
    now = [1000.0]
    service = JobService(Settings(data_dir=tmp_path), clock=lambda: now[0])
    worker = Worker(service.settings, service=service)

    def slow_cleanup():
        now[0] += service.settings.lease_seconds + 1

    monkeypatch.setattr(service, "cleanup", slow_cleanup)
    worker.run(once=True)
    assert not service.worker_available()


def test_cleanup_configuration_errors_are_not_silently_retried(tmp_path, monkeypatch):
    service = JobService(Settings(data_dir=tmp_path))
    worker = Worker(service.settings, service=service)

    def invalid_storage():
        raise ServiceError(503, "private_storage_required", "Invalid storage configuration")

    monkeypatch.setattr(service, "cleanup", invalid_storage)
    with pytest.raises(ServiceError) as exc:
        worker.run(once=True)
    assert exc.value.code == "private_storage_required"


def test_idle_worker_does_not_steal_another_live_lease(tmp_path, monkeypatch):
    service = JobService(Settings(data_dir=tmp_path))
    worker = Worker(service.settings, service=service)
    heartbeat = service.heartbeat

    def replace_owner(owner, job_id=None):
        service.release_worker(owner)
        assert service.acquire_worker("other")
        return heartbeat(owner, job_id)

    monkeypatch.setattr(service, "heartbeat", replace_owner)
    with pytest.raises(ServiceError, match="lease"):
        worker.run(once=True)
    assert service.worker_available()
    service.release_worker("other")
    service.db.engine.dispose()


def test_worker_reacquires_lock_and_lease_after_database_transport_loss(monkeypatch):
    import model_anatomy.service.worker as worker_module

    class TransportError(Exception):
        sqlstate = None

    disconnected = OperationalError("SELECT pg_advisory_unlock", {}, TransportError("DNS failure"))
    calls = []

    @contextmanager
    def worker_lock():
        calls.append("lock")
        yield
        if calls.count("lock") == 1:
            raise disconnected

    db = SimpleNamespace(
        sqlite=False,
        worker_lock=worker_lock,
        engine=SimpleNamespace(dispose=lambda: calls.append("dispose")),
    )
    service = SimpleNamespace(db=db, initialize=lambda: calls.append("initialize"))
    worker = Worker(Settings(), service=service)
    monkeypatch.setattr(worker, "run_locked", lambda **kwargs: calls.append("lease_and_work"))
    monkeypatch.setattr(worker_module.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))

    worker.run()

    assert calls == [
        "initialize",
        "lock",
        "lease_and_work",
        "dispose",
        ("sleep", 1.0),
        "initialize",
        "lock",
        "lease_and_work",
        "dispose",
    ]


def test_worker_database_retry_rejects_sql_errors_and_sqlite():
    class ServerError(Exception):
        sqlstate = "42P01"

    class TransportError(Exception):
        sqlstate = None

    server_error = OperationalError("SELECT", {}, ServerError("undefined table"))
    transport_error = OperationalError("SELECT", {}, TransportError("connection failed"))
    assert database_connection_lost(transport_error, sqlite=False)
    assert not database_connection_lost(server_error, sqlite=False)
    assert not database_connection_lost(transport_error, sqlite=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Production meta-import memory gate requires real Linux")
def test_linux_real_tiny_meta_model_under_production_address_limit():
    backend = Path(__file__).resolve().parents[1]
    script = """
import json, os, resource
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
from model_anatomy.api.settings import Settings
from model_anatomy.service.worker import enforce_limits
enforce_limits(Settings())
from model_anatomy.src.registry import instantiate_native, load_native_config
config = load_native_config({'model_type':'gpt2', 'vocab_size':32, 'n_positions':16,
                            'n_embd':8, 'n_layer':1, 'n_head':2})
model, selection = instantiate_native(config)
assert all(p.device.type == 'meta' for p in model.parameters())
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert rss < 4 * 1024**3
print(json.dumps({'meta':True, 'peak_rss_bytes':rss}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=backend, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["meta"] is True
