import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import Job
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, cache_key
from model_anatomy.service.views import load_ir
from model_anatomy.service.worker import Worker, classify_failure, enforce_limits, execute_job


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


def test_resolve_once_before_extraction_and_cache_hit_skips_engine(tmp_path):
    settings = Settings(data_dir=tmp_path, guest_jobs_per_hour=5)
    service = JobService(settings)
    service.initialize()
    service.acquire_worker("w")
    guest, _ = service.guest(None)
    ir = load_ir((settings.prepared_dir / "gpt2/model.json").read_bytes())
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
    service.claim("w")
    execute_job(service, second, "w", metadata, reconstruct, parse)
    assert len(calls) == 1
    assert len(metadata.resolved) == 2
    assert service.get(second, guest).cached is True
    assert service.get(first, guest).blob_digest == service.get(second, guest).blob_digest
    third = service.create(guest, "gpt2")["id"]
    service.claim("w")
    with service.db.write() as session:
        session.get(Job, third).revision = ir.metadata["revision"]
    execute_job(service, third, "w", metadata, reconstruct, parse)
    assert metadata.resolved[-1].revision == ir.metadata["revision"]
    service.db.engine.dispose()


def sleeping_child(settings, job_id, owner):
    time.sleep(20)


def successful_child(settings, job_id, owner):
    service = JobService(settings)
    data = (settings.prepared_dir / "gpt2/model.json").read_bytes()
    document = json.loads(data)
    service.resolved(job_id, owner, document["model_id"], document["metadata"]["revision"])
    key = cache_key(document["model_id"], document["metadata"]["revision"], settings.engine_version, "1.1")
    service.finish(job_id, owner, data, key, cached=False)
    service.db.engine.dispose()


@pytest.mark.parametrize(
    "target,timeout,status", [(successful_child, 10, "partial"), (sleeping_child, 1, "timed_out")]
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
