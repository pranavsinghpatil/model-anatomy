"""Revision-aware reuse without importing or executing the extraction engine."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import CacheEntry
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService, cache_key
from model_anatomy.service.views import load_ir
from model_anatomy.service.worker import execute_job
from model_anatomy.src.artifacts import parse_reference


@pytest.fixture
def harness(tmp_path):
    now = [10000.0]
    service = JobService(Settings(data_dir=tmp_path, guest_jobs_per_hour=10), clock=lambda: now[0])
    service.initialize()
    service.acquire_worker("w")
    guest, _ = service.guest(None)
    original = (Path(__file__).resolve().parents[2] / "output/gpt2/model.json").read_bytes()
    revision = json.loads(original)["metadata"]["revision"]
    state = {"revision": revision, "resolved": [], "extractions": [], "offline": False}

    class Resolver:
        def resolve(self, reference):
            state["resolved"].append(reference)
            if state["offline"]:
                raise requests.Timeout()
            return state["revision"]

    def reconstruct(value, *, client, revision, on_progress):
        state["extractions"].append(revision)
        data = json.loads(original)
        data["metadata"]["revision"] = revision
        return load_ir(json.dumps(data).encode())

    def run():
        job_id = service.claim("w")
        assert job_id is not None
        execute_job(service, job_id, "w", Resolver(), reconstruct, parse_reference)
        return service.get(job_id, guest)

    yield service, guest, state, now, run
    service.db.engine.dispose()


def test_concurrent_submissions_share_one_extraction_and_new_commit_gets_new_report(harness, caplog):
    service, guest, state, _, run = harness
    caplog.set_level(logging.INFO, logger="model_anatomy.worker")
    with ThreadPoolExecutor(max_workers=3) as pool:
        submitted = list(pool.map(lambda _: service.create(guest, "gpt2")["id"], range(3)))
    jobs = [run() for _ in submitted]
    assert set(submitted) == {job.id for job in jobs}
    assert [job.cached for job in jobs] == [False, True, True]
    assert len({job.blob_digest for job in jobs}) == 1
    assert state["extractions"] == [state["revision"]]
    assert len(state["resolved"]) == 3

    previous = state["revision"]
    state["revision"] = "b" * 40
    service.create(guest, "gpt2")
    changed = run()
    assert changed.cached is False
    assert changed.revision == state["revision"]
    assert changed.blob_digest != jobs[0].blob_digest
    service.create(guest, "gpt2")
    assert run().cached is True
    assert state["extractions"] == [previous, state["revision"]]
    assert all(reference.revision == "main" for reference in state["resolved"])
    events = [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]
    assert [event["event"] for event in events].count("analysis_cache_miss") == 2
    assert [event["event"] for event in events].count("analysis_cache_hit") == 3


@pytest.mark.parametrize(
    "damage,code", [("missing", "artifact_unavailable"), ("corrupt", "artifact_corrupt")]
)
def test_broken_cache_fails_honestly_without_repeating_extraction(harness, damage, code):
    service, guest, state, _, run = harness
    service.create(guest, "gpt2")
    first = run()
    path = service.storage.path(first.blob_digest)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    service.create(guest, "gpt2")
    with pytest.raises(ServiceError) as exc:
        run()
    assert exc.value.code == code
    assert len(state["extractions"]) == 1
    key = cache_key(first.model_id, first.revision, service.settings.engine_version, "1.1")
    with service.db.read() as session:
        assert session.get(CacheEntry, key).blob_digest == first.blob_digest


def test_upstream_outage_cannot_be_mistaken_for_an_unchanged_revision(harness):
    service, guest, state, _, run = harness
    service.create(guest, "gpt2")
    run()
    state["offline"] = True
    service.create(guest, "gpt2")
    with pytest.raises(requests.Timeout):
        run()
    assert len(state["extractions"]) == 1
