"""A delayed Sandbox stop must never admit new work into its old session."""

import json

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import Job, RateBucket
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService


def service(tmp_path, session_id="session-one"):
    jobs = JobService(Settings(
        data_dir=tmp_path, database_url=f"sqlite:///{tmp_path / 'jobs.sqlite3'}",
        demo_control_file=tmp_path / "stop.json", demo_session_id=session_id,
        guest_jobs_per_hour=20,
    ))
    jobs.initialize()
    jobs.acquire_worker("worker")
    return jobs


def test_stop_fence_rejects_new_job_without_spending_quota_and_new_session_can_resume(tmp_path):
    jobs = service(tmp_path)
    guest_id, _ = jobs.guest(None)
    (tmp_path / "stop.json").write_text(json.dumps({"session_id": "session-one", "expires_at": 0}))
    with pytest.raises(ServiceError) as rejected:
        jobs.create(guest_id, "gpt2")
    assert rejected.value.code == "demo_stopping"
    with jobs.db.read() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 0
        assert session.scalar(select(func.count()).select_from(RateBucket)) == 0
    # The writer reservation may have expired; admission remains fenced until
    # the API is launched with another session identity after an actual resume.
    jobs.db.engine.dispose()
    resumed = service(tmp_path, "session-two")
    try:
        assert resumed.create(guest_id, "gpt2")["status"] == "queued"
    finally:
        resumed.db.engine.dispose()


@pytest.mark.parametrize("content", ["{invalid", "[]", '"invalid"', "{}", '{"session_id":null}'])
def test_unreadable_demo_fence_fails_closed(tmp_path, content):
    jobs = service(tmp_path)
    try:
        guest_id, _ = jobs.guest(None)
        (tmp_path / "stop.json").write_text(content)
        with pytest.raises(ServiceError) as rejected:
            jobs.create(guest_id, "gpt2")
        assert rejected.value.code == "demo_control_unavailable"
    finally:
        jobs.db.engine.dispose()


def test_admission_guard_applies_before_a_saved_report_can_create_a_job(tmp_path, monkeypatch):
    from types import SimpleNamespace

    jobs = service(tmp_path)
    try:
        guest_id, _ = jobs.guest(None)
        (tmp_path / "stop.json").write_text(json.dumps({"session_id": "session-one"}))
        report = SimpleNamespace(model_id="openai-community/gpt2")
        monkeypatch.setattr(jobs, "_registry_candidates_by_id", lambda _: [])
        monkeypatch.setattr(jobs, "_valid_registry", lambda _: (report, {}))
        with pytest.raises(ServiceError) as rejected:
            jobs.inspect_registry(guest_id, "saved")
        assert rejected.value.code == "demo_stopping"
        with pytest.raises(ServiceError) as rejected:
            jobs.create(guest_id, "gpt2")
        assert rejected.value.code == "demo_stopping"
    finally:
        jobs.db.engine.dispose()


def test_demo_settings_require_a_complete_local_profile(tmp_path):
    for fields in [
        {"demo_control_file": tmp_path / "stop.json"},
        {"demo_session_id": "session-one"},
        {"demo_session_id": "session-one", "demo_control_file": "relative.json"},
    ]:
        with pytest.raises(ValidationError):
            Settings(**fields)
