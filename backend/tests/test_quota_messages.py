import pytest

from model_anatomy.api.settings import Settings
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.jobs import JobService


@pytest.mark.parametrize("scope", ["guest", "global"])
def test_quota_error_identifies_scope_allowance_and_reset(tmp_path, scope):
    settings = Settings(
        data_dir=tmp_path,
        guest_jobs_per_hour=1 if scope == "guest" else 20,
        global_jobs_per_day=1 if scope == "global" else 20,
    )
    service = JobService(settings, clock=lambda: 3700)
    service.initialize()
    service.acquire_worker("worker")
    guest, _ = service.guest(None)
    service.create(guest, "org/model")
    with pytest.raises(ServiceError) as error:
        service.create(guest, "org/model")
    assert error.value.status == 429 and error.value.code == "rate_limited"
    label = "Hourly visitor" if scope == "guest" else "Daily shared"
    minutes = 59 if scope == "guest" else 1379
    assert error.value.message == f"{label} analysis allowance reached (1). Try again in {minutes} minutes."
    service.db.engine.dispose()
