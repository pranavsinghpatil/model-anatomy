from pathlib import Path

import pytest
from pydantic import ValidationError

from model_anatomy.api.settings import Settings


def test_local_launcher_pins_all_storage_and_browser_settings():
    launcher = (Path(__file__).resolve().parents[2] / "scripts" / "start-local.ps1").read_text()

    assert "MODEL_ANATOMY_ENVIRONMENT = 'local'" in launcher
    assert "MODEL_ANATOMY_DATABASE_URL = $databaseUrl" in launcher
    assert "DATABASE_URL = $databaseUrl" in launcher
    assert "MODEL_ANATOMY_STORAGE_BACKEND = 'local'" in launcher
    assert 'MODEL_ANATOMY_ALLOWED_HOSTS = \'["127.0.0.1","localhost","[::1]"]\'' in launcher
    assert (
        'MODEL_ANATOMY_CORS_ORIGINS = \'["http://localhost:5173","http://127.0.0.1:5173","http://localhost:5174","http://127.0.0.1:5174"]\''
        in launcher
    )
    assert "AddSeconds(30)" in launcher
    assert "-PassThru" in launcher
    assert "torch, transformers, safetensors" in launcher
    assert "2> $dependencyErrorLog" in launcher
    assert "Get-Command node.exe" in launcher
    assert "-FilePath $node.Source" in launcher
    assert "'--strictPort'" in launcher
    assert "Get-Command npm.cmd" not in launcher
    assert "$env:ComSpec" not in launcher
    assert "-UseBasicParsing" in launcher
    assert "print(int(j.worker_available()))" in launcher
    assert "2>> $preflightErrorLog" in launcher
    assert "existing workers are never reused" in launcher
    assert launcher.index("existing workers are never reused") < launcher.index(
        "$processes.Api = Start-Process"
    )


def test_public_quota_defaults_remain_unchanged():
    settings = Settings(_env_file=None)

    assert settings.guest_jobs_per_hour == 2
    assert settings.global_jobs_per_day == 20
    assert settings.guest_questions_per_hour == 10
    assert settings.global_questions_per_day == 100
    assert settings.cache_ttl_seconds == 0


def test_cache_retention_is_an_explicit_operator_choice():
    assert Settings(cache_ttl_seconds=30 * 86400).cache_ttl_seconds == 30 * 86400
    with pytest.raises(ValidationError):
        Settings(cache_ttl_seconds=-1)


def test_production_keeps_public_quota_defaults():
    settings = Settings(
        _env_file=None,
        environment="production",
        database_url="postgresql://user:password@localhost/model_anatomy",
        storage_backend="supabase",
        supabase_url="https://storage.example",
        supabase_service_role_key="test-only-key",
        allowed_hosts=["app.example"],
        cors_origins=["https://app.example"],
    )

    assert settings.guest_jobs_per_hour == 2
    assert settings.global_jobs_per_day == 20
    assert settings.guest_questions_per_hour == 10
    assert settings.global_questions_per_day == 100


def test_bounded_local_quota_overrides_are_supported():
    settings = Settings(
        _env_file=None,
        guest_jobs_per_hour=100,
        global_jobs_per_day=500,
        guest_questions_per_hour=100,
        global_questions_per_day=500,
    )

    assert settings.environment == "local"
    assert settings.guest_jobs_per_hour == 100
    assert settings.global_jobs_per_day == 500
    assert settings.guest_questions_per_hour == 100
    assert settings.global_questions_per_day == 500
