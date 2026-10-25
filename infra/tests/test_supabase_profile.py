from pathlib import Path

import pytest
from scripts.supabase_profile import load_supabase_profile


def test_supabase_profile_requires_config_without_echoing_values(tmp_path: Path):
    secret = "service-role-very-secret"
    env_file = tmp_path / ".env"
    env_file.write_text(f"MODEL_ANATOMY_SUPABASE_KEY={secret}\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_supabase_profile(env_file)

    assert secret not in str(error.value)


def test_supabase_profile_loads_database_and_private_storage_settings(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_ANATOMY_DATABASE_URL=postgresql+psycopg://user:pass@db.example:5432/app?sslmode=require\n"
        "MODEL_ANATOMY_SUPABASE_URL=https://project.supabase.co\n"
        "MODEL_ANATOMY_SUPABASE_KEY=server-key\n"
        "MODEL_ANATOMY_SUPABASE_BUCKET=private-bucket\n",
        encoding="utf-8",
    )

    profile = load_supabase_profile(env_file)

    assert profile["DATABASE_URL"].startswith("postgresql+psycopg://")
    assert profile["MODEL_ANATOMY_DATABASE_URL"] == profile["DATABASE_URL"]
    assert profile["MODEL_ANATOMY_SUPABASE_URL"] == "https://project.supabase.co"
    assert profile["MODEL_ANATOMY_SUPABASE_BUCKET"] == "private-bucket"


def test_template_words_in_real_password_are_not_rejected(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_ANATOMY_DATABASE_URL=postgresql://user:project-password@db.example:5432/app?sslmode=require\n"
        "MODEL_ANATOMY_SUPABASE_URL=https://project.supabase.co\n"
        "MODEL_ANATOMY_SUPABASE_KEY=server-key\n",
        encoding="utf-8",
    )
    assert load_supabase_profile(env_file)["DATABASE_URL"].startswith("postgresql://")


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://user:pass@db.example:5432/app",
        "postgresql://user:pass@db.example:6543/app?sslmode=require",
        "sqlite:///jobs.db",
    ],
)
def test_supabase_profile_rejects_non_tls_or_non_5432_databases(tmp_path: Path, database_url: str):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"MODEL_ANATOMY_DATABASE_URL={database_url}\n"
        "MODEL_ANATOMY_SUPABASE_URL=https://project.supabase.co\n"
        "MODEL_ANATOMY_SUPABASE_KEY=server-key\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="TLS"):
        load_supabase_profile(env_file)


@pytest.mark.parametrize(
    "storage_url",
    [
        "https://secret-user:secret-password@project.supabase.co",
        "https://project.supabase.co?secret-query",
        "https://project.supabase.co#secret-fragment",
        "https://project.supabase.co:secret-port",
        "https://project.supabase.co:8443",
        "https://[secret-host",
    ],
)
def test_storage_origin_rejects_credential_bearing_urls_without_echo(tmp_path, storage_url):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_ANATOMY_DATABASE_URL=postgresql://u:p@db.example:5432/app?sslmode=require\n"
        f"MODEL_ANATOMY_SUPABASE_URL={storage_url}\n"
        "MODEL_ANATOMY_SUPABASE_KEY=secret-key\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as error:
        load_supabase_profile(env_file)
    assert "secret" not in str(error.value)
    assert str(error.value) == "Supabase profile storage URL must be an HTTPS project origin"


def test_malformed_database_port_does_not_echo_secrets(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_ANATOMY_DATABASE_URL=postgresql://u:secret-password@db.example:secret-port/app?sslmode=require\n"
        "MODEL_ANATOMY_SUPABASE_URL=https://project.supabase.co\n"
        "MODEL_ANATOMY_SUPABASE_KEY=secret-key\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as error:
        load_supabase_profile(env_file)
    assert "secret" not in str(error.value)
    assert "TLS" in str(error.value)
