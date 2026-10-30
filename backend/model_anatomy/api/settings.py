"""Explicit local service configuration."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from model_anatomy.runtime_paths import project_root
from model_anatomy.src.version import ENGINE_VERSION


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODEL_ANATOMY_",
        env_file=project_root() / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        hide_input_in_errors=True,
    )

    port: int = Field(default=8001, ge=1024, le=65535)
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost", "[::1]"], min_length=1
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    docs_enabled: bool = True
    environment: Literal["local", "production"] = "local"
    data_dir: Path = Path(".local/model-anatomy")
    database_url: str | None = Field(
        default=None, validation_alias=AliasChoices("DATABASE_URL", "MODEL_ANATOMY_DATABASE_URL"), repr=False
    )
    storage_backend: Literal["local", "supabase"] = "local"
    demo_control_file: Path | None = None
    demo_session_id: str | None = None
    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MODEL_ANATOMY_SUPABASE_KEY", "MODEL_ANATOMY_SUPABASE_SERVICE_ROLE_KEY"
        ),
    )
    supabase_bucket: str = "model-anatomy-private"
    engine_version: str = ENGINE_VERSION
    schema_version: Literal["1.1"] = "1.1"
    job_ttl_seconds: int = Field(default=86400, ge=60, le=86400)
    cache_ttl_seconds: int = Field(default=604800, ge=60, le=604800)
    job_timeout_seconds: int = Field(default=120, ge=1, le=120)
    job_memory_bytes: int = Field(default=4 * 1024**3, ge=64 * 1024**2, le=4 * 1024**3)
    lease_seconds: int = Field(default=30, ge=5, le=300)
    max_attempts: int = Field(default=2, ge=1, le=3)
    queue_capacity: int = Field(default=10, ge=1, le=1000)
    guest_jobs_per_hour: int = Field(default=2, ge=1)
    global_jobs_per_day: int = Field(default=20, ge=1)
    guest_questions_per_hour: int = Field(default=10, ge=1)
    global_questions_per_day: int = Field(default=100, ge=1)
    max_blob_bytes: int = Field(default=32 * 1024**2, ge=1024)
    nebius_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("NEBIUS_API_KEY", "MODEL_ANATOMY_NEBIUS_API_KEY")
    )
    nebius_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "NEBIUS_MODEL", "MODEL_ANATOMY_NEBIUS_MODEL", "MODEL_ANATOMY_NEBIUS_MODEL_ID"
        ),
    )
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nebius_timeout_seconds: int = Field(default=30, ge=1, le=60)
    ai_context_bytes: int = Field(default=24000, ge=2048, le=48000)

    @field_validator("data_dir")
    @classmethod
    def project_data_directory(cls, value: Path) -> Path:
        return value if value.is_absolute() else project_root() / value

    @model_validator(mode="after")
    def production_is_explicit(self) -> "Settings":
        if self.demo_control_file is not None or self.demo_session_id is not None:
            if not self.demo_control_file or not self.demo_session_id:
                raise ValueError("Demo control file and session identity must be configured together")
            if self.environment != "local" or self.storage_backend != "local" or not self.resolved_database_url.startswith("sqlite:"):
                raise ValueError("Visitor demo controls require the local SQLite profile")
            if not self.demo_control_file.is_absolute():
                raise ValueError("Demo control file must have an absolute path")
        if "*" in self.cors_origins:
            raise ValueError("Credentialed CORS requires explicit origins")
        if self.environment == "production":
            if not self.database_url or not self.database_url.startswith(
                ("postgresql://", "postgresql+psycopg://")
            ):
                raise ValueError("Production requires a PostgreSQL DATABASE_URL")
            if self.storage_backend != "supabase":
                raise ValueError("Production requires private Supabase Storage")
            if not self.supabase_url or not self.supabase_url.startswith("https://"):
                raise ValueError("Production requires HTTPS Supabase URL")
            if not self.supabase_service_role_key:
                raise ValueError("Production requires Supabase service-role credentials")
            if "*" in self.allowed_hosts or any(origin.startswith("http://") for origin in self.cors_origins):
                raise ValueError("Production requires explicit hosts and HTTPS origins")
        if self.storage_backend == "supabase" and (
            not self.supabase_url or not self.supabase_service_role_key
        ):
            raise ValueError("Supabase Storage credentials are required")
        return self

    @property
    def resolved_database_url(self) -> str:
        value = self.database_url or f"sqlite:///{(self.data_dir / 'jobs.sqlite3').resolve().as_posix()}"
        if value.startswith("sqlite:"):
            from sqlalchemy.engine import make_url

            url = make_url(value)
            if url.database not in {None, ":memory:"} and not Path(url.database).is_absolute():
                value = url.set(database=str(project_root() / url.database)).render_as_string(hide_password=False)
        if value.startswith("postgresql://"):
            value = value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value
