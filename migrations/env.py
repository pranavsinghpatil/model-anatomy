"""Use the same DATABASE_URL and settings as API and worker; never log credentials."""

from alembic import context
from sqlalchemy import create_engine, pool

from model_anatomy.api.settings import Settings
from model_anatomy.service.database import Base

settings = Settings()
target_metadata = Base.metadata


if context.is_offline_mode():
    context.configure(
        url=settings.resolved_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    if settings.resolved_database_url.startswith("sqlite:"):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.resolved_database_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
