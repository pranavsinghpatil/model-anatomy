"""Portable durable jobs. SQLite writers serialize before reading mutable state."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import JSON, Boolean, Float, Integer, String, create_engine, event, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .errors import ServiceError


class Base(DeclarativeBase):
    pass


class Mutex(Base):
    __tablename__ = "service_mutex"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    worker_owner: Mapped[str | None] = mapped_column(String(64))
    worker_until: Mapped[float] = mapped_column(Float, default=0)


class Guest(Base):
    __tablename__ = "guests"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float)
    last_seen_at: Mapped[float] = mapped_column(Float, index=True)


class Tombstone(Base):
    __tablename__ = "analysis_tombstones"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    guest_id: Mapped[str] = mapped_column(String(64), index=True)
    expires_at: Mapped[float] = mapped_column(Float, index=True)


class RateBucket(Base):
    __tablename__ = "rate_buckets"
    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    window: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class Job(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    guest_id: Mapped[str] = mapped_column(String(64), index=True)
    model_input: Mapped[str] = mapped_column(String(512))
    model_id: Mapped[str] = mapped_column(String(256))
    revision: Mapped[str | None] = mapped_column(String(40))
    engine_version: Mapped[str] = mapped_column(String(128))
    schema_version: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    created_at: Mapped[float] = mapped_column(Float, index=True)
    expires_at: Mapped[float] = mapped_column(Float, index=True)
    available_at: Mapped[float] = mapped_column(Float)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_until: Mapped[float | None] = mapped_column(Float)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    blob_digest: Mapped[str | None] = mapped_column(String(64), index=True)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    refresh_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict | None] = mapped_column(JSON)


class CacheEntry(Base):
    __tablename__ = "analysis_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    blob_digest: Mapped[str] = mapped_column(String(64), index=True)
    expires_at: Mapped[float] = mapped_column(Float, index=True)


class ModelReport(Base):
    """Durable registry identity for a successfully inspected private report blob."""
    __tablename__ = "model_reports"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cache_identity: Mapped[str] = mapped_column(String(64), index=True)
    model_id: Mapped[str] = mapped_column(String(256), index=True)
    request_revision: Mapped[str] = mapped_column(String(256), index=True)
    revision: Mapped[str] = mapped_column(String(40), index=True)
    engine_version: Mapped[str] = mapped_column(String(128))
    schema_version: Mapped[str] = mapped_column(String(16))
    artifact: Mapped[str] = mapped_column(String(512))
    explicit_artifact: Mapped[bool] = mapped_column(Boolean)
    blob_digest: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[dict] = mapped_column(JSON)
    inspected_at: Mapped[float] = mapped_column(Float, index=True)


class Blob(Base):
    __tablename__ = "analysis_blobs"
    digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float)
    deleting: Mapped[bool] = mapped_column(Boolean, default=False)


class Database:
    def __init__(self, url: str):
        self.sqlite = url.startswith("sqlite:")
        if self.sqlite and ":memory:" not in url:
            path = url.removeprefix("sqlite:///")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            **({"connect_args": {"timeout": 30, "check_same_thread": False}} if self.sqlite else {}),
        )
        if self.sqlite:

            @event.listens_for(self.engine, "connect")
            def sqlite_settings(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=30000")
                connection.execute("PRAGMA journal_mode=WAL")

    def initialize(self, *, create: bool) -> None:
        if create and self.sqlite:
            Base.metadata.create_all(self.engine)
            # Preserve early local databases created before retention fields were introduced.
            inspector = inspect(self.engine)
            with self.engine.begin() as connection:
                if "refresh_requested" not in {c["name"] for c in inspector.get_columns("analysis_jobs")}:
                    connection.execute(
                        text("ALTER TABLE analysis_jobs ADD COLUMN refresh_requested BOOLEAN NOT NULL DEFAULT FALSE")
                    )
                if "last_seen_at" not in {c["name"] for c in inspector.get_columns("guests")}:
                    connection.execute(
                        text("ALTER TABLE guests ADD COLUMN last_seen_at FLOAT NOT NULL DEFAULT 0")
                    )
                    connection.execute(text("UPDATE guests SET last_seen_at = created_at"))
                if "deleting" not in {c["name"] for c in inspector.get_columns("analysis_blobs")}:
                    connection.execute(
                        text("ALTER TABLE analysis_blobs ADD COLUMN deleting BOOLEAN NOT NULL DEFAULT FALSE")
                    )
        with self.write() as session:
            if session.get(Mutex, 1) is None:
                session.add(Mutex(id=1, worker_until=0))

    @contextmanager
    def write(self) -> Iterator[Session]:
        with Session(self.engine, expire_on_commit=False) as session:
            if self.sqlite:
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            else:
                session.execute(select(Mutex).where(Mutex.id == 1).with_for_update())
            try:
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise

    @contextmanager
    def read(self) -> Iterator[Session]:
        with Session(self.engine, expire_on_commit=False) as session:
            yield session

    @contextmanager
    def worker_lock(self) -> Iterator[None]:
        """Hold a lifetime lock so a paused live worker cannot overlap lease recovery."""
        if not self.sqlite:
            with self.engine.connect() as connection:
                key = 0x4D4F44454C
                if not connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}):
                    raise ServiceError(503, "worker_busy", "Another worker holds the global execution lock")
                connection.commit()
                try:
                    yield
                finally:
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                    connection.commit()
            return
        import os

        database_path = self.engine.url.database
        if database_path in {None, ":memory:"}:
            raise ValueError("Worker requires a durable SQLite file")
        lock_path = Path(database_path).resolve().with_suffix(".worker.lock")
        with lock_path.open("a+b") as handle:
            if lock_path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ServiceError(
                    503, "worker_busy", "Another worker holds the global execution lock"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
