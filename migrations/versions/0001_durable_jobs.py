"""Initial durable anonymous service schema."""

import sqlalchemy as sa
from alembic import op

revision = "0001_durable_jobs"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "service_mutex",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("worker_owner", sa.String(64)),
        sa.Column("worker_until", sa.Float(), nullable=False),
    )
    op.execute(sa.text("INSERT INTO service_mutex (id, worker_until) VALUES (1, 0)"))
    op.create_table(
        "guests",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("last_seen_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_guests_last_seen_at", "guests", ["last_seen_at"])
    op.create_table(
        "analysis_tombstones",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("guest_id", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_analysis_tombstones_guest_id", "analysis_tombstones", ["guest_id"])
    op.create_index("ix_analysis_tombstones_expires_at", "analysis_tombstones", ["expires_at"])
    op.create_table(
        "rate_buckets",
        sa.Column("key", sa.String(160), primary_key=True),
        sa.Column("window", sa.Integer(), primary_key=True),
        sa.Column("count", sa.Integer(), nullable=False),
    )
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("guest_id", sa.String(64), nullable=False),
        sa.Column("model_input", sa.String(512), nullable=False),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("revision", sa.String(40)),
        sa.Column("engine_version", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("available_at", sa.Float(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.String(64)),
        sa.Column("lease_until", sa.Float()),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("blob_digest", sa.String(64)),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON()),
    )
    for column in ("guest_id", "status", "created_at", "expires_at", "blob_digest"):
        op.create_index(f"ix_analysis_jobs_{column}", "analysis_jobs", [column])
    op.create_table(
        "analysis_cache",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("blob_digest", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
    )
    for column in ("blob_digest", "expires_at"):
        op.create_index(f"ix_analysis_cache_{column}", "analysis_cache", [column])
    op.create_table(
        "analysis_blobs",
        sa.Column("digest", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("deleting", sa.Boolean(), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "analysis_tombstones",
        "analysis_blobs",
        "analysis_cache",
        "analysis_jobs",
        "rate_buckets",
        "guests",
        "service_mutex",
    ):
        op.drop_table(table)
