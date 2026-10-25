"""Durable inspected model registry backed by private report blobs."""

import sqlalchemy as sa
from alembic import op

revision = "0002_model_reports"
down_revision = "0001_durable_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("refresh_requested", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.create_table(
        "model_reports",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("cache_identity", sa.String(64), nullable=False),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("request_revision", sa.String(256), nullable=False),
        sa.Column("revision", sa.String(40), nullable=False),
        sa.Column("engine_version", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(16), nullable=False),
        sa.Column("artifact", sa.String(512), nullable=False),
        sa.Column("explicit_artifact", sa.Boolean(), nullable=False),
        sa.Column("blob_digest", sa.String(64), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("inspected_at", sa.Float(), nullable=False),
    )
    for column in ("cache_identity", "model_id", "request_revision", "revision", "blob_digest", "inspected_at"):
        op.create_index(f"ix_model_reports_{column}", "model_reports", [column])
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE model_reports ENABLE ROW LEVEL SECURITY")
        op.execute(sa.text("""
            DO $$ DECLARE role_name text;
            BEGIN
              FOREACH role_name IN ARRAY ARRAY['PUBLIC', 'anon', 'authenticated'] LOOP
                IF role_name = 'PUBLIC' THEN
                  EXECUTE 'REVOKE ALL ON TABLE model_reports FROM PUBLIC';
                ELSIF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
                  EXECUTE format('REVOKE ALL ON TABLE model_reports FROM %I', role_name);
                END IF;
              END LOOP;
            END $$
        """))


def downgrade() -> None:
    op.drop_table("model_reports")
    op.drop_column("analysis_jobs", "refresh_requested")
