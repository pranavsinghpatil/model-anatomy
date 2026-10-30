"""Persist public-safe terminal inspection outcomes."""

import sqlalchemy as sa
from alembic import op

revision = "0004_inspection_failures"
down_revision = "0003_benchmark_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inspection_failures",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("revision", sa.String(40), nullable=True),
        sa.Column("revision_source", sa.String(16), nullable=True),
        sa.Column("artifact", sa.String(512), nullable=False),
        sa.Column("explicit_artifact", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("engine_version", sa.String(128), nullable=False),
        sa.Column("started_at", sa.Float(), nullable=False),
        sa.Column("finished_at", sa.Float(), nullable=False),
    )
    for column in ("model_id", "revision", "status", "finished_at"):
        op.create_index(f"ix_inspection_failures_{column}", "inspection_failures", [column])
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE inspection_failures ENABLE ROW LEVEL SECURITY")
        op.execute(sa.text("""
            DO $$ DECLARE role_name text;
            BEGIN
              EXECUTE 'REVOKE ALL ON TABLE inspection_failures FROM PUBLIC';
              FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated'] LOOP
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
                  EXECUTE format('REVOKE ALL ON TABLE inspection_failures FROM %I', role_name);
                END IF;
              END LOOP;
            END $$
        """))


def downgrade() -> None:
    op.drop_table("inspection_failures")
