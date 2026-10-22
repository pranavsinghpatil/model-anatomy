"""Persist immutable sanitized external validation publications."""

import sqlalchemy as sa
from alembic import op

revision = "0005_external_validations"
down_revision = "0004_inspection_failures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "external_validation_publications",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("source_repository", sa.String(512), nullable=False),
        sa.Column("source_commit", sa.String(40), nullable=False),
        sa.Column("source_path", sa.String(512), nullable=False),
        sa.Column("csv_sha256", sa.String(64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("rows", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "ix_external_validation_publications_source_repository",
        "external_validation_publications", ["source_repository"],
    )
    op.create_index(
        "ix_external_validation_publications_created_at",
        "external_validation_publications", ["created_at"],
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TABLE external_validation_publications ENABLE ROW LEVEL SECURITY")
        op.execute(sa.text("""
            DO $$ DECLARE role_name text;
            BEGIN
              EXECUTE 'REVOKE ALL ON TABLE external_validation_publications FROM PUBLIC';
              FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated'] LOOP
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
                  EXECUTE format('REVOKE ALL ON TABLE external_validation_publications FROM %I', role_name);
                END IF;
              END LOOP;
            END $$
        """))


def downgrade() -> None:
    op.drop_table("external_validation_publications")
