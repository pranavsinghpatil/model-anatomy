"""Persist immutable public-safe benchmark publications and source runs."""

import sqlalchemy as sa
from alembic import op

revision = "0003_benchmark_catalog"
down_revision = "0002_model_reports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])
    op.create_table(
        "benchmark_publications",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("run_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_benchmark_publications_created_at", "benchmark_publications", ["created_at"])
    op.create_table(
        "benchmark_cases",
        sa.Column(
            "publication_id", sa.String(64),
            sa.ForeignKey("benchmark_publications.id", ondelete="CASCADE"), primary_key=True,
        ),
        sa.Column("model_id", sa.String(256), primary_key=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
    )
    op.create_index("ix_benchmark_cases_publication_ordinal", "benchmark_cases", ["publication_id", "ordinal"])
    if op.get_bind().dialect.name == "postgresql":
        for table in ("benchmark_runs", "benchmark_publications", "benchmark_cases"):
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(sa.text("""
            DO $$ DECLARE role_name text; table_name text;
            BEGIN
              FOREACH table_name IN ARRAY ARRAY['benchmark_runs', 'benchmark_publications', 'benchmark_cases'] LOOP
                EXECUTE format('REVOKE ALL ON TABLE %I FROM PUBLIC', table_name);
                FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated'] LOOP
                  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
                    EXECUTE format('REVOKE ALL ON TABLE %I FROM %I', table_name, role_name);
                  END IF;
                END LOOP;
              END LOOP;
            END $$
        """))


def downgrade() -> None:
    op.drop_table("benchmark_cases")
    op.drop_table("benchmark_publications")
    op.drop_table("benchmark_runs")
