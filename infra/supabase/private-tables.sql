-- Run after Alembic upgrade head, as the project's postgres/database owner.
-- Application tables are accessed only by the trusted API and worker over SQL.
-- Browser visitors authenticate to FastAPI with cookies, not Supabase Auth.
-- Do not FORCE RLS: the database owner must retain backend access.

ALTER TABLE public.alembic_version ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.service_mutex ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.guests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_tombstones ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rate_buckets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_blobs ENABLE ROW LEVEL SECURITY;

-- No anon/authenticated policies: browser keys must not bypass guest ownership.
REVOKE ALL PRIVILEGES ON TABLE
    public.alembic_version,
    public.service_mutex,
    public.guests,
    public.analysis_tombstones,
    public.rate_buckets,
    public.analysis_jobs,
    public.analysis_cache,
    public.analysis_blobs
FROM PUBLIC, anon, authenticated;

REVOKE ALL PRIVILEGES ON SEQUENCE public.service_mutex_id_seq
FROM PUBLIC, anon, authenticated;

-- Remove schema-level defaults for future objects created by this owner.
-- Global default grants are additive: inspect pg_default_acl separately, and
-- rerun table hardening after migrations. This is not automatic RLS for new tables.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL PRIVILEGES ON TABLES FROM PUBLIC, anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
    REVOKE ALL PRIVILEGES ON SEQUENCES FROM PUBLIC, anon, authenticated;
