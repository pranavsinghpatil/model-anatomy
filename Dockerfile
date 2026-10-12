# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.9.6 AS uv

FROM python:3.11-slim-bookworm AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" HOME=/tmp \
    HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp/cache
WORKDIR /app

FROM python-base AS python-build
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY backend ./backend

FROM python-build AS api-build
RUN uv sync --frozen --extra api --no-dev --no-default-groups --no-editable \
    && .venv/bin/python -c "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('pytest') is None; assert importlib.util.find_spec('ruff') is None"

FROM python-build AS worker-build
RUN uv sync --frozen --extra api --extra engine --no-dev --no-default-groups --no-editable \
    && .venv/bin/python -c "import torch; import importlib.metadata as m; assert torch.version.cuda is None; assert not any(d.metadata['Name'].lower().startswith('nvidia-') for d in m.distributions())"

FROM python-base AS app-runtime
RUN groupadd --gid 10001 anatomy \
    && useradd --uid 10001 --gid 10001 --no-create-home anatomy \
    && mkdir -p /data && chown anatomy:anatomy /data
COPY backend ./backend
COPY infra ./infra
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
ENV MODEL_ANATOMY_DATA_DIR=/data
USER 10001:10001

FROM app-runtime AS api
COPY --from=api-build /app/.venv /app/.venv
EXPOSE 8000
CMD ["uvicorn", "model_anatomy.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-proxy-headers", "--limit-concurrency", "32"]

FROM app-runtime AS worker
COPY --from=worker-build /app/.venv /app/.venv
ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
CMD ["python", "-m", "model_anatomy.service.worker"]

FROM node:22-alpine AS frontend-build
WORKDIR /build
COPY frontend ./
RUN npm ci --no-audit --no-fund && npm run build

FROM caddy:2.10-alpine AS frontend
RUN addgroup -g 10001 anatomy && adduser -D -H -u 10001 -G anatomy anatomy \
    && mkdir -p /data /config /srv \
    && chown -R anatomy:anatomy /data /config /srv
COPY infra/Caddyfile /etc/caddy/Caddyfile
COPY --from=frontend-build /build/dist /srv
USER 10001:10001
EXPOSE 8080 8443
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
