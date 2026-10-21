"""Durable anonymous jobs, quotas, fenced leases, cache identity, and retention."""

import hashlib
import json
import math
import re
import secrets
import time
from datetime import UTC, datetime
from functools import lru_cache
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from sqlalchemy import delete, func, select, update

from .database import Blob, CacheEntry, Database, Guest, Job, ModelReport, Mutex, RateBucket, Tombstone
from .errors import ServiceError
from .storage import storage_for

TERMINAL = {"completed", "partial", "inspected", "unsupported", "failed", "timed_out", "cancelled"}


def timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def model_reference(value: str) -> tuple[str, str]:
    value = value.strip()
    if value == "gpt2":
        value = "openai-community/gpt2"
    model_id = value
    if "://" in value:
        url = urlsplit(value)
        parts = url.path.strip("/").split("/")
        if (
            url.scheme != "https"
            or url.netloc != "huggingface.co"
            or url.query
            or url.fragment
            or (len(parts) != 2 and len(parts) < 5 and not (
                len(parts) == 4 and parts[2] == "tree" and re.fullmatch(r"[0-9a-f]{40}", parts[3])
            ))
        ):
            raise ServiceError(422, "invalid_model", "Expected a Hugging Face repository ID or HTTPS URL")
        if len(parts) >= 5:
            revision = unquote(parts[3])
            file_parts = [unquote(part) for part in parts[4:]]
            filename = "/".join(file_parts)
            if (
                parts[2] not in {"blob", "resolve"}
                or revision in {"", ".", ".."}
                or any(not part or part in {".", ".."} or "/" in part or "\\" in part for part in file_parts)
                or not filename.endswith((".safetensors", ".safetensors.index.json", ".gguf"))
            ):
                raise ServiceError(422, "invalid_model", "Expected a Safetensors or GGUF artifact reference")
        model_id = "/".join(parts[:2])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id) or any(
        part in {".", ".."} for part in model_id.split("/")
    ):
        raise ServiceError(422, "invalid_model", "Expected a namespace/model repository ID")
    return value, model_id


def request_selector(value: str) -> tuple[str, str, str, bool]:
    """Return model, requested revision selector, artifact and explicit selection."""
    value, model_id = model_reference(value)
    revision, artifact, explicit = "main", "model.safetensors", False
    if "://" in value:
        parts = urlsplit(value).path.strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree":
            revision = parts[3]
        elif len(parts) >= 5:
            revision = unquote(parts[3])
            artifact = "/".join(unquote(part) for part in parts[4:])
            explicit = True
    return model_id, revision, artifact, explicit


def cache_key(
    model_id: str,
    revision: str,
    engine: str,
    schema: str,
    artifact: str = "model.safetensors",
    explicit: bool = False,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Cache identity requires an immutable Hugging Face commit")
    identity = [model_id, revision, engine, schema, artifact, explicit]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def ir_summary(data: dict) -> dict:
    metadata = data["metadata"]
    inspection = metadata.get("inspection")
    if inspection and inspection.get("kind") == "artifact_inspection":
        return {
            "architecture": None,
            "root_graph_id": None,
            "parameter_count": None,
            "dimensions": {},
            "input_names": [],
            "warnings": metadata.get("analysis_report", {}).get("warnings", []),
            "limitations": metadata.get("limitations", []),
            "quality": {
                "mode": "artifact_inspection",
                "forward_executed": False,
                "runtime_trace_available": False,
            },
            "partial_reasons": [],
            "engine_version": metadata.get("engine_version"),
            "result_status": "inspected",
            "inspection": {
                "kind": "artifact_inspection",
                "reason_code": inspection["reason_code"],
                "message": inspection["message"],
                "file_count": len(inspection.get("files", [])),
                "tensor_count": len(data.get("tensors", {})),
                "checkpoint_tensor_element_count": metadata.get("checkpoint_tensor_element_count"),
                "checkpoint_available": inspection.get("checkpoint_available", False),
            },
        }
    report = metadata.get("analysis_report", {})
    warnings = list(dict.fromkeys(report.get("warnings", [])))
    quality = {}
    reasons = []
    for key, label in (
        ("semantic_coverage", "Graph component classification"),
        ("module_semantic_coverage", "Native module classification"),
        ("checkpoint_parameter_coverage", "Checkpoint parameter mapping"),
        ("tensor_shape_coverage", "Graph connection shapes"),
    ):
        value = report.get(key)
        valid = type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)
        quality[key] = value if valid else None
        if not valid:
            reasons.append(f"{label}: coverage was not established.")
        elif value < 1:
            reasons.append(f"{label}: coverage is incomplete.")
    for key in ("unknown_nodes", "unmapped_checkpoint_tensors"):
        value = report.get(key)
        if key == "unmapped_checkpoint_tensors":
            value = len(value) if isinstance(value, list) else None
        quality[key] = value if type(value) is int and value >= 0 else None
        if quality[key] is None:
            label = "Unknown component count" if key == "unknown_nodes" else "Unmapped tensor count"
            reasons.append(f"{label}: check was not established.")
        elif quality[key]:
            label = "unclassified graph component" if key == "unknown_nodes" else "unmapped checkpoint tensor"
            if value != 1:
                label += "s"
            reasons.append(f"{value} {label}.")
    for key in ("forward_executed", "runtime_trace_available"):
        value = report.get(key)
        quality[key] = value if type(value) is bool else None
    quality["mode"] = report.get("mode") if isinstance(report.get("mode"), str) else None
    return {
        "architecture": data["architecture"],
        "root_graph_id": data["root_graph"]["id"],
        "parameter_count": metadata.get("unique_parameter_count"),
        "dimensions": metadata.get("dimensions", {}),
        "input_names": sorted(metadata.get("inputs", {}).get("input_specs", {})),
        "warnings": warnings,
        "limitations": list(dict.fromkeys(metadata.get("limitations", []))),
        "quality": quality,
        "partial_reasons": reasons,
        "engine_version": metadata.get("engine_version"),
        "result_status": "partial" if reasons or warnings else "completed",
    }


def job_detail(job: Job, summary: dict | None = None) -> dict:
    summary = summary if summary is not None else job.summary or {}
    return {
        "id": job.id,
        "model_id": job.model_id,
        "revision": job.revision,
        "architecture": summary.get("architecture"),
        "status": job.status,
        "stage": job.stage,
        "error": job.error,
        "warnings": summary.get("warnings", []),
        "limitations": summary.get("limitations", []),
        "partial_reasons": summary.get("partial_reasons", []),
        "quality": summary.get("quality", {}),
        "engine_version": summary.get("engine_version", job.engine_version),
        "created_at": timestamp(job.created_at),
        "expires_at": timestamp(job.expires_at),
        "parameter_count": summary.get("parameter_count"),
        "root_graph_id": summary.get("root_graph_id"),
        "dimensions": summary.get("dimensions", {}),
        "input_names": summary.get("input_names", []),
        "cached": job.cached,
        "inspection": summary.get("inspection"),
    }


class JobService:
    def __init__(self, settings, *, clock=time.time, storage=None):
        self.settings, self.clock = settings, clock
        self.db = Database(settings.resolved_database_url)
        self.storage = storage or storage_for(settings)
        self._legacy_summary = lru_cache(maxsize=64)(self._read_legacy_summary)

    def _read_legacy_summary(self, digest: str) -> dict:
        try:
            return ir_summary(json.loads(self.storage.get(digest)))
        except (ValueError, KeyError, TypeError) as exc:
            raise ServiceError(503, "invalid_artifact", "Stored report summary is unavailable") from exc

    def present(self, job: Job) -> dict:
        # Older persisted summaries mixed method limitations into extraction warnings.
        # Reproject immutable report facts; cache only the small summary, never the full IR.
        if (
            job.blob_digest
            and job.status in {"completed", "partial"}
            and not {"quality", "input_names"}.issubset(job.summary or {})
        ):
            summary = self._legacy_summary(job.blob_digest)
            detail = job_detail(job, summary)
            detail["status"] = detail["stage"] = summary["result_status"]
            return detail
        return job_detail(job)

    def initialize(self) -> None:
        self.db.initialize(create=self.settings.environment == "local")
        self.storage.check()

    def guest(self, token: str | None) -> tuple[str, str | None]:
        if token and re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            guest_id = hashlib.sha256(token.encode()).hexdigest()
            with self.db.write() as session:
                known = session.get(Guest, guest_id)
                if known:
                    known.last_seen_at = self.clock()
                    return guest_id, None
        new_token = secrets.token_urlsafe(32)
        guest_id = hashlib.sha256(new_token.encode()).hexdigest()
        with self.db.write() as session:
            session.add(Guest(id=guest_id, created_at=self.clock(), last_seen_at=self.clock()))
        return guest_id, new_token

    def _quota(self, session, guest_id: str, action: str) -> None:
        suffix = "jobs" if action == "jobs" else "questions"
        limits = [
            (f"global:{action}:day", getattr(self.settings, f"global_{suffix}_per_day"), 86400),
            (f"guest:{guest_id}:{action}:hour", getattr(self.settings, f"guest_{suffix}_per_hour"), 3600),
        ]
        for key, limit, seconds in limits:
            window = int(self.clock() // seconds)
            bucket = session.get(RateBucket, (key, window))
            if bucket and bucket.count >= limit:
                minutes = math.ceil(((window + 1) * seconds - self.clock()) / 60)
                scope = "Daily shared" if key.startswith("global:") else "Hourly visitor"
                action_label = "analysis" if action == "jobs" else "AI question"
                raise ServiceError(
                    429,
                    "rate_limited",
                    f"{scope} {action_label} allowance reached ({limit}). Try again in {max(1, minutes)} minutes.",
                )
            if bucket:
                bucket.count += 1
            else:
                session.add(RateBucket(key=key, window=window, count=1))

    def _quota_available(self, session, guest_id: str, action: str) -> None:
        """Reject already-exhausted visitors before any report blob is read."""
        suffix = "jobs" if action == "jobs" else "questions"
        limits = [
            (f"global:{action}:day", getattr(self.settings, f"global_{suffix}_per_day"), 86400),
            (f"guest:{guest_id}:{action}:hour", getattr(self.settings, f"guest_{suffix}_per_hour"), 3600),
        ]
        for key, limit, seconds in limits:
            window = int(self.clock() // seconds)
            bucket = session.get(RateBucket, (key, window))
            if bucket and bucket.count >= limit:
                minutes = math.ceil(((window + 1) * seconds - self.clock()) / 60)
                scope = "Daily shared" if key.startswith("global:") else "Hourly visitor"
                label = "analysis" if action == "jobs" else "AI question"
                raise ServiceError(
                    429, "rate_limited",
                    f"{scope} {label} allowance reached ({limit}). Try again in {max(1, minutes)} minutes.",
                )

    def question_quota(self, guest_id: str) -> None:
        with self.db.write() as session:
            self._quota(session, guest_id, "questions")

    def worker_available(self) -> bool:
        with self.db.read() as session:
            mutex = session.get(Mutex, 1)
            return bool(mutex and mutex.worker_owner and mutex.worker_until > self.clock())

    def create(self, guest_id: str, value: str, *, refresh: bool = False) -> dict:
        value, model_id = model_reference(value)
        now = self.clock()
        model_id, selector, artifact, explicit = request_selector(value)
        with self.db.read() as session:
            self._quota_available(session, guest_id, "jobs")
        valid = None if refresh else self._valid_registry(
            self._registry_candidates(model_id, selector, artifact, explicit)
        )
        with self.db.write() as session:
            # A saved report is usable without a live worker or queue capacity.
            if valid:
                report, summary = valid
                self._quota(session, guest_id, "jobs")
                job = self._ready_job(session, guest_id, value, model_id, report, summary, now)
                return job_detail(job)
            mutex = session.get(Mutex, 1)
            if not mutex.worker_owner or mutex.worker_until <= now:
                raise ServiceError(503, "worker_unavailable", "Reconstruction worker is offline")
            active = session.scalar(
                select(func.count()).select_from(Job).where(Job.status == "queued", Job.expires_at > now)
            )
            if active >= self.settings.queue_capacity:
                raise ServiceError(503, "queue_full", "Reconstruction queue is full")
            self._quota(session, guest_id, "jobs")
            job = Job(
                id=uuid4().hex,
                guest_id=guest_id,
                model_input=value,
                model_id=model_id,
                engine_version=self.settings.engine_version,
                schema_version=self.settings.schema_version,
                status="queued",
                stage="queued",
                created_at=now,
                expires_at=now + self.settings.job_ttl_seconds,
                available_at=now,
                attempts=0,
                cancel_requested=False,
                cached=False,
                refresh_requested=refresh,
                summary={},
            )
            session.add(job)
            session.flush()
            return job_detail(job)

    def _registry_candidates(self, model_id: str, selector: str, artifact: str, explicit: bool):
        with self.db.read() as session:
            query = select(ModelReport).where(
                ModelReport.model_id == model_id,
                ModelReport.artifact == artifact,
                ModelReport.explicit_artifact.is_(explicit),
                ModelReport.engine_version == self.settings.engine_version,
                ModelReport.schema_version == self.settings.schema_version,
            )
            if selector == "main":
                query = query.where(ModelReport.request_revision == "main").order_by(
                    ModelReport.inspected_at.desc(), ModelReport.id
                )
            else:
                query = query.where(
                    (ModelReport.request_revision == selector) | (ModelReport.revision == selector)
                ).order_by(ModelReport.inspected_at.desc(), ModelReport.id)
            return list(session.scalars(query.limit(20)))

    def _valid_registry(self, candidates):
        for report in candidates:
            try:
                raw = self.storage.get(report.blob_digest)
                document = json.loads(raw)
                metadata = document.get("metadata") if isinstance(document, dict) else None
                if (
                    not isinstance(metadata, dict)
                    or
                    document.get("schema_version") != self.settings.schema_version
                    or document.get("model_id") != report.model_id
                    or metadata.get("revision") != report.revision
                    or report.cache_identity != cache_key(
                        report.model_id, report.revision, report.engine_version,
                        report.schema_version, report.artifact, report.explicit_artifact,
                    )
                    or metadata.get("engine_version", report.engine_version) != report.engine_version
                ):
                    raise ValueError("registry and report identity disagree")
                return report, ir_summary(document)
            except ServiceError as exc:
                if exc.code not in {"artifact_unavailable", "artifact_corrupt"}:
                    raise
                with self.db.write() as session:
                    stale = session.get(ModelReport, report.id)
                    if stale:
                        session.delete(stale)
            except (ValueError, KeyError, TypeError, AttributeError):
                # A damaged index row is never served. Remove it and let a live worker
                # reconstruct the submitted request if one is available.
                with self.db.write() as session:
                    stale = session.get(ModelReport, report.id)
                    if stale:
                        session.delete(stale)
        return None

    def _ready_job(self, session, guest_id, value, model_id, report, summary, now):
        job = Job(
            id=uuid4().hex, guest_id=guest_id, model_input=value, model_id=model_id,
            revision=report.revision, engine_version=report.engine_version,
            schema_version=self.settings.schema_version, status=summary["result_status"],
            stage=summary["result_status"], created_at=now,
            expires_at=now + self.settings.job_ttl_seconds, available_at=now, attempts=0,
            cancel_requested=False, blob_digest=report.blob_digest, cached=True, summary=summary,
        )
        session.add(job)
        session.flush()
        return job

    def inspect_registry(self, guest_id: str, report_id: str) -> dict:
        with self.db.read() as session:
            self._quota_available(session, guest_id, "jobs")
        candidates = self._registry_candidates_by_id(report_id)
        valid = self._valid_registry(candidates)
        if not valid:
            raise ServiceError(404, "not_found", "Catalog report not found")
        report, summary = valid
        now = self.clock()
        with self.db.write() as session:
            self._quota(session, guest_id, "jobs")
            value = f"https://huggingface.co/{report.model_id}/resolve/{report.revision}/{report.artifact}"
            job = self._ready_job(session, guest_id, value, report.model_id, report, summary, now)
            return job_detail(job)

    def _registry_candidates_by_id(self, report_id: str):
        with self.db.read() as session:
            report = session.get(ModelReport, report_id)
            if not report or report.schema_version != self.settings.schema_version:
                return []
            return [report]

    def catalog(self, query: str = "", offset: int = 0, limit: int = 24) -> dict:
        with self.db.read() as session:
            base = select(ModelReport).where(
                ModelReport.schema_version == self.settings.schema_version,
            )
            if query:
                base = base.where(ModelReport.model_id.ilike(f"%{query}%"))
            total = session.scalar(select(func.count()).select_from(base.subquery())) or 0
            rows = session.scalars(
                base.order_by(ModelReport.inspected_at.desc(), ModelReport.id).offset(offset).limit(limit)
            ).all()
        models = []
        for row in rows:
            summary = row.summary
            dims = summary.get("dimensions", {})
            models.append({
                "id": row.id, "model_id": row.model_id, "revision": row.revision,
                "architecture": summary.get("architecture"), "status": summary.get("result_status"),
                "parameter_count": summary.get("parameter_count"),
                "dimensions": {key: dims.get(key) for key in ("hidden", "heads", "layers", "vocabulary")},
                "engine_version": row.engine_version, "schema_version": row.schema_version,
                "artifact": row.artifact, "explicit_artifact": row.explicit_artifact,
                "inspected_at": timestamp(row.inspected_at),
            })
        return {"models": models, "total": total, "offset": offset, "limit": limit}

    def get(self, job_id: str, guest_id: str) -> Job:
        with self.db.read() as session:
            job = session.get(Job, job_id)
            if job is None or job.guest_id != guest_id:
                expired = session.get(Tombstone, job_id)
                if expired and expired.guest_id == guest_id and expired.expires_at > self.clock():
                    raise ServiceError(410, "expired", "Analysis expired after 24 hours")
                raise ServiceError(404, "not_found", "Analysis not found")
            if job.expires_at <= self.clock():
                raise ServiceError(410, "expired", "Analysis expired after 24 hours")
            return job

    def recent(self, guest_id: str) -> list[dict]:
        with self.db.read() as session:
            jobs = session.scalars(
                select(Job)
                .where(Job.guest_id == guest_id, Job.expires_at > self.clock())
                .order_by(Job.created_at.desc())
                .limit(50)
            ).all()
            return [self.present(job) for job in jobs]

    def cancel(self, job_id: str, guest_id: str) -> dict:
        self.get(job_id, guest_id)
        with self.db.write() as session:
            job = session.get(Job, job_id)
            if job.status not in TERMINAL:
                job.cancel_requested = True
                if job.status == "queued":
                    job.status = job.stage = "cancelled"
                else:
                    job.stage = "cancelling"
        return self.present(job)

    def acquire_worker(self, owner: str) -> bool:
        with self.db.write() as session:
            mutex = session.get(Mutex, 1)
            now = self.clock()
            if mutex.worker_owner not in {None, owner} and mutex.worker_until > now:
                return False
            mutex.worker_owner, mutex.worker_until = owner, now + self.settings.lease_seconds
            return True

    def release_worker(self, owner: str) -> None:
        with self.db.write() as session:
            mutex = session.get(Mutex, 1)
            if mutex.worker_owner == owner:
                mutex.worker_owner, mutex.worker_until = None, 0

    def _owned(self, session, job_id: str, owner: str) -> Job:
        job = session.get(Job, job_id)
        mutex = session.get(Mutex, 1)
        now = self.clock()
        if (
            not job
            or job.status != "running"
            or job.lease_owner != owner
            or not job.lease_until
            or job.lease_until <= now
            or mutex.worker_owner != owner
            or mutex.worker_until <= now
        ):
            raise ServiceError(409, "lease_lost", "Worker lease is no longer valid")
        return job

    def heartbeat(self, owner: str, job_id: str | None = None) -> bool:
        with self.db.write() as session:
            mutex = session.get(Mutex, 1)
            now = self.clock()
            if mutex.worker_owner != owner or mutex.worker_until <= now:
                raise ServiceError(409, "lease_lost", "Worker lease is no longer valid")
            mutex.worker_until = now + self.settings.lease_seconds
            if job_id:
                job = self._owned(session, job_id, owner)
                job.lease_until = now + self.settings.lease_seconds
                return job.cancel_requested or job.expires_at <= now
            return False

    def claim(self, owner: str) -> str | None:
        with self.db.write() as session:
            now = self.clock()
            mutex = session.get(Mutex, 1)
            if mutex.worker_owner != owner or mutex.worker_until <= now:
                raise ServiceError(409, "lease_lost", "Worker lease is no longer valid")
            for job in session.scalars(select(Job).where(Job.status.in_(["queued", "running"]))):
                if job.expires_at <= now or job.cancel_requested:
                    job.status = job.stage = "cancelled"
                elif job.status == "running":
                    if job.lease_until and job.lease_until > now:
                        return None
                    if job.attempts >= self.settings.max_attempts:
                        job.status = job.stage = "failed"
                        job.error = {
                            "code": "worker_interrupted",
                            "message": "Worker recovery attempts exhausted",
                        }
                    else:
                        job.status = job.stage = "queued"
                        job.available_at = now
            session.flush()
            query = (
                select(Job)
                .where(Job.status == "queued", Job.available_at <= now, Job.expires_at > now)
                .order_by(Job.created_at)
                .limit(1)
            )
            if not self.db.sqlite:
                query = query.with_for_update(skip_locked=True)
            candidate = session.scalar(query)
            if candidate is None:
                return None
            result = session.execute(
                update(Job)
                .where(Job.id == candidate.id, Job.status == "queued")
                .values(
                    status="running",
                    stage="resolving_revision",
                    lease_owner=owner,
                    lease_until=now + self.settings.lease_seconds,
                    attempts=Job.attempts + 1,
                )
            )
            return candidate.id if result.rowcount == 1 else None

    def stage(self, job_id: str, owner: str, stage: str) -> None:
        with self.db.write() as session:
            job = self._owned(session, job_id, owner)
            if job.cancel_requested:
                raise ServiceError(409, "cancelled", "Cancellation requested")
            job.stage = re.sub(r"[^a-zA-Z0-9_-]", "_", str(stage))[:64]

    def resolved(self, job_id: str, owner: str, model_id: str, revision: str) -> Job:
        cache_key(model_id, revision, self.settings.engine_version, self.settings.schema_version)
        with self.db.write() as session:
            job = self._owned(session, job_id, owner)
            if job.revision is not None and job.revision != revision:
                raise ServiceError(409, "revision_changed", "Retry resolved a different immutable revision")
            job.model_id, job.revision, job.stage = model_id, revision, "checking_cache"
            return job

    def cached_result(
        self, key: str, model_id: str | None = None, revision: str | None = None,
        artifact: str = "model.safetensors", explicit: bool = False,
    ) -> bytes | None:
        with self.db.read() as session:
            entry = session.get(CacheEntry, key)
            if not entry or entry.expires_at <= self.clock():
                return None
            digest = entry.blob_digest
        try:
            data = self.storage.get(digest)
            document = json.loads(data)
            metadata = document.get("metadata") if isinstance(document, dict) else None
            if not isinstance(metadata, dict) or document.get("schema_version") != self.settings.schema_version or (
                model_id is not None and document.get("model_id") != model_id
            ) or (revision is not None and metadata.get("revision") != revision):
                raise ValueError("cache identity mismatch")
            if model_id is not None and revision is not None and key != cache_key(
                model_id, revision, self.settings.engine_version, self.settings.schema_version, artifact, explicit
            ):
                raise ValueError("cache key identity mismatch")
            return data
        except (ServiceError, ValueError, KeyError, TypeError, AttributeError) as exc:
            if isinstance(exc, ServiceError) and exc.code not in {"artifact_unavailable", "artifact_corrupt", "invalid_artifact"}:
                raise
            with self.db.write() as session:
                stale = session.get(CacheEntry, key)
                if stale and stale.blob_digest == digest:
                    session.delete(stale)
            return None

    def finish(self, job_id: str, owner: str, data: bytes, key: str, *, cached: bool) -> None:
        document = json.loads(data)
        if document.get("schema_version") != self.settings.schema_version:
            raise ServiceError(503, "schema_mismatch", "Worker returned an unsupported IR schema")
        digest = hashlib.sha256(data).hexdigest()
        # Reserve durably before network I/O, so interrupted uploads remain collectible.
        with self.db.write() as session:
            job = self._owned(session, job_id, owner)
            if job.cancel_requested or job.expires_at <= self.clock():
                job.status = job.stage = "cancelled"
                return
            if document["metadata"]["revision"] != job.revision or document["model_id"] != job.model_id:
                raise ServiceError(503, "identity_mismatch", "IR identity disagrees with resolved revision")
            blob = session.get(Blob, digest)
            if blob and blob.deleting:
                raise ServiceError(503, "storage_busy", "Metadata object is being retired; retry later")
            if blob is None:
                session.add(Blob(digest=digest, created_at=self.clock(), deleting=False))
        self.storage.put(data)
        with self.db.write() as session:
            job = self._owned(session, job_id, owner)
            if job.cancel_requested or job.expires_at <= self.clock():
                job.status = job.stage = "cancelled"
                return
            entry = session.get(CacheEntry, key)
            if entry is None:
                session.add(
                    CacheEntry(
                        key=key, blob_digest=digest, expires_at=self.clock() + self.settings.cache_ttl_seconds
                    )
                )
            elif not cached:
                entry.blob_digest, entry.expires_at = digest, self.clock() + self.settings.cache_ttl_seconds
            job.blob_digest, job.cached, job.summary = digest, cached, ir_summary(document)
            job.status = job.stage = job.summary["result_status"]
            job.error, job.lease_owner, job.lease_until = None, None, None
            requested_model, requested_revision, artifact, explicit = request_selector(job.model_input)
            if requested_model != job.model_id:
                raise ServiceError(503, "identity_mismatch", "Submitted reference disagrees with inspected model")
            registry_id = hashlib.sha256(f"{key}:{requested_revision}".encode()).hexdigest()
            existing_report = session.get(ModelReport, registry_id)
            values = {
                "cache_identity": key,
                "model_id": job.model_id,
                "request_revision": requested_revision,
                "revision": job.revision,
                "engine_version": job.engine_version,
                "schema_version": job.schema_version,
                "artifact": artifact,
                "explicit_artifact": explicit,
                "blob_digest": digest,
                "summary": job.summary,
                "inspected_at": self.clock(),
            }
            if existing_report is None:
                session.add(ModelReport(id=registry_id, **values))
            else:
                for field, value in values.items():
                    setattr(existing_report, field, value)

    def fail(
        self,
        job_id: str,
        owner: str,
        code: str,
        message: str,
        *,
        status: str = "failed",
        transient: bool = False,
    ) -> None:
        with self.db.write() as session:
            job = self._owned(session, job_id, owner)
            job.error = {"code": code, "message": message}
            if job.cancel_requested or job.expires_at <= self.clock():
                status, transient = "cancelled", False
            if transient and job.attempts < self.settings.max_attempts:
                job.status = job.stage = "queued"
                job.available_at = self.clock() + min(30, 2**job.attempts)
            else:
                job.status = job.stage = status
            job.lease_owner, job.lease_until = None, None

    def cleanup(self) -> None:
        retired = []
        with self.db.write() as session:
            now = self.clock()
            expired = session.scalars(select(Job).where(Job.expires_at <= now)).all()
            for job in expired:
                until = job.expires_at + self.settings.job_ttl_seconds
                if until > now and session.get(Tombstone, job.id) is None:
                    session.add(Tombstone(id=job.id, guest_id=job.guest_id, expires_at=until))
                session.delete(job)
            session.execute(delete(Tombstone).where(Tombstone.expires_at <= now))
            session.execute(delete(CacheEntry).where(CacheEntry.expires_at <= now))
            session.execute(
                delete(RateBucket).where(
                    RateBucket.key.like("global:%:day"), RateBucket.window < int(now // 86400)
                )
            )
            session.execute(
                delete(RateBucket).where(
                    RateBucket.key.like("guest:%:hour"), RateBucket.window < int(now // 3600)
                )
            )
            session.flush()
            session.execute(
                delete(Guest).where(
                    Guest.last_seen_at <= now - self.settings.job_ttl_seconds,
                    ~Guest.id.in_(select(Job.guest_id)),
                    ~Guest.id.in_(select(Tombstone.guest_id)),
                )
            )
            referenced = set(session.scalars(select(Job.blob_digest).where(Job.blob_digest.is_not(None))))
            referenced.update(session.scalars(select(CacheEntry.blob_digest)))
            referenced.update(session.scalars(select(ModelReport.blob_digest)))
            grace = self.settings.job_timeout_seconds + self.settings.lease_seconds
            for blob in session.scalars(select(Blob)):
                if blob.digest not in referenced and (blob.deleting or blob.created_at < now - grace):
                    blob.deleting = True
                    retired.append(blob.digest)
        # Deletions are retryable and never hold the API/heartbeat transaction mutex.
        for digest in retired:
            self.storage.delete(digest)
            with self.db.write() as session:
                blob = session.get(Blob, digest)
                if blob and blob.deleting:
                    session.delete(blob)
