"""Run separately: python -m model_anatomy.service.worker.

One globally leased worker, one fresh spawn process per job. Production requires Linux
RLIMIT_AS (4 GiB) and a parent-enforced 120 second wall clock deadline.
"""

import logging
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from uuid import uuid4

from model_anatomy.dependencies import require_dependencies
from model_anatomy.api.settings import Settings
from model_anatomy.src.errors import AnatomyError, ArtifactError, UnsupportedModelError

from .database import Job
from .errors import ServiceError
from .jobs import JobService, cache_key

logger = logging.getLogger("model_anatomy.worker")


def enforce_limits(settings: Settings) -> None:
    if sys.platform == "linux":
        import resource

        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (settings.job_memory_bytes, settings.job_memory_bytes))
        cpu = math.ceil(settings.job_timeout_seconds) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    elif settings.environment == "production":
        raise RuntimeError("Production worker requires Linux resource enforcement")


def guard_parent() -> None:
    if sys.platform != "linux":
        return
    import ctypes

    parent = multiprocessing.parent_process()
    if parent is None:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Unable to enforce parent-death termination")
    if os.getppid() != parent.pid:
        os._exit(1)


def execute_job(service: JobService, job_id: str, owner: str, client, reconstruct, parse_reference) -> None:
    """Testable child execution: resolve first; cache hit never enters reconstruction."""
    with service.db.read() as session:
        job = session.get(Job, job_id)
    if job.engine_version != service.settings.engine_version:
        raise ServiceError(503, "engine_version_mismatch", "Queued job requires a different engine version")
    reference = parse_reference(job.model_input)
    if job.revision:
        reference = replace(reference, revision=job.revision)
    revision = client.resolve(reference)
    service.resolved(job_id, owner, reference.model_id, revision)
    key = cache_key(
        reference.model_id,
        revision,
        job.engine_version,
        job.schema_version,
        reference.filename,
        reference.explicit_filename,
    )
    result = service.cached_result(key)
    if result is not None:
        service.finish(job_id, owner, result, key, cached=True)
        return
    ir = reconstruct(
        job.model_input,
        client=client,
        revision=revision,
        on_progress=lambda stage: service.stage(job_id, owner, stage),
    )
    service.stage(job_id, owner, "storing_metadata")
    service.finish(job_id, owner, ir.to_json().encode(), key, cached=False)


def child_main(settings: Settings, job_id: str, owner: str) -> None:
    service = JobService(settings)
    try:
        guard_parent()
        enforce_limits(settings)
        # Thread counts are set before numerical libraries load into this fresh process.
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = "1"
        import requests

        from model_anatomy.src.artifacts import MetadataClient, parse_reference
        from model_anatomy.src.native_provider import require_supported_provider

        require_supported_provider()

        def reconstruct(*args, **kwargs):
            from model_anatomy.src.pipeline import reconstruct as engine_reconstruct

            return engine_reconstruct(*args, **kwargs, allow_artifact_inspection=True)

        with requests.Session() as session:
            execute_job(service, job_id, owner, MetadataClient(session), reconstruct, parse_reference)
    except Exception as exc:  # noqa: BLE001 - record unexpected child failures without leaking secrets
        code, message, status, transient = classify_failure(exc)
        try:
            service.fail(job_id, owner, code, message, status=status, transient=transient)
        except ServiceError:
            pass  # A cancelled or recovered job cannot be written by its former child.
    finally:
        service.db.engine.dispose()


def classify_failure(exc: BaseException) -> tuple[str, str, str, bool]:
    if isinstance(exc, MemoryError):
        return "memory_limit", "Reconstruction exceeded the memory budget", "failed", False
    if isinstance(exc, ServiceError):
        return (
            exc.code,
            exc.message,
            "failed",
            exc.code in {"storage_unavailable", "artifact_unavailable", "storage_busy"},
        )
    # This runs only in the child; importing requests does not import model libraries.
    import requests

    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return "artifact_network", "Artifact provider temporarily unavailable", "failed", True
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return (
            "artifact_http",
            f"Artifact provider returned HTTP {status}",
            "failed",
            status in {429, 502, 503, 504},
        )
    from model_anatomy.src.native_provider import NativeResolutionError

    if isinstance(exc, NativeResolutionError) and exc.diagnostic.get("code") == "provider_version_mismatch":
        return "provider_version_mismatch", str(exc), "failed", False
    if isinstance(exc, UnsupportedModelError):
        return (
            "unsupported_model",
            "Model is outside the engine's supported native scope",
            "unsupported",
            False,
        )
    if isinstance(exc, (ArtifactError, AnatomyError)):
        return "invalid_artifacts", "Artifacts are unsupported, malformed, or inconsistent", "failed", False
    return (
        "reconstruction_failed",
        "Reconstruction failed; no architecture facts were fabricated",
        "failed",
        False,
    )


def kill_process(process) -> None:
    if process.is_alive():
        if sys.platform == "linux":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
        else:
            process.kill()


def stop_process(process) -> None:
    kill_process(process)
    process.join(timeout=5)


class Worker:
    def __init__(self, settings: Settings, *, service=None):
        if settings.environment == "production" and sys.platform != "linux":
            raise RuntimeError("Production worker requires Linux resource enforcement")
        self.settings = settings
        self.service = service or JobService(settings)
        self.owner = uuid4().hex
        self.stopping = False
        self.process = None

    def request_stop(self, *_):
        self.stopping = True

    def run_job(self, job_id: str) -> None:
        process = multiprocessing.get_context("spawn").Process(
            target=child_main,
            args=(self.settings, job_id, self.owner),
            daemon=True,
        )
        self.process = process
        started = time.monotonic()
        process.start()
        deadline = threading.Event()

        def expire():
            deadline.set()
            kill_process(process)

        watchdog = threading.Timer(self.settings.job_timeout_seconds, expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            while process.is_alive():
                process.join(timeout=0.5)
                if not process.is_alive():
                    break
                try:
                    cancelled = self.service.heartbeat(self.owner, job_id)
                except ServiceError:
                    with self.service.db.read() as session:
                        current = session.get(Job, job_id)
                        if current is None or current.status != "running":
                            break
                    raise
                if cancelled:
                    stop_process(process)
                    self.service.fail(
                        job_id, self.owner, "cancelled", "Cancellation requested", status="cancelled"
                    )
                    return
                if self.stopping:
                    stop_process(process)
                    self.service.fail(
                        job_id, self.owner, "worker_interrupted", "Worker stopped", transient=True
                    )
                    return
                if time.monotonic() - started >= self.settings.job_timeout_seconds:
                    stop_process(process)
                    self.service.fail(
                        job_id,
                        self.owner,
                        "timeout",
                        "Reconstruction exceeded 120 second budget",
                        status="timed_out",
                    )
                    return
            process.join()
            with self.service.db.read() as session:
                job = session.get(Job, job_id)
                unfinished = bool(job and job.status == "running" and job.lease_owner == self.owner)
            if unfinished:
                self.service.fail(
                    job_id,
                    self.owner,
                    "timeout" if deadline.is_set() else "worker_crashed",
                    "Reconstruction exceeded wall clock budget"
                    if deadline.is_set()
                    else "Worker child exited without a result",
                    status="timed_out" if deadline.is_set() else "failed",
                )
        finally:
            watchdog.cancel()
            watchdog.join()
            stop_process(process)
            self.process = None
            process.close()

    def run(self, *, once=False) -> None:
        self.service.initialize()
        try:
            with self.service.db.worker_lock():
                self.run_locked(once=once)
        finally:
            self.service.db.engine.dispose()

    def run_locked(self, *, once=False) -> None:
        if not self.service.acquire_worker(self.owner):
            raise RuntimeError("Another worker holds the global execution lease")
        try:
            while not self.stopping:
                try:
                    self.service.heartbeat(self.owner)
                except ServiceError as exc:
                    # A suspended idle process still holds the lifetime execution lock.
                    # Reclaim its expired lease without ever overlapping a live child.
                    if exc.code != "lease_lost" or not self.service.acquire_worker(self.owner):
                        raise
                    logger.warning("idle worker lease recovered")
                self.service.cleanup()
                job_id = self.service.claim(self.owner)
                if job_id:
                    self.run_job(job_id)
                elif once:
                    break
                else:
                    time.sleep(1)
                if once:
                    break
        finally:
            if self.process:
                stop_process(self.process)
            self.service.release_worker(self.owner)


def main() -> None:
    require_dependencies("api", "engine", component="Worker")
    worker = Worker(Settings())
    signal.signal(signal.SIGINT, worker.request_stop)
    signal.signal(signal.SIGTERM, worker.request_stop)
    worker.run()


if __name__ == "__main__":
    main()
