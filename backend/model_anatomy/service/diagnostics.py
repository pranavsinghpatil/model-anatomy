"""Correlated JSON events without request bodies, credentials or exception text."""

import json
import logging
import sys
import traceback
from collections import deque
from datetime import UTC, datetime
from pathlib import Path


def configure_logging(level: str) -> None:
    logger = logging.getLogger("model_anatomy")
    if not any(getattr(handler, "anatomy_diagnostics", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.anatomy_diagnostics = True
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    analysis_id: str | None = None,
    request_id: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    engine_version: str | None = None,
    stage: str | None = None,
    status: str | int | None = None,
    code: str | None = None,
    method: str | None = None,
    duration_ms: float | None = None,
    cached: bool | None = None,
    architecture: str | None = None,
    warning_count: int | None = None,
    exception: BaseException | None = None,
    level: int = logging.INFO,
) -> None:
    if not logger.isEnabledFor(level):
        return
    document = {
        "time": datetime.now(UTC).isoformat(),
        "level": logging.getLevelName(level),
        "event": event,
        "analysis_id": analysis_id,
        "request_id": request_id,
        "model_id": model_id,
        "revision": revision,
        "engine_version": engine_version,
        "stage": stage,
        "status": status,
        "code": code,
        "method": method,
        "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
        "cached": cached,
        "architecture": architecture,
        "warning_count": warning_count,
    }
    if exception is not None:
        document["exception_type"] = type(exception).__name__
        # Frame locations help reproduction; messages, source lines and locals can contain secrets.
        document["frames"] = list(
            deque(
                (
                    {
                        "file": Path(frame.f_code.co_filename).name,
                        "function": frame.f_code.co_name,
                        "line": line,
                    }
                    for frame, line in traceback.walk_tb(exception.__traceback__)
                ),
                maxlen=12,
            )
        )
    logger.log(level, json.dumps({key: value for key, value in document.items() if value is not None}))
