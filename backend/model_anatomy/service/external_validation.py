"""Immutable public-safe import of an external model validation CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from collections import Counter
from datetime import datetime

from sqlalchemy import select

from .database import ExternalValidationPublication

ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,200}$")
SOURCE_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REQUIRED_COLUMNS = {
    "input_row", "model_id", "status", "graph_generated", "reason_code", "architecture", "mode",
    "revision", "engine_version", "graph_count", "node_count", "started_at",
}
SAFE_REASON_CODES = {
    "", "safetensors_unavailable", "http_401", "architecture_incomplete", "ArtifactError",
    "native_architecture_unavailable", "architecture_config_unavailable",
}
SAFE_STATUSES = {"reported_complete", "reported_partial", "out_of_scope", "reported_failure"}
MAX_CSV_BYTES = 2_000_000
MAX_ROWS = 5_000


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _integer(value: str, name: str, *, minimum: int = 0) -> int | None:
    if value == "":
        return None
    if not value.isdecimal():
        raise ValueError(f"invalid {name}")
    result = int(value)
    if result < minimum or result > 2**53 - 1:
        raise ValueError(f"invalid {name}")
    return result


def _timestamp(value: str) -> str | None:
    if not value:
        return None
    if len(value) > 40:
        raise ValueError("invalid started_at")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid started_at") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("started_at must include a timezone")
    return value


def _bool(value: str, name: str) -> bool:
    if value not in {"True", "False", "true", "false", "1", "0"}:
        raise ValueError(f"invalid {name}")
    return value.lower() in {"true", "1"}


def _normalize(row: dict[str, str]) -> tuple[tuple, dict]:
    model_id = row.get("model_id", "")
    if not ID_RE.fullmatch(model_id) or len(model_id) > 256:
        raise ValueError("invalid repository ID")
    revision = row.get("revision", "") or None
    if revision is not None and not SHA_RE.fullmatch(revision):
        raise ValueError("revision must be a pinned 40-character commit")
    input_row = _integer(row.get("input_row", ""), "input_row", minimum=1)
    if input_row is None:
        raise ValueError("input_row is required")
    source_status = row.get("status", "")
    reason_code = row.get("reason_code", "")
    if source_status not in {"complete", "partial", "fail"} or reason_code not in SAFE_REASON_CODES:
        raise ValueError("unknown source status or reason code")
    generated = _bool(row.get("graph_generated", ""), "graph_generated")
    if source_status in {"complete", "partial"}:
        if not generated or row.get("mode") != "static_module_architecture":
            raise ValueError("successful observations require generated static module architecture")
        if _integer(row.get("graph_count", ""), "graph_count") in (None, 0) or _integer(
            row.get("node_count", ""), "node_count"
        ) in (None, 0):
            raise ValueError("successful observations require nonempty graph counts")
        if source_status == "complete" and reason_code:
            raise ValueError("complete observations cannot carry a reason code")
        if source_status == "partial" and reason_code != "architecture_incomplete":
            raise ValueError("partial observations require the architecture_incomplete reason code")
    elif generated:
        raise ValueError("failed observations cannot claim a generated graph")
    if not revision and reason_code != "http_401":
        raise ValueError("only access-restricted observations may omit a pinned revision")
    if revision is None and source_status != "fail":
        raise ValueError("unresolved revision cannot be reported as successful")
    report_mode = row.get("mode", "") or None
    if report_mode is not None and (not TOKEN_RE.fullmatch(report_mode) or len(report_mode) > 80):
        raise ValueError("invalid report mode")
    architecture = row.get("architecture", "") or None
    if architecture is not None and (not TOKEN_RE.fullmatch(architecture) or len(architecture) > 200):
        architecture = None  # Class-like strings are optional display metadata, never identity.
    engine_version = row.get("engine_version", "") or None
    if engine_version is not None and (not TOKEN_RE.fullmatch(engine_version) or len(engine_version) > 128):
        raise ValueError("invalid engine version")
    graph_count = _integer(row.get("graph_count", ""), "graph_count")
    node_count = _integer(row.get("node_count", ""), "node_count")
    started_at = _timestamp(row.get("started_at", ""))
    if source_status == "complete":
        status, scope_reason = "reported_complete", None
    elif source_status == "partial":
        status, scope_reason = "reported_partial", None
    elif reason_code == "safetensors_unavailable":
        status, scope_reason = "out_of_scope", "no_safetensors"
    elif reason_code == "http_401":
        status, scope_reason = "out_of_scope", "access_restricted"
    else:
        status, scope_reason = "reported_failure", None
    observation = {
        "model_id": model_id,
        "revision": revision,
        "artifact": "model.safetensors",
        "explicit_artifact": False,
        "status": status,
        "source_status": source_status,
        "source_reason_code": reason_code or None,
        "scope_reason": scope_reason,
        "report_mode": report_mode,
        "architecture": architecture,
        "engine_version": engine_version,
        "graph_count": graph_count,
        "node_count": node_count,
        "evidence_basis": "reported",
        "source_rows": [input_row],
        "source_started_at": started_at,
    }
    identity = (model_id.casefold(), revision, "model.safetensors", False)
    return identity, observation


def prepare_publication(
    csv_bytes: bytes, *, source_repository: str, source_commit: str, source_path: str,
) -> tuple[str, dict, list[dict]]:
    """Validate and sanitize a bounded source CSV without retaining diagnostic prose."""
    if not isinstance(csv_bytes, bytes) or len(csv_bytes) > MAX_CSV_BYTES:
        raise ValueError("CSV exceeds the import size limit")
    if not SOURCE_RE.fullmatch(source_repository) or len(source_repository) > 512:
        raise ValueError("source repository must be a canonical public GitHub URL")
    if not SHA_RE.fullmatch(source_commit):
        raise ValueError("source commit must be a full 40-character Git commit")
    if not source_path or len(source_path) > 512 or source_path.startswith("/") or ".." in source_path.split("/"):
        raise ValueError("source path must be a repository-relative path")
    if "\\" in source_path or any(ord(char) < 32 or ord(char) == 127 for char in source_path):
        raise ValueError("source path contains unsupported characters")
    try:
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("CSV is missing required columns")
        if len(reader.fieldnames) > 100 or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV has invalid column headers")
        grouped: dict[tuple, dict[bytes, dict]] = {}
        raw_statuses: Counter[str] = Counter()
        raw_reason_codes: Counter[str] = Counter()
        observation_reason_codes: Counter[str] = Counter()
        repository_ids: set[str] = set()
        raw_rows = 0
        input_rows: set[int] = set()
        for raw in reader:
            raw_rows += 1
            if raw_rows > MAX_ROWS:
                raise ValueError("CSV exceeds the row limit")
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("CSV row does not match its header")
            identity, observation = _normalize(raw)
            row_no = observation["source_rows"][0]
            if row_no in input_rows:
                raise ValueError("input_row values must be unique")
            input_rows.add(row_no)
            raw_statuses[observation["source_status"]] += 1
            raw_reason_codes[observation["source_reason_code"] or "none"] += 1
            repository_ids.add(identity[0])
            row_material = {key: value for key, value in observation.items() if key != "source_rows"}
            row_key = _canonical(row_material)
            bucket = grouped.setdefault(identity, {})
            if row_key in bucket:
                bucket[row_key]["source_rows"].append(row_no)
            else:
                bucket[row_key] = observation
        for bucket in grouped.values():
            for observation in bucket.values():
                observation_reason_codes[observation["source_reason_code"] or "none"] += 1
        if not raw_rows:
            raise ValueError("CSV contains no observations")
        rows = [observation for bucket in grouped.values() for observation in bucket.values()]
        rows.sort(key=lambda item: (item["model_id"].casefold(), item["revision"] or "", item["source_rows"][0]))
        normalized_counts = Counter(row["status"] for row in rows)
        metadata = {
            "source_counts": {
                "input_rows": raw_rows,
                "unique_repository_ids": len(repository_ids),
                "unique_observations": len(rows),
                "duplicate_input_rows": raw_rows - len(rows),
                "raw_statuses": dict(sorted(raw_statuses.items())),
                "observation_statuses": dict(sorted(normalized_counts.items())),
                "raw_reason_codes": dict(sorted(raw_reason_codes.items())),
                "observation_reason_codes": dict(sorted(observation_reason_codes.items())),
            },
        }
        if sum(metadata["source_counts"]["raw_statuses"].values()) != raw_rows:
            raise ValueError("source status counts are inconsistent")
        csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
        material = {
            "source_repository": source_repository,
            "source_commit": source_commit,
            "source_path": source_path,
            "csv_sha256": csv_sha256,
            "metadata": metadata,
            "rows": rows,
        }
        publication_id = hashlib.sha256(_canonical(material)).hexdigest()
        return publication_id, {**material, "created_at": None}, rows
    except (UnicodeDecodeError, csv.Error, OverflowError) as exc:
        raise ValueError("CSV is malformed or contains unsupported values") from exc


class ExternalValidationService:
    def __init__(self, database, *, clock=time.time):
        self.db, self.clock = database, clock

    def publish(self, csv_bytes: bytes, *, source_repository: str, source_commit: str, source_path: str) -> str:
        publication_id, prepared, rows = prepare_publication(
            csv_bytes, source_repository=source_repository, source_commit=source_commit, source_path=source_path,
        )
        with self.db.write() as session:
            if session.get(ExternalValidationPublication, publication_id) is None:
                session.add(ExternalValidationPublication(
                    id=publication_id,
                    source_repository=prepared["source_repository"],
                    source_commit=prepared["source_commit"],
                    source_path=prepared["source_path"],
                    csv_sha256=prepared["csv_sha256"],
                    metadata_json=prepared["metadata"],
                    rows=rows,
                    created_at=self.clock(),
                ))
        return publication_id

    def latest(self) -> list[ExternalValidationPublication]:
        with self.db.read() as session:
            publications = session.scalars(select(ExternalValidationPublication).order_by(
                ExternalValidationPublication.created_at.desc(), ExternalValidationPublication.id.desc()
            )).all()
        latest, seen = [], set()
        for publication in publications:
            key = (publication.source_repository, publication.source_path)
            if key not in seen:
                seen.add(key)
                latest.append(publication)
        return latest
