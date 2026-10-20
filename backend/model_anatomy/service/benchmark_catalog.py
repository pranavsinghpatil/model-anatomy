"""Private persistence and public-safe reads for measured benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime

from sqlalchemy import select

from model_anatomy.src.artifact_inspection import INSPECTION_MESSAGES

from .database import BenchmarkCase, BenchmarkPublication, BenchmarkRun, ModelReport
from .errors import ServiceError
from .jobs import request_selector

PUBLIC_ROW_FIELDS = {
    "model_id", "category", "revision", "status", "architecture", "engine_version",
    "started_at", "finished_at", "warning_count", "graph_count", "node_count", "tensor_count",
    "parameter_counts", "coverage", "wall_seconds", "service_contract_valid", "case_identity",
    "warnings", "reason_codes", "diagnostic_notes", "reference", "declared_task", "error",
    "previous_observations", "run_kind",
}
OUTCOMES = {
    "completed", "partial", "inspected", "http_access", "network", "artifact_error", "unsupported",
    "invalid_report", "failure", "memorylimit", "timeout",
}
COUNT_FIELDS = ("warning_count", "graph_count", "node_count", "tensor_count")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,200}$")
SAFE_NOTES = {
    "The recorded static coverage is incomplete.",
    "Static-analysis warnings were recorded. Run a pinned inspection for full diagnostics.",
    "Publisher custom-code equivalence is unverified; the installed native implementation was used.",
    "This attempt did not produce a validated report.",
    "Native architecture reconstruction is unavailable; verified artifacts were inspected.",
    "The model's native architecture is not supported by this inspection profile.",
    "The repository did not expose a supported model artifact.",
} | set(INSPECTION_MESSAGES.values())


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 40:
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _validate_case(row: object) -> dict:
    if not isinstance(row, dict) or not set(row).issubset(PUBLIC_ROW_FIELDS):
        raise ValueError("benchmark case contains unsupported public fields")
    model_id, revision, status = row.get("model_id"), row.get("revision"), row.get("status")
    if not isinstance(model_id, str) or not ID_RE.fullmatch(model_id) or len(model_id) > 256:
        raise ValueError("benchmark case has invalid repository ID")
    if not isinstance(revision, str) or not SHA_RE.fullmatch(revision):
        raise ValueError("benchmark case must pin a 40-character revision")
    if status not in OUTCOMES:
        raise ValueError("benchmark case has an unknown outcome")
    for key in ("category", "engine_version"):
        value = row.get(key)
        if not isinstance(value, str) or not TOKEN_RE.fullmatch(value):
            raise ValueError(f"benchmark case has invalid {key}")
    if not _valid_timestamp(row.get("started_at")) or not _valid_timestamp(row.get("finished_at")):
        raise ValueError("benchmark case timestamps are required")
    if not isinstance(row.get("case_identity"), str) or not HEX64_RE.fullmatch(row["case_identity"]):
        raise ValueError("benchmark case identity must be SHA-256")
    for key in COUNT_FIELDS:
        value = row.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"benchmark case has invalid {key}")
    if "wall_seconds" in row and (
        type(row["wall_seconds"]) not in (int, float) or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0
    ):
        raise ValueError("benchmark case has invalid wall_seconds")
    if "architecture" in row and row["architecture"] is not None and (
        not isinstance(row["architecture"], str) or not TOKEN_RE.fullmatch(row["architecture"])
    ):
        raise ValueError("benchmark case has invalid architecture")
    if "parameter_counts" in row:
        counts = row["parameter_counts"]
        if not isinstance(counts, dict) or not set(counts).issubset(
            {"checkpoint_tensor_elements", "logical", "unique"}
        ) or any(value is not None and (type(value) is not int or value < 0) for value in counts.values()):
            raise ValueError("benchmark case has invalid parameter counts")
    if "coverage" in row:
        coverage = row["coverage"]
        if not isinstance(coverage, dict) or not set(coverage).issubset({
            "semantic_coverage", "module_semantic_coverage", "checkpoint_parameter_coverage", "tensor_shape_coverage"
        }) or any(
            value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1)
            for value in coverage.values()
        ):
            raise ValueError("benchmark case has invalid coverage")
    for key in ("declared_task", "run_kind"):
        if key in row and row[key] is not None and (
            not isinstance(row[key], str) or not TOKEN_RE.fullmatch(row[key])
        ):
            raise ValueError(f"benchmark case has invalid {key}")
    if "reference" in row and row["reference"] is not None:
        if not isinstance(row["reference"], str):
            raise ValueError("benchmark case has invalid reference")
        model, revision_selector, _, _ = request_selector(row["reference"])
        if model != model_id or revision_selector != revision:
            raise ValueError("benchmark reference differs from its pinned case")
    for key in ("warnings", "diagnostic_notes", "reason_codes"):
        values = row.get(key, [])
        if not isinstance(values, list) or len(values) > 20 or any(not isinstance(v, str) for v in values):
            raise ValueError(f"benchmark case has invalid {key}")
        if key == "warnings" and values:
            raise ValueError("raw warning text is not public benchmark evidence")
        if key == "diagnostic_notes" and any(note not in SAFE_NOTES for note in values):
            raise ValueError("benchmark case contains an unapproved diagnostic summary")
        if key == "reason_codes" and any(not re.fullmatch(r"[a-z0-9_]{1,80}", value) for value in values):
            raise ValueError("benchmark case contains an invalid reason code")
    if "previous_observations" in row:
        history = row["previous_observations"]
        if not isinstance(history, list) or len(history) > 20:
            raise ValueError("benchmark case has invalid observation history")
        for item in history:
            if (
                not isinstance(item, dict)
                or set(item) != {"status", "engine_version", "finished_at", "case_identity"}
                or item["status"] not in OUTCOMES
                or not isinstance(item["engine_version"], str)
                or not _valid_timestamp(item["finished_at"])
                or not isinstance(item["case_identity"], str)
                or not HEX64_RE.fullmatch(item["case_identity"])
            ):
                raise ValueError("benchmark case has invalid observation history")
    if "error" in row:
        error = row["error"]
        expected = rf"{re.escape(status)} during (case|reconstruction|report_validation)"
        if not isinstance(error, dict) or set(error) != {"message"} or not isinstance(error["message"], str) or not re.fullmatch(expected, error["message"]):
            raise ValueError("benchmark case error must use a generic public message")
    # Return a new allowlisted mapping so input dictionaries cannot mutate publication data.
    return {key: row[key] for key in PUBLIC_ROW_FIELDS if key in row}


class BenchmarkCatalogService:
    def __init__(self, job_service, *, clock=time.time):
        self.jobs, self.db, self.clock = job_service, job_service.db, clock

    def publish(self, catalog: dict, runs: list[dict]) -> str:
        """Validate and atomically persist an immutable 250-case publication and its source runs."""
        if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
            raise ValueError("benchmark catalog schema_version must be 1")
        rows = catalog.get("results")
        if not isinstance(rows, list) or len(rows) != 250:
            raise ValueError("benchmark catalog requires exactly 250 measured cases")
        cases = [_validate_case(row) for row in rows]
        source_cases = catalog.get("case_definitions")
        if source_cases is not None:
            if not isinstance(source_cases, list) or len(source_cases) != 250:
                raise ValueError("original benchmark case definitions must contain 250 cases")
            for source_case, case in zip(source_cases, cases, strict=True):
                if (
                    not isinstance(source_case, dict)
                    or source_case.get("model_id") != case["model_id"]
                    or source_case.get("revision") != case["revision"]
                    or source_case.get("category") != case["category"]
                    or set(source_case) - {"model_id", "revision", "category", "reference", "declared_task", "notes"}
                ):
                    raise ValueError("original benchmark case definition differs from published evidence")
                if source_case.get("reference") is not None:
                    _validate_case({**case, "reference": source_case["reference"]})
                if source_case.get("declared_task") is not None and (
                    not isinstance(source_case["declared_task"], str) or not TOKEN_RE.fullmatch(source_case["declared_task"])
                ):
                    raise ValueError("original benchmark task metadata is invalid")
                if source_case.get("notes") is not None and (
                    not isinstance(source_case["notes"], str) or len(source_case["notes"]) > 500
                ):
                    raise ValueError("original benchmark notes are invalid")
        names = [row["model_id"].casefold() for row in cases]
        if len(set(names)) != 250:
            raise ValueError("benchmark catalog requires 250 distinct repositories")
        counts = dict(Counter(row["status"] for row in cases))
        if catalog.get("status_counts") != counts:
            raise ValueError("benchmark status counts do not match case outcomes")
        metadata_keys = {
            "schema_version", "generated_at", "engine_version", "manifest_sha256", "manifest_hash_encoding",
            "runner_sha256", "engine_source_sha256", "baseline_runner_sha256", "baseline_engine_source_sha256",
            "method", "status_counts", "baseline_cases",
            "recheck_attempts", "engine_versions", "runs", "case_definitions", "results",
        }
        if not set(catalog).issubset(metadata_keys) or not HEX64_RE.fullmatch(catalog.get("manifest_sha256", "")):
            raise ValueError("benchmark catalog has invalid metadata fields or manifest hash")
        if catalog.get("manifest_hash_encoding") != "canonical-json-v1" or catalog.get("method") != (
            "Pinned metadata and bounded static reconstruction; no inference quality or speed measurement."
        ):
            raise ValueError("benchmark catalog metadata does not match the public evidence contract")
        if "runner_sha256" in catalog and not HEX64_RE.fullmatch(catalog["runner_sha256"]):
            raise ValueError("benchmark runner hash must be SHA-256")
        if "engine_source_sha256" in catalog and not HEX64_RE.fullmatch(catalog["engine_source_sha256"]):
            raise ValueError("benchmark engine source hash must be SHA-256")
        for key in ("baseline_runner_sha256", "baseline_engine_source_sha256"):
            if key in catalog and not HEX64_RE.fullmatch(catalog[key]):
                raise ValueError(f"benchmark {key} must be SHA-256")
        if not _valid_timestamp(catalog.get("generated_at")) or not isinstance(catalog.get("engine_version"), str):
            raise ValueError("benchmark catalog timestamp or engine version is invalid")
        catalog_runs = catalog.get("runs")
        if not isinstance(catalog_runs, list) or not catalog_runs or len(catalog_runs) > 100:
            raise ValueError("benchmark run history is required")
        for index, run in enumerate(catalog_runs):
            allowed = {"kind", "cases", "generated_at", "engine_version", "runner_sha256", "engine_source_sha256", "status_counts"}
            if (
                not isinstance(run, dict) or not set(run).issubset(allowed)
                or run.get("kind") != ("baseline" if index == 0 else "recheck")
                or type(run.get("cases")) is not int or run["cases"] < 1
                or not _valid_timestamp(run.get("generated_at"))
                or not isinstance(run.get("engine_version"), str) or not TOKEN_RE.fullmatch(run["engine_version"])
                or not HEX64_RE.fullmatch(run.get("runner_sha256", ""))
                or not HEX64_RE.fullmatch(run.get("engine_source_sha256", ""))
                or not isinstance(run.get("status_counts"), dict)
                or any(key not in OUTCOMES or type(value) is not int or value < 0 for key, value in run["status_counts"].items())
                or sum(run["status_counts"].values()) != run["cases"]
            ):
                raise ValueError("benchmark run history contains invalid public metadata")
        if not isinstance(runs, list) or not runs or any(not isinstance(run, dict) for run in runs):
            raise ValueError("benchmark source run payloads are required")
        if len(runs) != len(catalog_runs):
            raise ValueError("source run payload count differs from catalog run history")
        for source, descriptor in zip(runs, catalog_runs, strict=True):
            environment = source.get("environment")
            if (
                not isinstance(environment, dict)
                or source.get("cases") != descriptor["cases"]
                or source.get("generated_at") != descriptor["generated_at"]
                or source.get("status_counts") != descriptor["status_counts"]
                or not isinstance(source.get("results"), list)
                or len(source["results"]) != descriptor["cases"]
                or environment.get("engine_version") != descriptor["engine_version"]
                or environment.get("runner_sha256") != descriptor["runner_sha256"]
                or environment.get("engine_source_sha256") != descriptor["engine_source_sha256"]
            ):
                raise ValueError("portable source run does not match its public run history")
        # Canonicalize before opening the write transaction; malformed/non-portable JSON fails without a lock.
        run_payloads = [(run, _content_id(run)) for run in runs]
        metadata = {key: value for key, value in catalog.items() if key not in {"results", "case_definitions"}}
        _canonical(metadata)
        publication_material = {
            "metadata": metadata, "run_ids": [item[1] for item in run_payloads],
            "results": cases, "case_definitions": source_cases or [],
        }
        publication_id = _content_id(publication_material)
        now = self.clock()
        with self.db.write() as session:
            for payload, run_id in run_payloads:
                if session.get(BenchmarkRun, run_id) is None:
                    session.add(BenchmarkRun(id=run_id, payload=payload, created_at=now))
            if session.get(BenchmarkPublication, publication_id) is None:
                session.add(BenchmarkPublication(
                    id=publication_id, metadata_json=metadata,
                    run_ids=[item[1] for item in run_payloads], created_at=now,
                ))
                session.flush()
                session.add_all([
                    BenchmarkCase(
                        publication_id=publication_id, model_id=row["model_id"], ordinal=index,
                        data={**row, **({"_source_case_definition": source_cases[index]} if source_cases else {})},
                    )
                    for index, row in enumerate(cases)
                ])
        return publication_id

    def get_catalog(self) -> dict:
        with self.db.read() as session:
            publication = session.scalar(
                select(BenchmarkPublication).order_by(BenchmarkPublication.created_at.desc(), BenchmarkPublication.id.desc()).limit(1)
            )
            if publication is None:
                raise ServiceError(404, "benchmark_catalog_not_found", "No benchmark evidence has been published")
            cases = session.scalars(
                select(BenchmarkCase).where(BenchmarkCase.publication_id == publication.id).order_by(BenchmarkCase.ordinal)
            ).all()
            result = dict(publication.metadata_json)
            rows = [
                {key: value for key, value in case.data.items() if key != "_source_case_definition"}
                for case in cases
            ]
            reports = session.scalars(select(ModelReport).where(
                ModelReport.schema_version == self.jobs.settings.schema_version,
                ModelReport.model_id.in_([row["model_id"] for row in rows]),
            ).order_by(ModelReport.inspected_at.desc(), ModelReport.id.desc())).all()
            by_model = {}
            for report in reports:
                by_model.setdefault(report.model_id, []).append(report)
            for row in rows:
                row["stored_report_id"] = None
                try:
                    if row.get("reference"):
                        _, _, artifact, explicit = request_selector(row["reference"])
                    else:
                        artifact, explicit = "model.safetensors", False
                    compatible_reports = [report for report in by_model.get(row["model_id"], []) if
                        report.revision == row["revision"] and report.artifact == artifact and
                        report.explicit_artifact is explicit]
                    compatible = next((
                        report for report in compatible_reports
                        if report.engine_version == self.jobs.settings.engine_version
                    ), None)
                    if compatible is None and compatible_reports:
                        compatible = compatible_reports[0]
                    if compatible:
                        row["stored_report_id"] = compatible.id
                except ServiceError:
                    # Invalid selectors should have been rejected on publish; never use them to match a report.
                    pass
            result["results"] = rows
            return result
