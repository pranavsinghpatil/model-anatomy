"""Public read model joining measured benchmark cases and real stored inspections."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from urllib.parse import quote

from sqlalchemy import select

from .database import BenchmarkCase, BenchmarkPublication, InspectionFailure, ModelReport
from .errors import ServiceError
from .external_validation import ExternalValidationService
from .jobs import request_selector, timestamp

TASK_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,200}$")


def _safe_task(value) -> str | None:
    return value if isinstance(value, str) and TASK_TOKEN.fullmatch(value) else None


def _epoch(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return 0.0


def _key(model_id: str, revision: str | None, artifact: str, explicit: bool):
    return model_id, revision, artifact, explicit


def _selection_id(artifact: str, explicit: bool) -> str:
    identity = json.dumps([artifact, explicit], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _pinned_reference(model_id: str, revision: str | None, artifact: str, explicit: bool) -> str | None:
    if revision is None:
        return None
    if explicit:
        filename = "/".join(quote(part, safe="") for part in artifact.split("/"))
        return f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    return f"https://huggingface.co/{model_id}/tree/{revision}"


def _diagnostics(status: str, summary: dict | None = None) -> list[str]:
    if status == "inspected":
        return ["Native architecture reconstruction is unavailable; verified artifacts were inspected."]
    if status in {"partial", "completed"}:
        coverage = (summary or {}).get("quality", {})
        return ["The recorded static coverage is incomplete."] if status == "partial" and any(
            value is None or value < 1 for key, value in coverage.items()
            if key in {"semantic_coverage", "module_semantic_coverage", "checkpoint_parameter_coverage", "tensor_shape_coverage"}
        ) else []
    if status in {"network", "artifact_error", "http_access", "unsupported", "invalid_report", "failure", "memorylimit", "timeout"}:
        return ["This attempt did not produce a validated report."]
    return []


class DiscoveryCatalogService:
    def __init__(self, jobs, *, clock=None):
        self.jobs, self.db = jobs, jobs.db
        self.clock = clock

    def get_catalog(self) -> dict:
        with self.db.read() as session:
            publication = session.scalar(
                select(BenchmarkPublication).order_by(
                    BenchmarkPublication.created_at.desc(), BenchmarkPublication.id.desc()
                ).limit(1)
            )
            if publication is None:
                benchmark_cases = []
                publication_id = None
            else:
                publication_id = publication.id
                benchmark_cases = session.scalars(
                    select(BenchmarkCase).where(
                        BenchmarkCase.publication_id == publication.id
                    ).order_by(BenchmarkCase.ordinal)
                ).all()
            reports = session.scalars(select(ModelReport).where(
                ModelReport.schema_version == self.jobs.settings.schema_version
            ).order_by(ModelReport.inspected_at.desc(), ModelReport.id)).all()
            failures = session.scalars(select(InspectionFailure).order_by(
                InspectionFailure.finished_at.desc(), InspectionFailure.id
            )).all()

        candidates: dict[tuple, list[dict]] = {}
        for case in benchmark_cases:
            row = {key: value for key, value in case.data.items() if key != "_source_case_definition"}
            try:
                if row.get("reference"):
                    model, revision, artifact, explicit = request_selector(row["reference"])
                    if model != row["model_id"]:
                        continue
                else:
                    revision, artifact, explicit = row.get("revision"), "model.safetensors", False
            except ServiceError:
                continue
            status = row["status"]
            entry = {
                "model_id": row["model_id"], "category": row.get("category") or "unclassified",
                "category_source": "benchmark" if row.get("category") else "unclassified",
                "revision": revision, "reference": row.get("reference"), "stored_report_id": None,
                "status": status, "architecture": row.get("architecture"),
                "declared_task": _safe_task(row.get("declared_task")),
                "declared_task_source": "benchmark" if _safe_task(row.get("declared_task")) else None,
                "started_at": row.get("started_at"),
                "finished_at": row.get("finished_at"), "engine_version": row["engine_version"],
                "warning_count": row.get("warning_count"), "graph_count": row.get("graph_count"),
                "tensor_count": row.get("tensor_count"),
                "parameter_counts": {"unique": (row.get("parameter_counts") or {}).get("unique")},
                "coverage": row.get("coverage") or {}, "dimensions": None,
                "artifact": artifact, "explicit_artifact": explicit, "source": "benchmark",
                "artifact_verified": status in {"completed", "partial", "inspected"} if explicit else None,
                "diagnostic_notes": row.get("diagnostic_notes") or _diagnostics(status),
                "reason_codes": row.get("reason_codes") or [],
                "benchmark_observations": [{
                    "publication_id": publication_id,
                    "case_identity": row.get("case_identity"),
                    "run_kind": row.get("run_kind"), "status": status,
                    "engine_version": row["engine_version"], "started_at": row.get("started_at"),
                    "finished_at": row.get("finished_at"),
                    "reference": row.get("reference"), "artifact": artifact,
                    "explicit_artifact": explicit, "architecture": row.get("architecture"),
                    "warning_count": row.get("warning_count"), "graph_count": row.get("graph_count"),
                    "node_count": row.get("node_count"), "tensor_count": row.get("tensor_count"),
                    "parameter_counts": row.get("parameter_counts"), "coverage": row.get("coverage"),
                    "declared_task": _safe_task(row.get("declared_task")),
                    "reason_codes": row.get("reason_codes", []),
                    "diagnostic_notes": row.get("diagnostic_notes", []),
                    "previous_observations": row.get("previous_observations", []),
                }],
                "_sort_time": _epoch(row.get("finished_at")),
            }
            candidates.setdefault(_key(row["model_id"], revision, artifact, explicit), []).append(entry)

        # Reports already passed storage digest and IR identity checks on insertion. This read model
        # uses only their persisted summaries; optional backfill can enrich older summaries safely.
        reports_by_key: dict[tuple, list] = {}
        for report in reports:
            key = _key(report.model_id, report.revision, report.artifact, report.explicit_artifact)
            reports_by_key.setdefault(key, []).append(report)
        for key, group in reports_by_key.items():
            latest = group[0]
            summary = latest.summary or {}
            status = summary.get("result_status")
            if status not in {"completed", "partial", "inspected"}:
                continue
            task = _safe_task(summary.get("declared_task"))
            finished = timestamp(latest.inspected_at)
            observations = [
                {"status": (r.summary or {}).get("result_status", "partial"),
                 "engine_version": r.engine_version, "finished_at": timestamp(r.inspected_at),
                 "stored_report_id": r.id}
                for r in group[1:20]
            ]
            inspection = {
                "model_id": latest.model_id, "category": summary.get("category") or "unclassified",
                "category_source": "inspection" if summary.get("category") and summary.get("category") != "unclassified" else "unclassified",
                "revision": latest.revision,
                "reference": _pinned_reference(latest.model_id, latest.revision, latest.artifact, latest.explicit_artifact),
                "stored_report_id": latest.id, "status": status,
                "architecture": summary.get("architecture"), "declared_task": task,
                "declared_task_source": "inspection" if task else None,
                "started_at": None, "finished_at": finished, "engine_version": latest.engine_version,
                "warning_count": summary.get("warning_count"), "graph_count": summary.get("graph_count"),
                "tensor_count": summary.get("tensor_count"),
                "parameter_counts": {"unique": summary.get("parameter_count")},
                "coverage": {key: (summary.get("quality") or {}).get(key) for key in (
                    "semantic_coverage", "module_semantic_coverage", "checkpoint_parameter_coverage", "tensor_shape_coverage"
                )},
                "dimensions": {name: (summary.get("dimensions") or {}).get(name)
                               for name in ("hidden", "heads", "layers", "vocabulary")},
                "artifact": latest.artifact,
                "artifact_verified": (
                    status in {"completed", "partial"}
                    or bool((summary.get("inspection") or {}).get("checkpoint_available"))
                ) if latest.explicit_artifact else None,
                "explicit_artifact": latest.explicit_artifact, "source": "inspection",
                "diagnostic_notes": _diagnostics(status, summary), "reason_codes": [],
                "previous_observations": observations,
                "benchmark_observations": [], "_sort_time": latest.inspected_at,
            }
            candidates.setdefault(key, []).append(inspection)

        for failure in failures:
            key = _key(failure.model_id, failure.revision, failure.artifact, failure.explicit_artifact)
            finished = timestamp(failure.finished_at)
            failure_entry = {
                "model_id": failure.model_id, "category": "unclassified", "revision": failure.revision,
                "category_source": "unclassified",
                "revision_source": failure.revision_source,
                "reference": _pinned_reference(failure.model_id, failure.revision, failure.artifact, failure.explicit_artifact),
                "stored_report_id": None, "status": failure.status, "architecture": None,
                "declared_task": None, "declared_task_source": None,
                "started_at": timestamp(failure.started_at), "finished_at": finished,
                "engine_version": failure.engine_version, "warning_count": None, "graph_count": None,
                "tensor_count": None, "parameter_counts": {"unique": None}, "coverage": {},
                "dimensions": None, "artifact": failure.artifact,
                "artifact_verified": False,
                "explicit_artifact": failure.explicit_artifact, "source": "inspection",
                "diagnostic_notes": _diagnostics(failure.status), "reason_codes": [failure.status],
                "benchmark_observations": [],
                "previous_observations": [], "_sort_time": failure.finished_at,
            }
            candidates.setdefault(key, []).append(failure_entry)

        external_publications = ExternalValidationService(self.db).latest()
        external_by_key: dict[tuple, list[dict]] = {}
        external_metadata = []
        for publication in external_publications:
            provenance = {
                "source_repository": publication.source_repository,
                "source_commit": publication.source_commit,
                "source_path": publication.source_path,
                "csv_sha256": publication.csv_sha256,
            }
            external_metadata.append({
                **provenance,
                "created_at": timestamp(publication.created_at),
                "source_counts": publication.metadata_json.get("source_counts", {}),
            })
            for observation in publication.rows:
                key = _key(
                    observation["model_id"], observation["revision"],
                    observation["artifact"], observation["explicit_artifact"],
                )
                external_by_key.setdefault(key, []).append({
                    **observation, **provenance, "_publication_created_at": publication.created_at,
                })

        for key, observations in external_by_key.items():
            if key in candidates:
                continue
            # When one repo/revision has conflicting source rows, show the newest dated
            # source observation while retaining every row in external_observations.
            latest = max(observations, key=lambda item: (
                _epoch(item.get("source_started_at")), item["_publication_created_at"],
            ))
            # Reported external outcomes are discoverable, but carry no verified report or
            # parameter/coverage claims. Graph counts are explicitly tagged as reported.
            candidates[key] = [{
                "model_id": latest["model_id"], "category": "unclassified",
                "category_source": "unclassified", "revision": latest["revision"],
                "reference": _pinned_reference(
                    latest["model_id"], latest["revision"], "model.safetensors", False,
                ),
                "stored_report_id": None, "status": latest["status"],
                "scope_reason": latest["scope_reason"], "source_status": latest["source_status"],
                "source_reason_code": latest["source_reason_code"],
                "report_mode": latest["report_mode"], "evidence_basis": "reported",
                "source_repository": latest["source_repository"], "source_commit": latest["source_commit"],
                "source_path": latest["source_path"], "csv_sha256": latest["csv_sha256"],
                "source_rows": latest["source_rows"], "source_started_at": latest["source_started_at"],
                "architecture": latest["architecture"], "declared_task": None,
                "declared_task_source": None, "started_at": latest["source_started_at"],
                "finished_at": None, "engine_version": latest["engine_version"],
                "warning_count": None, "graph_count": latest["graph_count"],
                "node_count": latest["node_count"], "tensor_count": None,
                "parameter_counts": {"unique": None}, "coverage": {}, "dimensions": None,
                "artifact": "model.safetensors", "explicit_artifact": False,
                "artifact_verified": None, "source": "external_validation",
                "diagnostic_notes": [], "reason_codes": [], "benchmark_observations": [],
                "previous_observations": [], "_sort_time": latest["_publication_created_at"],
            }]

        results = []
        for key, variants in candidates.items():
            variants.sort(key=lambda item: (
                item.get("source") != "external_validation", item["_sort_time"],
                item["source"] == "inspection", item.get("engine_version", ""),
            ), reverse=True)
            winner = dict(variants[0])
            if winner.get("category") == "unclassified":
                benchmark_category = next((variant.get("category") for variant in variants
                                           if variant.get("source") == "benchmark" and variant.get("category") != "unclassified"), None)
                if benchmark_category:
                    winner["category"] = benchmark_category
                    winner["category_source"] = "benchmark"
            if winner.get("declared_task") is None:
                benchmark_task = next((variant.get("declared_task") for variant in variants
                                       if variant.get("source") == "benchmark" and variant.get("declared_task")), None)
                if benchmark_task:
                    winner["declared_task"] = benchmark_task
                    winner["declared_task_source"] = "benchmark"
            benchmark_history = [observation for variant in variants for observation in variant.get("benchmark_observations", [])]
            if benchmark_history:
                winner["benchmark_observations"] = benchmark_history
            prior = [observation for variant in variants for observation in variant.get("previous_observations", [])]
            prior.extend({
                "status": variant["status"], "engine_version": variant.get("engine_version"),
                "finished_at": variant.get("finished_at"), "source": variant.get("source"),
                "stored_report_id": variant.get("stored_report_id"),
            } for variant in variants[1:])
            if prior:
                winner["previous_observations"] = prior[:20]
            winner["external_observations"] = [
                {name: value for name, value in observation.items() if name != "_publication_created_at"}
                for observation in external_by_key.get(key, [])
            ]
            if winner.get("source") == "inspection" and not winner.get("artifact_verified") and winner.get("explicit_artifact"):
                actual_artifact = winner["artifact"]
                winner["selection_id"] = _selection_id(actual_artifact, True)
                winner["artifact"] = None
                winner["reference"] = _pinned_reference(
                    winner["model_id"], winner.get("revision"), actual_artifact, False
                )
            winner.pop("_sort_time", None)
            results.append(winner)
        results.sort(key=lambda item: (_epoch(item.get("finished_at")), item["model_id"], item.get("revision") or ""), reverse=True)
        status_counts = dict(Counter(row["status"] for row in results))
        generated_at = timestamp(self.clock() if self.clock is not None else time.time())
        return {
            "results": results, "total": len(results), "status_counts": status_counts,
            "generated_at": generated_at, "benchmark_publication_id": publication_id,
            "external_publications": external_metadata,
        }
