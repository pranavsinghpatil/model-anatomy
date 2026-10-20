"""Run pinned Hugging Face anatomy cases in isolated, bounded Python processes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REFERENCE_RE = re.compile(
    r"^https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/(?:blob|resolve)/([0-9a-f]{40})/.+$"
)
COVERAGE_KEYS = (
    "semantic_coverage",
    "module_semantic_coverage",
    "checkpoint_parameter_coverage",
    "tensor_shape_coverage",
)
MAX_WARNINGS = 20
MAX_WARNING_CHARS = 500
SERVICE_STORAGE_LIMIT_BYTES = 32 * 1024**2


class CaseStageError(RuntimeError):
    """Preserve the pipeline stage when a case cannot produce a report."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def validate_manifest(data: Any, *, allow_small: bool = False) -> list[dict[str, Any]]:
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get("cases"), list)
    ):
        raise ValueError("manifest must be an object with schema_version: 1 and a cases array")
    cases = data["cases"]
    if not allow_small and len(cases) < 100:
        raise ValueError(
            f"production manifest requires at least 100 cases (found {len(cases)}); use --allow-small only for development"
        )
    seen: set[str] = set()
    for i, case in enumerate(cases):
        where = f"cases[{i}]"
        if not isinstance(case, dict):
            raise TypeError(f"{where} must be an object")
        model_id = case.get("model_id")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or model_id != model_id.strip()
            or "/" not in model_id
        ):
            raise ValueError(f"{where}.model_id must be a non-empty Hugging Face repository ID")
        if model_id in seen:
            raise ValueError(f"duplicate model_id: {model_id}")
        seen.add(model_id)
        if not isinstance(case.get("category"), str) or not case["category"].strip():
            raise ValueError(f"{where}.category must be a non-empty string")
        if not isinstance(case.get("revision"), str) or not SHA_RE.fullmatch(case["revision"]):
            raise ValueError(f"{where}.revision must be a pinned lowercase 40-character commit SHA")
        reference = case.get("reference")
        if reference is not None:
            match = REFERENCE_RE.fullmatch(reference) if isinstance(reference, str) else None
            if match is None:
                raise ValueError(f"{where}.reference must be a pinned HTTPS Hugging Face blob/resolve URL")
            if match.group(1) != model_id or match.group(2) != case["revision"]:
                raise ValueError(
                    f"{where}.reference repository and revision must match model_id and revision"
                )
        for optional in ("declared_task", "notes"):
            if optional in case and case[optional] is not None and not isinstance(case[optional], str):
                raise ValueError(f"{where}.{optional} must be a string when present")
    return cases


def source_tree_sha256(source_root: Path = BACKEND_ROOT / "model_anatomy") -> str:
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def case_identity(
    case: dict[str, Any],
    engine_version: str,
    runner_sha256: str | None = None,
    engine_source_sha256: str | None = None,
    run_environment: dict[str, Any] | None = None,
) -> str:
    identity = {
        k: case.get(k) for k in ("model_id", "category", "revision", "reference", "declared_task", "notes")
    }
    identity["engine_version"] = engine_version
    identity["runner_sha256"] = runner_sha256
    identity["engine_source_sha256"] = engine_source_sha256
    identity["run_environment"] = run_environment
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for name in (
        "torch",
        "transformers",
        "safetensors",
        "requests",
        "huggingface-hub",
        "pydantic",
        "psutil",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def summarize_observation(records: list[dict[str, Any]], key: str, expected: Any) -> dict[str, Any]:
    observed = [record[key] for record in records if key in record and record[key] is not None]
    if any(value != expected for value in observed):
        value = False
    elif len(observed) == len(records) and records:
        value = True
    else:
        value = None
    return {"value": value, "observed_cases": len(observed), "total_cases": len(records)}


def classify_failure(exc: Exception) -> dict[str, str]:
    """Classify failures by type; artifact_error does not prove repository corruption."""
    from requests.exceptions import RequestException

    from model_anatomy.src.errors import ArtifactError, UnsupportedModelError

    stage = exc.stage if isinstance(exc, CaseStageError) else "case"
    cause = exc.cause if isinstance(exc, CaseStageError) else exc
    try:
        import huggingface_hub.errors as hf_errors

        http_errors = (RequestException, hf_errors.HfHubHTTPError)
    except ImportError:
        http_errors = (RequestException,)

    if isinstance(cause, MemoryError):
        status = "memorylimit"
    elif isinstance(cause, UnsupportedModelError):
        status = "unsupported"
    elif isinstance(cause, http_errors):
        status = "http_access" if getattr(cause, "response", None) is not None else "network"
    elif isinstance(cause, ArtifactError):
        status = "artifact_error"
    elif stage == "report_validation":
        status = "invalid_report"
    else:
        status = "failure"
    return {
        "status": status,
        "stage": stage,
        "cause_type": type(cause).__name__,
        "cause_message": str(cause)[:1500],
    }


def _bounded(items: Any, limit: int = MAX_WARNINGS) -> list[str]:
    if not isinstance(items, (list, tuple)):
        return []
    return [str(item)[:MAX_WARNING_CHARS] for item in items[:limit]]


def _analyze_case(case: dict[str, Any], *, report_directory: Path | None = None) -> dict[str, Any]:
    """Worker entry point; imports the reconstruction engine only in the fresh child."""
    from model_anatomy.api.schemas import ArtifactInspectionResponse, GraphResponse, NodeInspection
    from model_anatomy.service.views import artifact_view, graph_view, node_view
    from model_anatomy.src.ir import ModelIR
    from model_anatomy.src.pipeline import reconstruct
    from model_anatomy.src.version import ENGINE_VERSION

    started = time.perf_counter()
    target = case.get("reference") or case["model_id"]
    try:
        ir = reconstruct(target, revision=case["revision"], allow_artifact_inspection=True)
    except Exception as exc:
        raise CaseStageError("reconstruction", exc) from exc
    try:
        ir.root_graph.validate()
        report_json = ir.to_json()
        report_bytes = len(report_json.encode("utf-8"))
        restored = ModelIR.from_json(report_json)
        restored.root_graph.validate()
    except Exception as exc:
        raise CaseStageError("report_validation", exc) from exc
    inspection_only = ir.metadata.get("inspection", {}).get("kind") == "artifact_inspection"
    graphs = list(ir.root_graph.walk())
    graph_node_count = sum(len(graph.nodes) for graph in graphs)
    graph_edge_count = sum(len(graph.edges) for graph in graphs)
    tensor_count = len(ir.tensors)
    analysis = ir.metadata.get("analysis_report", {})
    warning_items = _bounded(analysis.get("warnings", []))
    reasons = []
    if inspection_only and ir.metadata.get("inspection", {}).get("reason_code"):
        reasons.append(ir.metadata["inspection"]["reason_code"])
    blockers = ir.metadata.get("model_profile", {}).get("architecture_blockers", [])
    reasons.extend(_bounded(blockers, 10))

    graph_validation = {"count": 0, "errors": []}
    public_schema = {"graphs": 0, "errors": []}
    if not inspection_only:
        for graph in graphs:
            graph.validate()
            graph_validation["count"] += 1
            try:
                GraphResponse.model_validate(graph_view(ir, graph.id))
                public_schema["graphs"] += 1
            except Exception as exc:  # noqa: BLE001 - preserve every schema incompatibility as evidence
                public_schema["errors"].append(f"{graph.id}: {type(exc).__name__}: {str(exc)[:300]}")
        if graph_node_count:
            empty_node_sample = None
            for graph in graphs:
                for candidate in graph.nodes:
                    if not (candidate.module_path or candidate.source_modules):
                        continue
                    try:
                        view = node_view(ir, graph.id, candidate.id)
                        NodeInspection.model_validate(view)
                        sample = {
                            "graph_id": graph.id,
                            "node_id": candidate.id,
                            "tensor_count": view["tensor_count"],
                        }
                        if view["tensor_count"]:
                            graph_validation["node_inspection"] = sample
                            break
                        empty_node_sample = sample
                    except Exception as exc:  # noqa: BLE001 - preserve any node response incompatibility
                        graph_validation.setdefault("node_inspection_errors", []).append(
                            f"{type(exc).__name__}: {str(exc)[:300]}"
                        )
                if graph_validation.get("node_inspection", {}).get("tensor_count", 0):
                    break
            if "node_inspection" not in graph_validation and empty_node_sample:
                graph_validation["node_inspection"] = empty_node_sample
    artifact_response = None
    artifact_schema_error = None
    try:
        artifact_response = ArtifactInspectionResponse.model_validate(artifact_view(ir)).model_dump()
    except Exception as exc:  # noqa: BLE001 - report all artifact response validation failures
        artifact_schema_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    report_roundtrip = {
        "valid": restored.to_dict() == ir.to_dict(),
        "schema_version": ir.schema_version,
        "errors": [],
    }
    payload_bytes = ir.metadata.get("artifacts", {}).get("weight_payload_bytes_read")
    forward_executed = analysis.get("forward_executed")
    storage_contract_valid = report_bytes <= SERVICE_STORAGE_LIMIT_BYTES
    invalid_report = (
        not report_roundtrip["valid"]
        or bool(graph_validation["errors"])
        or bool(graph_validation.get("node_inspection_errors"))
        or bool(public_schema["errors"])
        or artifact_schema_error is not None
        or payload_bytes != 0
        or forward_executed is not False
        or not storage_contract_valid
    )
    coverages = {key: analysis.get(key) for key in COVERAGE_KEYS}
    fully_covered = all(
        isinstance(coverages[key], (int, float)) and coverages[key] == 1 for key in COVERAGE_KEYS
    )
    if not fully_covered:
        reasons.append("coverage_not_complete")
    status = (
        "invalid_report"
        if invalid_report
        else "inspected"
        if inspection_only
        else "completed"
        if fully_covered
        and not analysis.get("warnings")
        and not analysis.get("unknown_nodes", 0)
        and not analysis.get("unmapped_checkpoint_tensors", [])
        else "partial"
    )
    report_digest = None
    if report_directory is not None and status in {"completed", "partial", "inspected"}:
        report_digest = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
        report_path = report_directory / f"{report_digest}.json"
        report_directory.mkdir(parents=True, exist_ok=True)
        report_data = report_json.encode("utf-8")
        if not report_path.exists() or report_path.read_bytes() != report_data:
            temporary_path = report_path.with_suffix(f".{os.getpid()}.tmp")
            try:
                with temporary_path.open("wb") as stream:
                    stream.write(report_data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, report_path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return {
        "model_id": case["model_id"],
        "category": case["category"],
        "revision": ir.metadata.get("revision", case["revision"]),
        "engine_version": ENGINE_VERSION,
        "status": status,
        "architecture": None if inspection_only else ir.architecture,
        "inspection_only": inspection_only,
        "coverage": coverages,
        "warning_count": len(analysis.get("warnings", []))
        if isinstance(analysis.get("warnings", []), list)
        else 0,
        "warnings": warning_items,
        "graph_count": len(graphs) if not inspection_only else 0,
        "node_count": graph_node_count if not inspection_only else 0,
        "edge_count": graph_edge_count if not inspection_only else 0,
        "tensor_count": tensor_count,
        "serialized_report_bytes": report_bytes,
        "report_digest": report_digest,
        "service_storage_limit_bytes": SERVICE_STORAGE_LIMIT_BYTES,
        "storage_contract_valid": storage_contract_valid,
        "service_contract_valid": not invalid_report,
        "parameter_counts": {
            "unique": ir.metadata.get("unique_parameter_count"),
            "logical": ir.metadata.get("logical_parameter_count"),
            "checkpoint_tensor_elements": ir.metadata.get("checkpoint_tensor_element_count"),
        },
        "weight_payload_bytes_read": payload_bytes,
        "zero_weight_payload_bytes": payload_bytes == 0,
        "forward_executed": forward_executed,
        "report_roundtrip": report_roundtrip,
        "graph_validation": {**graph_validation, "valid": not graph_validation["errors"]},
        "public_graph_schema": {**public_schema, "valid": not public_schema["errors"]},
        "artifact_inspection_schema": {
            "valid": artifact_schema_error is None,
            "error": artifact_schema_error,
            "kind": artifact_response.get("kind") if artifact_response else None,
        },
        "reason_codes": list(dict.fromkeys(reasons)),
        "native_blockers": _bounded(blockers, 10),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _child_entry(case: dict[str, Any], send_conn) -> None:
    try:
        record = _analyze_case(case)
        send_conn.send(("result", record))
    except Exception as exc:  # noqa: BLE001 - child must return an outcome for every engine failure
        classified = classify_failure(exc)
        try:
            send_conn.send(
                (
                    "error",
                    {
                        "classification": classified["status"],
                        "stage": classified["stage"],
                        "cause_type": classified["cause_type"],
                        "message": classified["cause_message"],
                        "traceback": "".join(traceback.format_exception(exc))[-4000:],
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        send_conn.close()


def _terminate_tree(process: mp.Process, *, grace: float = 1.0) -> None:
    if process.pid is None or not process.is_alive():
        return
    children = []
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for item in reversed(children):
        try:
            item.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # multiprocessing owns this child's waitpid status. psutil.wait_procs on
    # the same PID can reap it first on POSIX and strand Process.exitcode=None.
    process.terminate()
    process.join(timeout=grace)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace)
    if children:
        _, alive = psutil.wait_procs(children, timeout=grace)
        for item in alive:
            try:
                item.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass


def run_case(
    case: dict[str, Any], *, timeout: float = 120, memory_bytes: int = 4 * 1024**3, worker_target=_child_entry
) -> dict[str, Any]:
    if timeout <= 0 or timeout > 120 or not math.isfinite(timeout):
        raise ValueError("timeout must be greater than zero and at most 120 seconds")
    if memory_bytes <= 0 or memory_bytes > 4 * 1024**3 or not math.isfinite(memory_bytes):
        raise ValueError("memory budget must be positive and at most 4 GiB")
    context = mp.get_context("spawn")
    recv_conn, send_conn = context.Pipe(duplex=False)
    process = context.Process(target=worker_target, args=(case, send_conn), daemon=False)
    started = time.perf_counter()
    process.start()
    send_conn.close()
    peak = 0
    outcome = None
    try:
        monitor = psutil.Process(process.pid)
        while process.is_alive():
            try:
                tree = [monitor, *monitor.children(recursive=True)]
                rss = sum(item.memory_info().rss for item in tree if item.is_running())
                peak = max(peak, rss)
                if rss > memory_bytes:
                    outcome = {
                        "status": "memorylimit",
                        "message": f"process tree RSS {rss} exceeded {memory_bytes} bytes",
                    }
                    _terminate_tree(process)
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.perf_counter() - started > timeout:
                outcome = {"status": "timeout", "message": f"case exceeded {timeout:g} second wall timeout"}
                _terminate_tree(process)
                break
            if recv_conn.poll(0):
                try:
                    kind, payload = recv_conn.recv()
                    outcome = payload
                    if kind == "error":
                        outcome["status"] = outcome.pop("classification")
                    break
                except EOFError:
                    pass
            time.sleep(0.05)
        if outcome is None and time.perf_counter() - started > timeout:
            outcome = {"status": "timeout", "message": f"case exceeded {timeout:g} second wall timeout"}
            _terminate_tree(process)
        process.join(timeout=1)
        if process.is_alive():
            _terminate_tree(process)
            process.join(timeout=1)
        if outcome is None:
            if recv_conn.poll(0.1):
                try:
                    kind, payload = recv_conn.recv()
                    outcome = payload
                    if kind == "error":
                        outcome["status"] = outcome.pop("classification")
                except EOFError:
                    pass
            if outcome is None:
                outcome = {
                    "status": "failure",
                    "message": f"child exited without a result (exit code {process.exitcode})",
                }
    finally:
        recv_conn.close()
        if process.is_alive():
            _terminate_tree(process)
            process.join(timeout=1)
        process.close()
    outcome.update(
        {"peak_process_tree_rss_bytes": peak, "wall_seconds": round(time.perf_counter() - started, 3)}
    )
    if "model_id" not in outcome:
        outcome.update(
            {
                "model_id": case.get("model_id"),
                "category": case.get("category"),
                "revision": case.get("revision"),
            }
        )
    return outcome


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return found
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid NDJSON at line {line_number}: {exc}") from exc
            if isinstance(item, dict) and isinstance(item.get("case_identity"), str):
                found[item["case_identity"]] = item
    return found


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_matrix(
    manifest_path: Path,
    output_path: Path,
    *,
    timeout=120,
    memory_bytes=4 * 1024**3,
    allow_small=False,
    resume=False,
    limit: int | None = None,
) -> dict[str, Any]:
    from model_anatomy.src.version import ENGINE_VERSION

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = validate_manifest(manifest, allow_small=allow_small)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    completed = read_existing(output_path) if resume else {}
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine_source_hash = source_tree_sha256()
    dependencies = dependency_versions()
    run_environment = {
        "timeout_seconds": timeout,
        "memory_bytes": memory_bytes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
    results = []
    total = len(cases)
    for index, case in enumerate(cases, 1):
        identity = case_identity(case, ENGINE_VERSION, runner_sha256, engine_source_hash, run_environment)
        previous = completed.get(identity)
        if previous is not None:
            record = {**previous, "resumed": True}
            print(f"[{index}/{total}] {case['model_id']} -> resume {record.get('status')}", flush=True)
        else:
            started_at = datetime.now(UTC).isoformat()
            print(f"[{index}/{total}] {case['model_id']} ({case['category']}) starting", flush=True)
            record = run_case(case, timeout=timeout, memory_bytes=memory_bytes)
            record.update(
                {
                    "case_identity": identity,
                    "runner_sha256": runner_sha256,
                    "engine_source_sha256": engine_source_hash,
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "resumed": False,
                }
            )
            append_record(output_path, record)
            print(
                f"[{index}/{total}] {case['model_id']} -> {record.get('status')} ({record.get('wall_seconds', 0):.1f}s)",
                flush=True,
            )
        results.append(record)
    counts: dict[str, int] = {}
    for item in results:
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "record_file": str(output_path.resolve()),
        "cases": total,
        "status_counts": counts,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "engine_version": ENGINE_VERSION,
            "runner_sha256": runner_sha256,
            "engine_source_sha256": engine_source_hash,
            "engine_source_root": str((BACKEND_ROOT / "model_anatomy").resolve()),
            "dependencies": dependencies,
            "workers": 1,
            "wall_timeout_seconds": timeout,
            "process_tree_rss_limit_bytes": memory_bytes,
            "rss_sampling_interval_ms": 50,
            "weight_payload_loading_enabled": False,
            "forward_execution_enabled": False,
        },
        "observed_invariants": {
            "zero_weight_payload_bytes": summarize_observation(results, "weight_payload_bytes_read", 0),
            "forward_not_executed": summarize_observation(results, "forward_executed", False),
        },
        "results": results,
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/model-matrix-100.json"))
    parser.add_argument("--output", type=Path, default=Path("output/model_matrix.ndjson"))
    parser.add_argument("--summary", type=Path, default=None, help="optional JSON summary path")
    parser.add_argument(
        "--timeout", type=float, default=120, help="per-case wall timeout (maximum 120 seconds)"
    )
    parser.add_argument(
        "--memory-gib", type=float, default=4, help="per-case sampled process-tree RSS cap (maximum 4 GiB)"
    )
    parser.add_argument(
        "--allow-small", action="store_true", help="allow fewer than 100 cases for development runs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse NDJSON rows matching case identity and engine version"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first N cases (the full manifest is validated)"
    )
    args = parser.parse_args(argv)
    try:
        summary = run_matrix(
            args.manifest,
            args.output,
            timeout=args.timeout,
            memory_bytes=int(args.memory_gib * 1024**3),
            allow_small=args.allow_small,
            resume=args.resume,
            limit=args.limit,
        )
        encoded = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports validation and run failures consistently
        print(f"model_matrix: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
