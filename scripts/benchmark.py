"""Measure real inspections and visitor-owned HTTP reads; never use sample reports."""

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import quote

import httpx
import psutil

TARGETS = {
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "smollm2": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
}
PRIMARY_URL = (
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/"
    "93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors"
)
GPT2_URL = (
    "https://huggingface.co/openai-community/gpt2/blob/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
)
PENDING = {"queued", "running"}
TERMINAL = {"completed", "partial"}
ENDPOINTS = ("health", "analysis", "root_graph", "component")


def percentile(values, proportion):
    if not values:
        raise ValueError("Cannot calculate a percentile without measurements")
    return sorted(values)[max(0, math.ceil(len(values) * proportion) - 1)]


def inspect(target, root):
    model, revision = TARGETS[target]
    output = root / target
    command = [
        sys.executable,
        "-c",
        "from model_anatomy.src.cli import main; raise SystemExit(main())",
        model,
        "--revision",
        revision,
        "--output",
        str(output),
        "--config-dir",
        str(root / "config"),
        "--preview-dir",
        str(root / "preview"),
        "--inspect",
    ]
    start, peak = time.perf_counter(), 0
    with (root / f"{target}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [monitor, *monitor.children(recursive=True)]
                peak = max(peak, sum(p.memory_info().rss for p in processes if p.is_running()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if time.perf_counter() - start > 180:
                process.kill()
                process.wait()
                raise RuntimeError("Inspection exceeded benchmark timeout")
            time.sleep(0.025)
    if process.returncode != 0:
        raise RuntimeError(f"Inspection failed; see {target}.log")
    report = json.loads((output / "model.json").read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    analysis = metadata.get("analysis_report", {})
    coverage_keys = (
        "semantic_coverage",
        "module_semantic_coverage",
        "checkpoint_parameter_coverage",
        "tensor_shape_coverage",
    )
    return {
        "model_id": model,
        "revision": revision,
        "seconds": time.perf_counter() - start,
        "peak_process_tree_rss_bytes": peak,
        "weight_payload_bytes": metadata["artifacts"]["weight_payload_bytes_read"],
        "tensor_count": len(report["tensors"]),
        "engine_version": metadata.get("engine_version"),
        "coverage": {key: analysis.get(key) for key in coverage_keys},
        "warnings": analysis.get("warnings", report.get("warnings", [])),
        "conditions": "Fresh Python process; provider/OS caches not flushed; sampled RSS every 25 ms",
    }


def new_client(base, *, transport=None):
    """Create one isolated visitor session; each client owns its cookie jar."""
    return httpx.Client(
        base_url=base,
        timeout=30,
        trust_env=False,
        transport=transport,
    )


def analysis_path(analysis_id):
    return f"/api/v1/analyses/{quote(analysis_id, safe='')}"


def request_json(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def wait_for_analysis(client, analysis_id, *, timeout=300, poll_interval=0.2, on_status=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = request_json(client, "GET", analysis_path(analysis_id))
        if on_status is not None:
            on_status(detail)
        if detail["status"] not in PENDING:
            return detail
        time.sleep(poll_interval)
    raise RuntimeError(f"Analysis {analysis_id} did not finish within {timeout} seconds")


def quality_facts(detail):
    quality = detail.get("quality", {})
    coverage_keys = (
        "semantic_coverage",
        "module_semantic_coverage",
        "checkpoint_parameter_coverage",
        "tensor_shape_coverage",
    )
    return {
        "engine_version": detail.get("engine_version"),
        "coverage": {key: quality.get(key) for key in coverage_keys},
        "quality": quality,
        "warnings": detail.get("warnings", []),
    }


def create_analysis(client, model_url):
    queued = request_json(client, "POST", "/api/v1/analyses", json={"model": model_url})
    return {"id": queued["id"], "model_id": queued["model_id"], "status": queued["status"]}


def submit_analysis(client, model_url):
    started = time.perf_counter()
    queued = create_analysis(client, model_url)
    detail = wait_for_analysis(client, queued["id"])
    if detail["status"] not in TERMINAL:
        raise RuntimeError(f"Analysis ended in unexpected status {detail['status']!r}")
    result = {
        "id": detail["id"],
        "model_id": detail["model_id"],
        "revision": detail["revision"],
        "status": detail["status"],
        "cached": detail["cached"],
        "root_graph_id": detail.get("root_graph_id"),
        "seconds": time.perf_counter() - started,
        **quality_facts(detail),
    }
    return result


def assert_cached_reuse(first, second):
    if first["revision"] != second["revision"]:
        raise RuntimeError("Primary model admissions resolved to different revisions")
    if not second["cached"]:
        raise RuntimeError("Second primary-model admission did not reuse the immutable cache")
    return {
        "verified": True,
        "revision": first["revision"],
        "first_admission_cached": first["cached"],
        "second_admission_cached": second["cached"],
    }


def assert_pinned_target(analysis, target):
    model_id, revision = TARGETS[target]
    if analysis["model_id"] != model_id or analysis["revision"] != revision:
        raise RuntimeError(f"{target} analysis did not match its pinned model ID and revision")


def read_paths_for_analysis(client, analysis):
    """Build read routes only from the submitted analysis owned by this client."""
    root_id = analysis.get("root_graph_id")
    if not root_id:
        raise RuntimeError(f"Analysis {analysis['id']} has no root graph")
    base = analysis_path(analysis["id"])
    graph_id = quote(root_id, safe="")
    graph_path = f"{base}/graphs/{graph_id}"
    graph = request_json(client, "GET", graph_path)
    nodes = graph.get("nodes", [])
    node = next((item for item in nodes if item.get("kind") == "embedding"), None)
    node = node or (nodes[0] if nodes else None)
    if node is None:
        raise RuntimeError(f"Analysis {analysis['id']} root graph has no inspectable component")
    component_path = f"{base}/nodes/{quote(graph['id'], safe='')}/{quote(node['id'], safe='')}"
    return {
        "health": "/health",
        "analysis": base,
        "root_graph": graph_path,
        "component": component_path,
    }


def wait_until_running(client, analysis_id, *, timeout=60, poll_interval=0.1, elapsed_origin=None):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        detail = request_json(client, "GET", analysis_path(analysis_id))
        status = detail["status"]
        now = time.perf_counter()
        observed_at = now - elapsed_origin if elapsed_origin is not None else now
        if status == "running":
            if detail.get("cached"):
                raise RuntimeError(
                    f"Fresh GPT-2 precondition failed: pinned revision {detail.get('revision')} was already cached"
                )
            return {
                "status": status,
                "observed_at_s": observed_at,
                "previous_status_observation": previous,
                "analysis": detail,
            }
        if status not in PENDING:
            if detail.get("cached"):
                raise RuntimeError(
                    f"Fresh GPT-2 precondition failed: pinned revision {detail.get('revision')} was already cached"
                )
            raise RuntimeError(f"Fresh GPT-2 analysis reached {status!r} before a running state was observed")
        previous = {"status": status, "observed_at_s": observed_at}
        time.sleep(poll_interval)
    raise RuntimeError("Fresh GPT-2 analysis was not observed running before the timeout")


def watch_until_terminal(client, analysis_id, elapsed_origin, *, timeout=300, poll_interval=0.1):
    observations = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        detail = request_json(client, "GET", analysis_path(analysis_id))
        observed = {
            "status": detail["status"],
            "observed_at_s": time.perf_counter() - elapsed_origin,
            "analysis": detail,
        }
        observations.append(observed)
        if detail["status"] not in PENDING:
            if detail["status"] not in TERMINAL:
                raise RuntimeError(f"Fresh GPT-2 analysis ended in {detail['status']!r}")
            return observations
        time.sleep(poll_interval)
    raise RuntimeError("Fresh GPT-2 analysis did not finish within the benchmark timeout")


def wholly_inside(sample, window_start, window_end):
    return sample["started_at_s"] >= window_start and sample["finished_at_s"] <= window_end


def summarize_load(
    samples,
    first_running_at,
    last_running_at,
    visitor_count,
    *,
    rounds,
    start_transition_bracket_s=None,
    end_transition_bracket_s=None,
):
    """Score only reads wholly between first and last status polls that returned running."""
    if last_running_at < first_running_at:
        raise RuntimeError("Invalid observed GPT-2 running interval")
    active = [sample for sample in samples if wholly_inside(sample, first_running_at, last_running_at)]
    if not active:
        raise RuntimeError("HTTP load produced no measurements wholly inside observed running polls")
    observed_visitors = {sample["visitor"] for sample in active}
    expected_visitors = set(range(visitor_count))
    if observed_visitors != expected_visitors:
        missing = sorted(expected_visitors - observed_visitors)
        raise RuntimeError(
            f"Not enough visitors measured inside observed running polls: "
            f"observed {len(observed_visitors)}/{visitor_count}; missing {missing}"
        )
    observed_endpoints = sorted({sample["endpoint"] for sample in active})
    missing = sorted(set(ENDPOINTS) - set(observed_endpoints))
    if missing:
        raise RuntimeError(f"Observed running polls lack HTTP measurements for: {', '.join(missing)}")
    if rounds >= 20:
        missing_by_visitor = {
            visitor: sorted(
                set(ENDPOINTS) - {sample["endpoint"] for sample in active if sample["visitor"] == visitor}
            )
            for visitor in range(visitor_count)
        }
        missing_by_visitor = {visitor: routes for visitor, routes in missing_by_visitor.items() if routes}
        if missing_by_visitor:
            raise RuntimeError(
                f"Per-visitor route coverage is incomplete inside running polls: {missing_by_visitor}"
            )
    durations = [sample["seconds"] for sample in active]
    p95 = percentile(durations, 0.95)
    return {
        "visitors": visitor_count,
        "requests_attempted": len(samples),
        "requests_wholly_inside_observed_running_polls": len(active),
        "visitors_wholly_inside_observed_running_polls": len(observed_visitors),
        "endpoints_wholly_inside_observed_running_polls": observed_endpoints,
        "p50_seconds": percentile(durations, 0.5),
        "p95_seconds": p95,
        "max_seconds": max(durations),
        "p95_under_one_second": p95 < 1,
        "observed_running_window_s": {
            "start_first_running_poll": first_running_at,
            "end_last_running_poll": last_running_at,
        },
        "status_poll_uncertainty_s": {
            "start_transition_bracket": start_transition_bracket_s,
            "end_transition_bracket": end_transition_bracket_s,
            "meaning": "State transitions occurred between polls; only requests wholly within running polls count",
        },
        "conditions": "Independent visitor cookie sessions; all visitors and all four routes required",
    }


def http_load(visitor_sessions, rounds, running_client, running_analysis_id, *, elapsed_origin):
    initial = wait_until_running(running_client, running_analysis_id, elapsed_origin=elapsed_origin)
    samples = []

    def observe(index, session):
        client, paths = session["client"], session["paths"]
        measurements = []
        for round_index in range(rounds):
            endpoint = ENDPOINTS[round_index % len(ENDPOINTS)]
            started = time.perf_counter()
            request_json(client, "GET", paths[endpoint])
            finished = time.perf_counter()
            measurements.append(
                {
                    "visitor": index,
                    "endpoint": endpoint,
                    "started_at_s": started - elapsed_origin,
                    "finished_at_s": finished - elapsed_origin,
                    "seconds": finished - started,
                }
            )
            time.sleep(0.05)
        return measurements

    with ThreadPoolExecutor(max_workers=len(visitor_sessions) + 1) as pool:
        watcher = pool.submit(
            watch_until_terminal,
            running_client,
            running_analysis_id,
            elapsed_origin,
        )
        visitors = pool.map(lambda item: observe(*item), enumerate(visitor_sessions))
        for measurements in visitors:
            samples.extend(measurements)
        observations = watcher.result()

    terminal = next(item for item in observations if item["status"] not in PENDING)
    running_observations = [initial, *[item for item in observations if item["status"] == "running"]]
    last_running = running_observations[-1]
    previous = initial["previous_status_observation"]
    start_bracket = initial["observed_at_s"] - previous["observed_at_s"] if previous else None
    end_bracket = terminal["observed_at_s"] - last_running["observed_at_s"]
    try:
        summary = summarize_load(
            samples,
            initial["observed_at_s"],
            last_running["observed_at_s"],
            len(visitor_sessions),
            rounds=rounds,
            start_transition_bracket_s=start_bracket,
            end_transition_bracket_s=end_bracket,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}; requests_attempted={len(samples)}; "
            f"first_running_poll_s={initial['observed_at_s']:.6f}; "
            f"last_running_poll_s={last_running['observed_at_s']:.6f}; "
            f"terminal_poll_s={terminal['observed_at_s']:.6f}; "
            f"start_transition_poll_bracket_s={start_bracket}; "
            f"end_transition_poll_bracket_s={end_bracket:.6f}"
        ) from exc
    summary["status_observations"] = [
        *([previous] if previous else []),
        {"status": initial["status"], "observed_at_s": initial["observed_at_s"]},
        *[{"status": item["status"], "observed_at_s": item["observed_at_s"]} for item in observations],
    ]
    summary["fresh_gpt2"] = quality_facts(observations[-1]["analysis"])
    summary["fresh_gpt2"].update(
        {
            "model_id": observations[-1]["analysis"]["model_id"],
            "revision": observations[-1]["analysis"]["revision"],
            "status": observations[-1]["status"],
            "cached": observations[-1]["analysis"]["cached"],
        }
    )
    return summary


def submit_benchmark(base, visitors, rounds, *, client_factory=new_client):
    """Admit real jobs and measure owned reads while a fresh GPT-2 job runs."""
    if visitors < 1 or rounds < len(ENDPOINTS):
        raise ValueError(f"Need at least one visitor and {len(ENDPOINTS)} rounds per visitor")
    with ExitStack() as stack:
        clients = [stack.enter_context(client_factory(base)) for _ in range(visitors)]
        admissions = []
        first = submit_analysis(clients[0], PRIMARY_URL)
        assert_pinned_target(first, "smollm2")
        admissions.append({"visitor": 0, "purpose": "primary", **first})
        second = submit_analysis(clients[0], PRIMARY_URL)
        assert_pinned_target(second, "smollm2")
        admissions.append({"visitor": 0, "purpose": "primary_cache_check", **second})
        cache = assert_cached_reuse(first, second)

        sessions = [{"client": clients[0], "analysis": first}]
        for index, client in enumerate(clients[1:], start=1):
            analysis = submit_analysis(client, PRIMARY_URL)
            assert_pinned_target(analysis, "smollm2")
            admissions.append({"visitor": index, "purpose": "primary", **analysis})
            sessions.append({"client": client, "analysis": analysis})
        for session in sessions:
            session["paths"] = read_paths_for_analysis(session["client"], session["analysis"])

        gpt2_client = stack.enter_context(client_factory(base))
        gpt2 = create_analysis(gpt2_client, GPT2_URL)
        elapsed_origin = time.perf_counter()
        http = http_load(
            sessions,
            rounds,
            gpt2_client,
            gpt2["id"],
            elapsed_origin=elapsed_origin,
        )
        gpt2 = http["fresh_gpt2"]
        if gpt2["cached"]:
            raise RuntimeError(f"Fresh GPT-2 precondition failed: {gpt2['revision']} finished from cache")
        assert_pinned_target(gpt2, "gpt2")
        admissions.append({"visitor": "separate-gpt2-visitor", "purpose": "fresh_gpt2", **gpt2})
    return {
        "admissions": admissions,
        "admission_count": len(admissions),
        "primary_cache_reuse": cache,
        "fresh_gpt2": {
            "model_id": gpt2["model_id"],
            "revision": gpt2["revision"],
            "cached": gpt2["cached"],
            "status": gpt2["status"],
        },
        "http": http,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=Path(".local/benchmark"))
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Required for HTTP benchmarking; submits real jobs and consumes visitor quotas",
    )
    parser.add_argument("--visitors", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args(argv)
    if not args.inspect and not args.submit:
        parser.error("choose --inspect and/or --submit; HTTP benchmarking always requires --submit")
    if not 1 <= args.visitors <= 18 or not len(ENDPOINTS) <= args.rounds <= 1000:
        parser.error("Visitors/rounds are outside the benchmark bounds (maximum admissions: 20)")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "conditions": {
            "http_admissions_are_opt_in": True,
            "http_admissions_expected": args.visitors + 2 if args.submit else 0,
            "visitor_sessions_are_independent": True,
            "runtime_or_report_mocks": False,
        },
    }

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    try:
        if args.inspect:
            result["engine"] = []
            for target in TARGETS:
                result["engine"].append(inspect(target, args.output))
                save()
        if args.submit:
            result["http_benchmark"] = submit_benchmark(args.base_url, args.visitors, args.rounds)
        save()
    except Exception as exc:  # noqa: BLE001 - persist any benchmark failure as machine-readable evidence
        result["failure"] = {"stage": "benchmark", "type": type(exc).__name__, "message": str(exc)}
        save()
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    if args.submit:
        return 0 if result["http_benchmark"]["http"]["p95_under_one_second"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
