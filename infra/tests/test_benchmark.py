"""Deterministic tests for benchmark routing and measurement claims."""

import json
import time
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest
from scripts import benchmark


def test_cookie_clients_submit_and_read_only_their_own_analyses():
    owners = {}
    guest_ids = iter(range(10))
    model_admissions = {}

    def handle(request):
        cookie = SimpleCookie()
        cookie.load(request.headers.get("cookie", ""))
        guest = cookie.get("model_anatomy_visitor")
        guest = guest.value if guest else None
        if request.method == "POST" and request.url.path == "/api/v1/analyses":
            guest = guest or f"guest-{next(guest_ids)}"
            model_url = json.loads(request.content)["model"]
            model_id, _revision = (
                ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
                if "SmolLM2-135M" in model_url
                else ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
            )
            count = model_admissions.get(model_id, 0)
            model_admissions[model_id] = count + 1
            analysis_id = f"job-{guest}-{count}"
            owners[analysis_id] = (guest, model_id)
            headers = {"set-cookie": f"model_anatomy_visitor={guest}; Path=/"}
            return httpx.Response(
                202,
                headers=headers,
                json={"id": analysis_id, "model_id": model_id, "status": "queued"},
            )

        parts = [unquote(part) for part in request.url.path.split("/") if part]
        if len(parts) >= 4 and parts[:2] == ["api", "v1"] and parts[2] == "analyses":
            analysis_id = parts[3]
            owner = owners.get(analysis_id)
            if owner is None or owner[0] != guest:
                return httpx.Response(404, json={"error": "not found"})
            if len(parts) == 4:
                return httpx.Response(
                    200,
                    json={
                        "id": analysis_id,
                        "model_id": owner[1],
                        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                        "status": "completed",
                        "cached": model_admissions[owner[1]] > 1,
                        "root_graph_id": "root",
                        "engine_version": "test-engine",
                        "quality": {"semantic_coverage": 1.0},
                        "warnings": [],
                    },
                )
            if len(parts) == 6 and parts[4] == "graphs":
                return httpx.Response(
                    200,
                    json={"id": "root", "nodes": [{"id": "embed", "kind": "embedding"}]},
                )
        return httpx.Response(404, json={"error": "unknown route"})

    transport = httpx.MockTransport(handle)
    with (
        benchmark.new_client("https://benchmark.test", transport=transport) as first_client,
        benchmark.new_client("https://benchmark.test", transport=transport) as second_client,
    ):
        first = benchmark.submit_analysis(first_client, benchmark.PRIMARY_URL)
        second = benchmark.submit_analysis(second_client, benchmark.PRIMARY_URL)
        assert first_client.cookies.get("model_anatomy_visitor") != second_client.cookies.get(
            "model_anatomy_visitor"
        )
        assert (
            benchmark.request_json(first_client, "GET", benchmark.analysis_path(first["id"]))["id"]
            == first["id"]
        )
        assert second_client.get(benchmark.analysis_path(first["id"])).status_code == 404
        paths = benchmark.read_paths_for_analysis(first_client, first)
        assert paths["analysis"] == f"/api/v1/analyses/{first['id']}"
        assert paths["root_graph"].startswith(paths["analysis"] + "/graphs/")
        assert paths["component"].startswith(paths["analysis"] + "/nodes/")
        assert benchmark.assert_cached_reuse(first, second)["verified"]


def test_no_historical_sample_route_or_non_artifact_model_urls():
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    assert "example-smollm" not in source
    assert "/analyses/example-" not in source
    assert "/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/model.safetensors" in benchmark.PRIMARY_URL
    assert "/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors" in benchmark.GPT2_URL


def test_cache_reuse_claim_requires_same_revision_and_cached_second_admission():
    first = {"revision": "abc", "cached": False}
    second = {"revision": "abc", "cached": True}
    assert benchmark.assert_cached_reuse(first, second)["second_admission_cached"]
    with pytest.raises(RuntimeError, match="did not reuse"):
        benchmark.assert_cached_reuse(first, {"revision": "abc", "cached": False})
    with pytest.raises(RuntimeError, match="different revisions"):
        benchmark.assert_cached_reuse(first, {"revision": "xyz", "cached": True})


def test_target_claim_requires_exact_model_and_pinned_revision():
    benchmark.assert_pinned_target(
        {
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        },
        "smollm2",
    )
    with pytest.raises(RuntimeError, match="did not match its pinned"):
        benchmark.assert_pinned_target({"model_id": "openai-community/gpt2", "revision": "other"}, "smollm2")


def test_fresh_gpt2_cache_hit_is_rejected_before_reporting_a_running_interval():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"id": "job", "status": "completed", "cached": True, "revision": "old-revision"},
        )
    )
    with (
        benchmark.new_client("https://benchmark.test", transport=transport) as client,
        pytest.raises(RuntimeError, match="Fresh GPT-2 precondition failed.*old-revision"),
    ):
        benchmark.wait_until_running(client, "job", poll_interval=0)


def test_http_submission_requires_explicit_opt_in(capsys):
    with pytest.raises(SystemExit) as error:
        benchmark.main([])
    assert error.value.code == 2
    assert "HTTP benchmarking always requires --submit" in capsys.readouterr().err


def test_percentile_uses_nearest_rank_and_requires_data():
    assert benchmark.percentile([4, 1, 3, 2], 0.5) == 2
    assert benchmark.percentile([4, 1, 3, 2], 0.95) == 4
    with pytest.raises(ValueError, match="without measurements"):
        benchmark.percentile([], 0.95)


def test_latency_summary_scores_only_whole_requests_inside_first_and_last_running_polls():
    samples = [
        {
            "visitor": visitor,
            "endpoint": endpoint,
            "started_at_s": 1.1 + index * 0.01,
            "finished_at_s": 1.11 + index * 0.01,
            "seconds": 0.01,
        }
        for index, (visitor, endpoint) in enumerate(
            (visitor, endpoint) for visitor in range(2) for endpoint in benchmark.ENDPOINTS
        )
    ]
    samples.extend(
        [
            {
                "visitor": 0,
                "endpoint": "health",
                "started_at_s": 0.99,
                "finished_at_s": 1.01,
                "seconds": 0.02,
            },
            {
                "visitor": 1,
                "endpoint": "component",
                "started_at_s": 1.99,
                "finished_at_s": 2.01,
                "seconds": 0.02,
            },
        ]
    )
    result = benchmark.summarize_load(
        samples,
        1.0,
        2.0,
        visitor_count=2,
        rounds=10,
        start_transition_bracket_s=0.1,
        end_transition_bracket_s=0.2,
    )
    assert result["requests_wholly_inside_observed_running_polls"] == 8
    assert result["visitors_wholly_inside_observed_running_polls"] == 2
    assert set(result["endpoints_wholly_inside_observed_running_polls"]) == set(benchmark.ENDPOINTS)
    assert result["status_poll_uncertainty_s"]["start_transition_bracket"] == 0.1
    assert result["status_poll_uncertainty_s"]["end_transition_bracket"] == 0.2
    assert result["p95_under_one_second"]

    with pytest.raises(RuntimeError, match="no measurements"):
        benchmark.summarize_load(samples, 3.0, 4.0, visitor_count=2, rounds=10)
    with pytest.raises(RuntimeError, match="Not enough visitors"):
        benchmark.summarize_load(samples[:4], 1.0, 2.0, visitor_count=2, rounds=10)
    without_component = [sample for sample in samples if sample["endpoint"] != "component"]
    with pytest.raises(RuntimeError, match="lack HTTP measurements"):
        benchmark.summarize_load(without_component, 1.0, 2.0, visitor_count=2, rounds=10)


def test_twenty_rounds_require_all_routes_for_each_visitor():
    samples = [
        {
            "visitor": visitor,
            "endpoint": endpoint,
            "started_at_s": 1.1,
            "finished_at_s": 1.2,
            "seconds": 0.1,
        }
        for visitor in range(2)
        for endpoint in benchmark.ENDPOINTS
    ]
    incomplete = [
        sample for sample in samples if not (sample["visitor"] == 1 and sample["endpoint"] == "component")
    ]
    with pytest.raises(RuntimeError, match="Per-visitor route coverage"):
        benchmark.summarize_load(incomplete, 1.0, 2.0, visitor_count=2, rounds=20)


def test_http_load_measures_owned_read_paths_while_job_is_observed_running():
    state = {"running_started": None}

    def handle(request):
        if request.url.path == "/api/v1/analyses/gpt2-job":
            now = time.monotonic()
            if state["running_started"] is None:
                state["running_started"] = now
            status = "running" if now - state["running_started"] < 0.3 else "completed"
            return httpx.Response(
                200,
                json={
                    "id": "gpt2-job",
                    "model_id": "openai-community/gpt2",
                    "revision": "pinned",
                    "status": status,
                    "cached": False,
                    "engine_version": "test-engine",
                    "quality": {},
                    "warnings": [],
                },
            )
        if request.url.path == "/health" or "/api/v1/analyses/owned-" in request.url.path:
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": "unexpected path"})

    transport = httpx.MockTransport(handle)
    with (
        benchmark.new_client("https://benchmark.test", transport=transport) as status_client,
        benchmark.new_client("https://benchmark.test", transport=transport) as first_client,
        benchmark.new_client("https://benchmark.test", transport=transport) as second_client,
    ):
        sessions = [
            {
                "client": first_client,
                "paths": {
                    endpoint: f"/api/v1/analyses/owned-0/{endpoint}" for endpoint in benchmark.ENDPOINTS
                },
            },
            {
                "client": second_client,
                "paths": {
                    endpoint: f"/api/v1/analyses/owned-1/{endpoint}" for endpoint in benchmark.ENDPOINTS
                },
            },
        ]
        result = benchmark.http_load(
            sessions,
            rounds=4,
            running_client=status_client,
            running_analysis_id="gpt2-job",
            elapsed_origin=time.perf_counter(),
        )

    assert result["requests_attempted"] == 8
    assert result["requests_wholly_inside_observed_running_polls"] > 0
    assert set(result["endpoints_wholly_inside_observed_running_polls"]) == set(benchmark.ENDPOINTS)
    assert result["status_poll_uncertainty_s"]["end_transition_bracket"] > 0
    assert (
        result["observed_running_window_s"]["end_last_running_poll"]
        < result["status_observations"][-1]["observed_at_s"]
    )
