"""Real HTTP acceptance for the CPU images, without cloud credentials or AI mocks."""

import json
import time

import httpx

MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_URL = f"https://huggingface.co/{MODEL}/blob/{REVISION}/model.safetensors"


def require(response, status=200):
    if response.status_code != status:
        raise RuntimeError(f"Expected HTTP {status}; received {response.status_code}")
    return response


def run():
    result = {"model_id": MODEL, "revision": REVISION, "analyses": []}
    first_analysis_path = None
    with httpx.Client(base_url="http://127.0.0.1:8001", trust_env=False, timeout=15) as client:
        for _ in range(60):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Container services did not become ready")
        health = require(client.get("/health")).json()
        assert health["analysis_available"] and not health["ai_available"]
        for _ in range(2):
            start = time.perf_counter()
            job = require(client.post("/api/v1/analyses", json={"model": MODEL_URL}), 202).json()
            path = f"/api/v1/analyses/{job['id']}"
            if first_analysis_path is None:
                first_analysis_path = path
            while time.perf_counter() - start < 150:
                detail = require(client.get(path)).json()
                if detail["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Container inspection exceeded its deadline")
            assert detail["status"] == "completed", detail.get("error", {})
            assert detail["warnings"] == []
            assert detail["revision"] == REVISION
            assert detail["parameter_count"] == 134515008
            report = require(client.get(f"{path}/exports/json")).json()
            assert report["metadata"]["artifacts"]["weight_payload_bytes_read"] == 0
            graph = require(client.get(f"{path}/graphs/{detail['root_graph_id']}")).json()
            repeated = next(n for n in graph["nodes"] if n["repeat"] == 30)
            child = require(client.get(f"{path}/graphs/{repeated['child_graph_id']}")).json()
            assert child["nodes"]
            node = next(n for n in graph["nodes"] if n["kind"] == "embedding")
            inspection = require(client.get(f"{path}/nodes/{graph['id']}/{node['id']}")).json()
            assert inspection["tensor_count"] > 0 and inspection["evidence"]
            archive = require(client.get(f"{path}/exports/zip"))
            assert archive.content.startswith(b"PK\x03\x04")
            with httpx.Client(base_url=str(client.base_url), trust_env=False) as other:
                require(other.get(path), 404)
            result["analyses"].append(
                {
                    "status": detail["status"],
                    "cached": detail["cached"],
                    "seconds": time.perf_counter() - start,
                }
            )
        assert not result["analyses"][0]["cached"] and result["analyses"][1]["cached"]
        require(client.post("/api/v1/analyses", json={"model": MODEL_URL}), 429)
        require(
            client.post(f"{first_analysis_path}/questions", json={"question": "Explain attention"}),
            503,
        )
    result["checks"] = [
        "isolated CPU worker",
        "pinned primary model",
        "no weight payload",
        "immutable cache",
        "nested graph",
        "tensor evidence",
        "exports",
        "visitor isolation",
        "quotas",
        "AI outage",
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
