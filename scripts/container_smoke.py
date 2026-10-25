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


def check_architecture(client, path, detail, report):
    """Check registered layers and checkpoint evidence through public endpoints."""
    checks = detail["quality"]["architecture_checks"]
    assert checks == dict.fromkeys(
        ("module_inventory", "parameter_mapping", "native_architecture", "module_topology"), True
    )
    assert detail["quality"]["mode"] == "static_module_architecture"
    assert report["metadata"]["analysis_report"]["forward_executed"] is False
    index = require(client.get(f"{path}/graph-index")).json()
    assert index["root_graph_id"] == detail["root_graph_id"]
    graphs = []
    pending = [report["root_graph"]]
    while pending:
        graph = pending.pop()
        graphs.append(graph)
        pending.extend(node["child_graph"] for node in graph["nodes"] if node.get("child_graph"))
    assert {graph["id"] for graph in graphs} == {graph["id"] for graph in index["graphs"]}
    layers = detail["dimensions"]["layers"]
    assert type(layers) is int and layers > 0
    container = next(
        node for graph in graphs for node in graph["nodes"]
        if node["metadata"].get("instance_count") == layers and node.get("child_graph")
    )
    instances = container["metadata"]["instance_paths"]
    assert len(instances) == len(set(instances)) == layers
    child = require(client.get(f"{path}/graphs/{container['child_graph']['id']}")).json()
    assert {node["module_path"] for node in child["nodes"] if node["module_path"]} == set(instances)
    embedding = next(node for node in index["nodes"] if node["kind"] == "embedding")
    inspection = require(client.get(f"{path}/nodes/{embedding['graph_id']}/{embedding['id']}")).json()
    assert inspection["tensor_count"] > 0 and inspection["evidence"]
    assert inspection["node"]["child_graph_id"] is None
    assert all(tensor["shape"] and tensor["dtype"] for tensor in inspection["tensors"])


def run():
    result = {"model_id": MODEL, "revision": REVISION, "analyses": []}
    first_analysis_path = None
    with httpx.Client(base_url=base_url, trust_env=False, timeout=15) as client:
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
            assert detail["partial_reasons"] == []
            assert detail["revision"] == REVISION
            assert detail["parameter_count"] == 134515008
            report = require(client.get(f"{path}/exports/json")).json()
            assert report["metadata"]["artifacts"]["weight_payload_bytes_read"] == 0
            check_architecture(client, path, detail, report)
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
        "nested graph with all 30 registered layer instances",
        "tensor evidence",
        "exports",
        "visitor isolation",
        "quotas",
        "AI outage",
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
