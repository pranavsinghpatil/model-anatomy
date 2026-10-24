"""Exact-pin publisher topology remains separate from checkpoint verification."""

import copy
import json
from pathlib import Path

import pytest

from model_anatomy.service.jobs import ir_summary
from model_anatomy.src import limits
from model_anatomy.src.artifacts import MetadataClient
from model_anatomy.src.errors import UnsupportedModelError
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.publisher_topology import MODEL_ID, REQUIRED_FILES, REVISION

CONFIG = json.loads((Path(__file__).parent / "fixtures" / "deepseek_v41_config.json").read_text())


class IndexedClient(MetadataClient):
    def __init__(self, *, model_id=MODEL_ID, revision=REVISION, config=None, files=None):
        super().__init__()
        self.model_id = model_id
        self.revision = revision
        self.config = config if config is not None else CONFIG
        self.files = set(files if files is not None else REQUIRED_FILES | {"one.safetensors"})

    def resolve(self, reference):
        self.repository_files[(self.model_id, self.revision)] = self.files
        return self.revision

    def json(self, url, max_bytes=1024 * 1024):
        self.audit.append({"url": url, "bytes_read": 100, "status": 200})
        if url.endswith(".index.json"):
            return {"weight_map": {name: "one.safetensors" for name in ("a", "b", "c")}}
        return self.config

    checkpoint = MetadataClient.checkpoint

    def safetensors(self, url):
        pytest.fail("Oversized index must not read checkpoint shard headers")


def inspect(client, *, allow=True):
    return reconstruct(client.model_id, client=client, allow_artifact_inspection=allow)


def test_reviewed_deepseek_pin_yields_source_graph_with_honest_partial_status(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 2)
    ir = inspect(IndexedClient())
    report = ir_summary(ir.to_dict())

    assert report["result_status"] == "partial"
    assert report["quality"]["mode"] == "publisher_reference_topology"
    assert report["root_graph_id"] == "publisher_reference_topology"
    assert report["parameter_count"] is None
    assert ir.metadata["source_topology"]["checkpoint_headers_verified"] is False
    assert ir.metadata["artifacts"]["declared_checkpoint_index"]["declared_tensor_names"] == 3
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.tensors == {}
    assert len([node for node in ir.root_graph.nodes if node.id.startswith("layer_")]) == 40
    assert len([node for node in ir.root_graph.nodes if node.id.startswith("mtp_")]) == 3
    layer_14 = next(node for node in ir.root_graph.nodes if node.id == "layer_14")
    assert {node.id for node in layer_14.child_graph.nodes} >= {"engram", "attn", "ffn"}
    moe = next(node for node in layer_14.child_graph.nodes if node.id == "ffn")
    assert next(node for node in moe.child_graph.nodes if node.id == "experts").repeat == 384
    vision = next(node for node in ir.root_graph.nodes if node.id == "vision")
    assert next(node for node in vision.child_graph.nodes if node.id == "blocks").repeat == 32
    assert all(
        edge.relation == "reference_order" and edge.tensor is None
        for graph in ir.root_graph.walk()
        for edge in graph.edges
    )
    assert all(
        node.parameter_count == 0 and node.metadata["parameter_count_status"] == "unverified"
        for graph in ir.root_graph.walk()
        for node in graph.nodes
    )


@pytest.mark.parametrize("change", ["model", "revision", "config", "source_file"])
def test_source_graph_is_not_claimed_for_unreviewed_identity(monkeypatch, change):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 2)
    client = IndexedClient()
    if change == "model":
        client.model_id = "other/deepseek"
    elif change == "revision":
        client.revision = "a" * 40
    elif change == "config":
        client.config = copy.deepcopy(CONFIG)
        client.config["text_config"]["num_hidden_layers"] = 39
    else:
        client.files.remove("inference/model.py")
    ir = inspect(client)
    assert ir.metadata["inspection"]["kind"] == "artifact_inspection"
    assert ir.root_graph.nodes == []


def test_strict_cli_does_not_use_reviewed_fallback(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 2)
    with pytest.raises(UnsupportedModelError, match="No native Transformers config"):
        inspect(IndexedClient(), allow=False)
