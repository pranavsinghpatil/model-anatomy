"""Regression checks for artifact inspection, graph integrity, and tied parameters."""

import io
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from model_anatomy.src.artifacts import (
    DTYPE_BYTES,
    CheckpointMetadata,
    MetadataClient,
    parse_reference,
    parse_safetensors_header,
)
from model_anatomy.src.errors import AnatomyError, ArtifactError, UnsupportedModelError
from model_anatomy.src.facts import inspect_native_model
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.registry import instantiate_native, load_native_config
from model_anatomy.src.render import mermaid

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
CONFIG = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "vocab_size": 32,
    "n_positions": 16,
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 2,
}


def checkpoint_for(model):
    offset = 0
    header = {"__metadata__": {"format": "pt"}}
    for name, parameter in model.named_parameters():
        size = parameter.numel() * DTYPE_BYTES["F32"]
        header[name] = {
            "dtype": "F32",
            "shape": list(parameter.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    shard = parse_safetensors_header(encoded, 8 + len(encoded) + offset)
    return CheckpointMetadata(
        shard.tensors,
        {"model.safetensors": shard},
        {name: "model.safetensors" for name in shard.tensors},
    )


@pytest.fixture(scope="module")
def tiny_model():
    model, _ = instantiate_native(load_native_config(CONFIG))
    return model


class FixtureClient(MetadataClient):
    def __init__(self, checkpoint, config=CONFIG):
        super().__init__()
        self.fixture = checkpoint
        self.config = config

    def resolve(self, reference):
        self.repository_files[(reference.model_id, SHA)] = {"config.json", "model.safetensors"}
        return SHA

    def json(self, url, max_bytes=1024 * 1024):
        self.audit.append({"url": url, "bytes_read": 1, "status": 200})
        return dict(self.config)

    def checkpoint(self, reference, revision):
        assert revision == SHA
        return self.fixture


def test_pipeline_emits_valid_inspectable_report_without_weights(tiny_model):
    stages = []
    ir = reconstruct(
        "example/model", client=FixtureClient(checkpoint_for(tiny_model)), on_progress=stages.append
    )
    assert ir.metadata["revision"] == SHA
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    assert stages == [
        "resolving", "reading_metadata", "building_structure", "analyzing_components", "validating"
    ]
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
    assert "flowchart TD" in mermaid(ir.root_graph)


def test_native_vision_metadata_and_component_views_keep_ownership():
    from model_anatomy.service.views import node_view

    config = {
        "model_type": "vit",
        "architectures": ["ViTForImageClassification"],
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_attention_heads": 3,
        "num_hidden_layers": 2,
        "image_size": 32,
        "patch_size": 16,
        "num_channels": 3,
        "num_labels": 10,
    }
    model, _ = instantiate_native(load_native_config(config))
    ir = reconstruct("example/vision", client=FixtureClient(checkpoint_for(model), config))
    assert ir.architecture == "ViTForImageClassification"
    assert ir.metadata["inputs"]["input_specs"]["pixel_values"]["shape"] == ["B", 3, 32, 32]
    assert ir.metadata["dimensions"]["vocabulary"] is None
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    component = node_view(ir, ir.root_graph.id, "classifier")
    assert component["node"]["kind"] == "classifier"
    assert component["node"]["parameter_count"] == 250
    assert [tensor["name"] for tensor in component["tensors"]] == [
        "classifier.bias", "classifier.weight",
    ]
    assert all(tensor["parameter_refs"] for tensor in component["tensors"])
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()


def test_pinned_revision_is_used_and_mutable_revisions_are_rejected(tiny_model):
    client = FixtureClient(checkpoint_for(tiny_model))
    client.resolve(parse_reference("example/model"))
    ir = reconstruct("example/model", client=client, revision=SHA)
    assert ir.metadata["revision"] == SHA
    with pytest.raises(ArtifactError, match="immutable"):
        reconstruct("example/model", client=client, revision="main")


def test_tied_embeddings_are_counted_once_and_keep_aliases(tiny_model):
    raw = inspect_native_model(tiny_model, checkpoint_for(tiny_model))
    embedding = raw["parameters"]["transformer.wte.weight"]
    head = raw["parameters"]["lm_head.weight"]
    assert embedding["canonical_parameter"] == head["canonical_parameter"]
    assert embedding["aliases"] == ["transformer.wte.weight", "lm_head.weight"]
    assert head["checkpoint_matched"] is True
    assert raw["logical_parameter_count"] - raw["unique_parameter_count"] == 32 * 8
    assert all(p.device.type == "meta" for p in tiny_model.parameters())


def test_checkpoint_shape_mismatch_fails_honestly(tiny_model):
    checkpoint = checkpoint_for(tiny_model)
    tensor = checkpoint.tensors["transformer.wte.weight"]
    checkpoint.tensors[tensor.name] = replace(tensor, shape=(1, 1))
    with pytest.raises(ArtifactError, match="shape mismatch"):
        inspect_native_model(tiny_model, checkpoint)


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "custom.Model"}},
        {"backbone": "other/model"},
        {"use_pretrained_backbone": True},
        {"nested": [{"auto_map": {"AutoModel": "custom.Model"}}]},
    ],
)
def test_custom_code_or_implicit_weight_loading_is_rejected(change):
    with pytest.raises(UnsupportedModelError):
        load_native_config({**CONFIG, **change})


@pytest.mark.parametrize("value", [0, -1, True, "12"])
def test_invalid_model_dimensions_are_rejected(value):
    with pytest.raises(ArtifactError):
        load_native_config({**CONFIG, "n_embd": value})


def test_excessive_layer_count_is_rejected_before_construction():
    with pytest.raises(UnsupportedModelError, match="layer count"):
        load_native_config({**CONFIG, "n_layer": 10000})


@pytest.mark.parametrize(
    "reference",
    ["http://huggingface.co/a/b", "https://localhost/a/b", "https://huggingface.co/a/b?token=x", "../b"],
)
def test_non_model_urls_are_rejected(reference):
    with pytest.raises(ArtifactError):
        parse_reference(reference)


def test_file_reference_preserves_exact_revision():
    ref = parse_reference(f"https://huggingface.co/example/model/blob/{SHA}/model.safetensors")
    assert ref.model_id == "example/model"
    assert ref.revision == SHA
    assert ref.explicit_filename


def test_duplicate_tensor_json_and_invalid_offsets_are_rejected():
    with pytest.raises(ArtifactError, match="Duplicate"):
        parse_safetensors_header(b'{"x":{},"x":{}}', 25)
    header = json.dumps({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}).encode()
    with pytest.raises(ArtifactError, match="offsets"):
        parse_safetensors_header(header, 8 + len(header) + 4)


class Response:
    def __init__(self, body, status, headers):
        self.raw = io.BytesIO(body)
        self.status_code = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("HTTP failure", response=self)


class Session:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def test_range_not_honored_never_reads_weight_body():
    response = Response(b"weight payload must remain unread", 200, {})
    client = MetadataClient(Session(response))
    with pytest.raises(ArtifactError, match="body was not read"):
        client.range("https://huggingface.co/example/model/resolve/main/model.safetensors", 0, 7)
    assert response.raw.tell() == 0


def test_excessive_header_is_rejected_after_prefix_only():
    response = Response(struct.pack("<Q", 100 * 1024 * 1024), 206, {"Content-Range": "bytes 0-7/200000000"})
    client = MetadataClient(Session(response))
    with pytest.raises(ArtifactError, match="excessive"):
        client.safetensors("https://huggingface.co/example/model/resolve/main/model.safetensors")
    assert response.raw.tell() == 8


def test_unavailable_repository_is_not_reported_as_success():
    response = Response(b"", 404, {})
    with pytest.raises(requests.HTTPError):
        MetadataClient(Session(response)).resolve(parse_reference("example/missing"))


def test_existing_gpt2_report_roundtrip_and_unknown_shapes_are_preserved():
    ir = ModelIR.from_json((ROOT / "output/gpt2/model.json").read_text(encoding="utf-8"))
    assert ir.metadata["unique_parameter_count"] == 124439808
    assert len(ir.tensors) == 160
    for graph in ir.root_graph.walk():
        for node in graph.nodes:
            for tensor in node.inputs + node.outputs:
                tensor.shape = None
        for item in graph.edges:
            if item.tensor:
                item.tensor.shape = None
    restored = ModelIR.from_json(ir.to_json())
    assert restored.root_graph.edges[0].tensor.shape is None
    assert "unknown shape" in mermaid(restored.root_graph)


def test_dangling_edges_and_duplicate_node_ids_are_rejected():
    ir = ModelIR.from_json((ROOT / "output/gpt2/model.json").read_text(encoding="utf-8"))
    ir.root_graph.edges[0].source = "does-not-exist"
    with pytest.raises(AnatomyError, match="Dangling"):
        ir.to_json()
    ir = ModelIR.from_json((ROOT / "output/gpt2/model.json").read_text(encoding="utf-8"))
    ir.root_graph.nodes.append(ir.root_graph.nodes[0])
    with pytest.raises(AnatomyError, match="Duplicate node"):
        ir.to_json()
