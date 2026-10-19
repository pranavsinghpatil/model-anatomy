"""Focused resource-bound checks for metadata-driven engine inspection."""

import io
import json
from types import SimpleNamespace

import pytest
import torch

from model_anatomy.src import limits, registry
from model_anatomy.src.artifacts import (
    CheckpointMetadata,
    MetadataClient,
    ModelReference,
    parse_safetensors_header,
)
from model_anatomy.src.errors import ArtifactError, UnsupportedModelError
from model_anatomy.src.facts import inspect_native_model
from model_anatomy.src.profile import native_dimensions


class Response:
    def __init__(self, body):
        self.raw = self
        self._body = io.BytesIO(body)
        self.status_code = 200
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def read(self, size=-1, decode_content=False):
        return self._body.read(size)


class Session:
    def __init__(self, body):
        self.body = body

    def get(self, *args, **kwargs):
        return Response(self.body)


def tiny_config(**overrides):
    return {
        "model_type": "gpt2",
        "vocab_size": 32,
        "n_positions": 16,
        "n_embd": 8,
        "n_head": 2,
        "n_layer": 2,
        **overrides,
    }


def test_embedded_layer_count_rejected_before_autoconfig(monkeypatch):
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("AutoConfig must not run for invalid metadata")

    monkeypatch.setattr(registry.AutoConfig, "for_model", unexpected)
    config = tiny_config(encoder={"stages": [{"config": {"num_decoder_layers": 513}}]})
    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config(config)
    assert called is False


@pytest.mark.parametrize("model_type", ["distilbert", "xlm"])
@pytest.mark.parametrize("value", [513, 0, -1, True, "6", [6]])
def test_native_layer_aliases_are_bounded_before_autoconfig(monkeypatch, model_type, value):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for an invalid native layer alias"),
    )
    error = UnsupportedModelError if type(value) is int and value > 512 else ArtifactError
    with pytest.raises(error, match="layer count|dimension"):
        registry.load_native_config({"model_type": model_type, "n_layers": value})


def test_nested_native_layer_alias_is_bounded_before_autoconfig(monkeypatch):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for an excessive nested layer alias"),
    )
    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config(tiny_config(text_config={"model_type": "distilbert", "n_layers": 513}))


def test_native_layer_alias_retains_valid_nondefault_count():
    config = registry.load_native_config({"model_type": "distilbert", "n_layers": 2})
    assert config.n_layers == config.num_hidden_layers == 2


@pytest.mark.parametrize("value", [True, False, "2", 0, -1])
def test_embedded_layer_type_rejected_before_autoconfig(monkeypatch, value):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for invalid metadata"),
    )
    with pytest.raises(ArtifactError, match="dimension"):
        registry.load_native_config(tiny_config(components=[{"encoder_layers": value}]))


def test_nested_config_depth_and_node_budgets_precede_autoconfig(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CONFIG_NESTING_DEPTH", 2)
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for excessive nesting"),
    )
    config = tiny_config(nested={"level_1": [{"level_2": {"level_3": {}}}]})
    with pytest.raises(ArtifactError, match="depth limit"):
        registry.load_native_config(config)
    monkeypatch.setattr(limits, "MAX_CONFIG_NESTING_DEPTH", 64)
    monkeypatch.setattr(limits, "MAX_CONFIG_CONTAINER_NODES", 2)
    config = tiny_config(nested=[{}, {}, {}])
    with pytest.raises(ArtifactError, match="node limit"):
        registry.load_native_config(config)


@pytest.mark.parametrize("model_type", ["swin", "swinv2", "donut-swin", "maskformer-swin"])
def test_swin_stage_depth_sum_rejected_before_autoconfig(monkeypatch, model_type):
    monkeypatch.setattr(limits, "MAX_MODEL_LAYERS", 8)
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for an excessive stage schedule"),
    )
    config = {"model_type": model_type, "depths": [1, 1_000_000_000]}
    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config(config)


@pytest.mark.parametrize(
    ("model_type", "parent_fields", "inferred_stage_type"),
    [
        ("mask2former", {}, "swin"),
        ("maskformer", {}, "swin"),
        ("oneformer", {}, "swin"),
        ("dpt", {"is_hybrid": True}, "bit"),
        ("vit-hybrid", {}, "bit"),
    ],
)
def test_implicit_embedded_backbone_schedule_rejected_before_autoconfig(
    monkeypatch, model_type, parent_fields, inferred_stage_type
):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for an excessive backbone schedule"),
    )
    config = {
        "model_type": model_type,
        **parent_fields,
        "backbone_config": {"depths": [1_000_000_000]},
    }
    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config(config)
    assert registry._IMPLICIT_BACKBONE_STAGE_TYPES[model_type] == inferred_stage_type


@pytest.mark.parametrize("depths", [[True], [1, False], [0], [-1], [1.5], "12", []])
def test_swin_stage_depths_require_a_nonempty_list_of_positive_strict_ints(monkeypatch, depths):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for malformed stage depths"),
    )
    with pytest.raises(ArtifactError, match="layer schedule"):
        registry.load_native_config({"model_type": "swin", "depths": depths})


def test_nested_swin_config_depths_are_validated_before_autoconfig(monkeypatch):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for an invalid nested schedule"),
    )
    config = tiny_config(backbone_config={"components": [{"model_type": "swin", "depths": [513]}]})
    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config(config)


def test_valid_swin_stage_depths_and_unrelated_depths_remain_accepted(monkeypatch):
    config = registry.load_native_config({"model_type": "swin", "depths": [2, 2, 6, 2]})
    assert config.depths == [2, 2, 6, 2]

    # A generic field named depths is not necessarily a layer schedule.
    config = registry.load_native_config(tiny_config(auxiliary={"depths": [1_000_000_000]}))
    assert config.model_type == "gpt2"


def test_valid_segformer_stage_schedules_are_accepted():
    config = registry.load_native_config(
        {
            "model_type": "segformer",
            "depths": [1, 2, 1, 1],
            "num_attention_heads": [1, 2, 5, 8],
        }
    )

    assert config.depths == [1, 2, 1, 1]
    assert config.num_attention_heads == [1, 2, 5, 8]


def test_segformer_profile_layer_count_uses_sum_of_stage_depths():
    dimensions = native_dimensions(
        SimpleNamespace(model_type="segformer", num_hidden_layers=None, num_layers=None, depths=[1, 2, 1, 1])
    )

    assert dimensions["layers"] == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_attention_heads", 4),
        ("num_attention_heads", []),
        ("num_attention_heads", [1, 2, 5]),
        ("num_attention_heads", [1, 2, 5, 0]),
        ("num_attention_heads", [1, 2, 5, True]),
        ("depths", []),
        ("depths", [1, 2, 1]),
    ],
)
def test_malformed_segformer_stage_schedules_are_rejected_before_autoconfig(monkeypatch, field, value):
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for malformed SegFormer schedules"),
    )
    with pytest.raises(ArtifactError, match="schedule"):
        registry.load_native_config({"model_type": "segformer", field: value})


def test_excessive_segformer_total_depth_is_rejected_before_autoconfig(monkeypatch):
    monkeypatch.setattr(limits, "MAX_MODEL_LAYERS", 3)
    monkeypatch.setattr(
        registry.AutoConfig,
        "for_model",
        lambda *args, **kwargs: pytest.fail("AutoConfig must not run for excessive SegFormer depth"),
    )
    with pytest.raises(UnsupportedModelError, match="layer count"):
        registry.load_native_config({"model_type": "segformer", "depths": [1, 1, 2, 1]})


def test_text_family_still_rejects_head_schedules():
    with pytest.raises(ArtifactError, match="num_attention_heads"):
        registry.load_native_config(tiny_config(num_attention_heads=[2, 2]))
    with pytest.raises(ArtifactError, match="num_attention_heads"):
        registry.load_native_config(tiny_config(num_attention_heads=None))


def test_foreign_config_still_obeys_generic_container_budgets(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CONFIG_NESTING_DEPTH", 1)

    with pytest.raises(ArtifactError, match="depth limit"):
        registry.load_native_config(
            {"_class_name": "UNet2DConditionModel", "num_attention_heads": None, "nested": {"value": {}}}
        )


def test_shard_tensor_budget_aggregates_and_stops_before_later_shards(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 5)
    reference = ModelReference("example/model")
    revision = "a" * 40
    client = MetadataClient()
    client.repository_files[(reference.model_id, revision)] = {
        "model.safetensors.index.json",
        "a.safetensors",
        "b.safetensors",
        "c.safetensors",
    }
    client.json = lambda *args, **kwargs: {
        "weight_map": {
            "one": "a.safetensors",
            "two": "a.safetensors",
            "three": "a.safetensors",
            "four": "b.safetensors",
            "five": "c.safetensors",
        }
    }
    shard = lambda *names: SimpleNamespace(
        metadata={"format": "pt"}, tensors={name: object() for name in names}
    )
    responses = {
        "a.safetensors": shard("one", "two", "three", "unexpected_a"),
        # This extra would be the sixth record; the c shard must not be read.
        "b.safetensors": shard("four", "unexpected_b"),
        "c.safetensors": shard("five"),
    }
    read_shards = []

    def read_shard(url):
        filename = url.rsplit("/", 1)[-1]
        read_shards.append(filename)
        return responses[filename]

    client.safetensors = read_shard
    with pytest.raises(ArtifactError, match="tensor record count"):
        client.checkpoint(reference, revision)
    assert read_shards == ["a.safetensors", "b.safetensors"]


def test_malformed_deep_json_is_artifact_error():
    body = (b'{"x":' * 1_100) + b"0" + (b"}" * 1_100)
    client = MetadataClient(Session(body))
    with pytest.raises(ArtifactError, match="Invalid JSON metadata"):
        client.json("https://huggingface.co/config.json")


def test_repository_file_budget_rejected_before_file_set(monkeypatch):
    monkeypatch.setattr(limits, "MAX_REPOSITORY_FILES", 2)
    siblings = [{"rfilename": f"file-{index}"} for index in range(3)]
    body = json.dumps({"sha": "a" * 40, "siblings": siblings}).encode()
    client = MetadataClient(Session(body))
    with pytest.raises(ArtifactError, match="file record count"):
        client.resolve(ModelReference("example/model"))
    assert client.repository_files == {}


def test_checkpoint_tensor_budget_rejected_during_header_parse(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 1)
    header = json.dumps(
        {
            "first": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
            "second": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
        }
    ).encode()
    with pytest.raises(ArtifactError, match="tensor record count"):
        parse_safetensors_header(header, 8 + len(header))


def test_checkpoint_index_tensor_budget_precedes_shard_reads(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CHECKPOINT_TENSORS", 1)
    ref = ModelReference("example/model")
    client = MetadataClient()
    sha = "a" * 40
    client.repository_files[(ref.model_id, sha)] = {
        "model.safetensors.index.json",
        "one.safetensors",
    }
    client.json = lambda *args, **kwargs: {"weight_map": {"one": "one.safetensors", "two": "one.safetensors"}}
    client.safetensors = lambda *args, **kwargs: pytest.fail("Shard reads must not start")
    with pytest.raises(ArtifactError, match="tensor record count"):
        client.checkpoint(ref, sha)


def test_native_module_and_parameter_record_budgets(monkeypatch):
    empty_checkpoint = CheckpointMetadata({}, {}, {})
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    monkeypatch.setattr(limits, "MAX_NATIVE_MODULE_RECORDS", 1)
    with pytest.raises(ArtifactError, match="module record count"):
        inspect_native_model(model, empty_checkpoint)

    monkeypatch.setattr(limits, "MAX_NATIVE_MODULE_RECORDS", 20_000)
    monkeypatch.setattr(limits, "MAX_NATIVE_PARAMETER_RECORDS", 1)
    with pytest.raises(ArtifactError, match="parameter record count"):
        inspect_native_model(torch.nn.Linear(2, 2), empty_checkpoint)
