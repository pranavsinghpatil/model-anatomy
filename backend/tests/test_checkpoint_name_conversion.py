"""Checkpoint key normalization follows the installed native Transformers loader."""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest
from transformers.core_model_loading import WeightRenaming
from transformers.models.bert.modeling_bert import BertForMaskedLM

from model_anatomy.src.artifacts import DTYPE_BYTES, CheckpointMetadata, parse_safetensors_header
from model_anatomy.src.errors import ArtifactError
from model_anatomy.src.facts import inspect_native_model
from model_anatomy.src.registry import instantiate_native, load_native_config

BERT_CONFIG = {
    "model_type": "bert",
    "architectures": ["BertForMaskedLM"],
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 16,
    "type_vocab_size": 2,
}
LEGACY_SOURCE = "native Transformers checkpoint loading and conversion rules"


@pytest.fixture(scope="module")
def bert_model():
    model, _ = instantiate_native(load_native_config(BERT_CONFIG))
    assert isinstance(model, BertForMaskedLM)
    return model


def _header_checkpoint(tensors: dict[str, tuple[int, ...]]) -> CheckpointMetadata:
    offset = 0
    header = {"__metadata__": {"format": "pt"}}
    for name, shape in tensors.items():
        size = 1
        for dimension in shape:
            size *= dimension
        byte_size = size * DTYPE_BYTES["F32"]
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size
    encoded = json.dumps(header).encode()
    shard = parse_safetensors_header(encoded, 8 + len(encoded) + offset)
    return CheckpointMetadata(
        shard.tensors,
        {"model.safetensors": shard},
        {name: "model.safetensors" for name in shard.tensors},
    )


def _legacy_bert_names(model) -> dict[str, tuple[int, ...]]:
    tensors = {}
    for name, parameter in model.named_parameters():
        legacy = name
        if name.endswith(".LayerNorm.weight"):
            legacy = name.removesuffix("weight") + "gamma"
        elif name.endswith(".LayerNorm.bias"):
            legacy = name.removesuffix("bias") + "beta"
        tensors[legacy] = tuple(parameter.shape)
    return tensors


def test_bert_legacy_layer_norm_keys_are_fully_owned_with_conversion_provenance(bert_model, monkeypatch):
    checkpoint = _header_checkpoint(_legacy_bert_names(bert_model))

    def reject_model_override(_key):
        raise AssertionError("Model-specific checkpoint conversion hooks are not trusted")

    monkeypatch.setattr(
        type(bert_model), "_fix_state_dict_key_on_load", staticmethod(reject_model_override), raising=False
    )
    raw = inspect_native_model(bert_model, checkpoint)

    assert all(parameter["checkpoint_matched"] for parameter in raw["parameters"].values())
    unique_parameters = {item["canonical_parameter"] for item in raw["parameters"].values()}
    assert len(raw["checkpoint_name_mapping"]) == len(unique_parameters)
    conversions = raw["checkpoint_name_conversions"]
    legacy_names = {
        name for name in checkpoint.tensors if name.endswith((".LayerNorm.gamma", ".LayerNorm.beta"))
    }
    assert set(conversions) == legacy_names
    for source_name in legacy_names:
        converted = conversions[source_name]
        module_path, legacy_name = source_name.rsplit(".", 1)
        expected_name = module_path + "." + ("weight" if legacy_name == "gamma" else "bias")
        assert converted["target"] == expected_name
        assert converted["source"] == LEGACY_SOURCE
        rule = next(
            item
            for item in converted["steps"]
            if item["rule"] == "transformers.core_model_loading.WeightRenaming"
        )
        expected_target = "LayerNorm.weight" if legacy_name == "gamma" else "LayerNorm.bias"
        assert rule["source_patterns"] == [f"LayerNorm.{legacy_name}"]
        assert rule["target_patterns"] == [expected_target]
        assert rule["source_sha256"] == hashlib.sha256(
            inspect.getsource(WeightRenaming).encode()
        ).hexdigest()
        assert raw["checkpoint_name_mapping"][expected_name] == source_name
        assert raw["parameters"][expected_name]["tensor_name"] == source_name


def test_legacy_layer_norm_shape_mismatch_is_rejected(bert_model):
    name = "bert.embeddings.LayerNorm.gamma"
    checkpoint = _header_checkpoint({name: (1, 1)})

    with pytest.raises(ArtifactError, match="shape mismatch"):
        inspect_native_model(bert_model, checkpoint)


def test_legacy_and_modern_names_for_same_parameter_are_rejected(bert_model):
    modern_name = "bert.embeddings.LayerNorm.weight"
    legacy_name = "bert.embeddings.LayerNorm.gamma"
    checkpoint = _header_checkpoint({modern_name: (16,), legacy_name: (16,)})

    with pytest.raises(ArtifactError, match="Ambiguous checkpoint naming"):
        inspect_native_model(bert_model, checkpoint)


def test_unrelated_gamma_and_beta_keys_remain_unmapped(bert_model):
    checkpoint = _header_checkpoint({"bert.encoder.gamma": (16,), "bert.encoder.beta": (16,)})

    raw = inspect_native_model(bert_model, checkpoint)

    assert raw["checkpoint_name_conversions"] == {}
    assert raw["checkpoint_name_mapping"] == {}
    assert set(raw["checkpoint_extras"]) == {"bert.encoder.gamma", "bert.encoder.beta"}
    assert all(extra["kind"] == "unmapped" for extra in raw["checkpoint_extras"].values())
