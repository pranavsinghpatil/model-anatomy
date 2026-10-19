"""Native AutoModel routing from Transformers' upstream mapping-name metadata."""

from __future__ import annotations

import pytest

from model_anatomy.src.errors import ArtifactError, UnsupportedModelError
from model_anatomy.src.registry import instantiate_native, load_native_config, select_auto_class


@pytest.mark.parametrize(
    ("data", "auto_name", "model_name"),
    [
        (
            {
                "model_type": "t5",
                "architectures": ["T5ForConditionalGeneration"],
                "vocab_size": 64,
                "d_model": 16,
                "d_kv": 8,
                "d_ff": 32,
                "num_layers": 1,
                "num_decoder_layers": 1,
                "num_heads": 2,
                "decoder_start_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 0,
            },
            "AutoModelForSeq2SeqLM",
            "T5ForConditionalGeneration",
        ),
        (
            {
                "model_type": "swin",
                "architectures": ["SwinForImageClassification"],
                "image_size": 32,
                "patch_size": 4,
                "num_channels": 3,
                "embed_dim": 8,
                "depths": [1, 1, 1, 1],
                "num_heads": [1, 2, 4, 8],
                "window_size": 4,
                "num_labels": 2,
            },
            "AutoModelForImageClassification",
            "SwinForImageClassification",
        ),
        (
            {
                "model_type": "segformer",
                "architectures": ["SegformerForImageClassification"],
                "depths": [1, 1, 1, 1],
                "num_attention_heads": [1, 2, 5, 8],
                "hidden_sizes": [8, 10, 10, 16],
                "num_labels": 2,
            },
            "AutoModelForImageClassification",
            "SegformerForImageClassification",
        ),
        (
            {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
                "vocab_size": 32,
                "num_mel_bins": 16,
                "d_model": 16,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "encoder_attention_heads": 2,
                "decoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "decoder_ffn_dim": 32,
                "max_source_positions": 32,
                "max_target_positions": 32,
                "decoder_start_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            },
            "AutoModelForSpeechSeq2Seq",
            "WhisperForConditionalGeneration",
        ),
        (
            {
                "model_type": "clip",
                "architectures": ["CLIPModel"],
                "projection_dim": 8,
                "text_config": {
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "max_position_embeddings": 16,
                },
                "vision_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "image_size": 32,
                    "patch_size": 16,
                },
            },
            "AutoModel",
            "CLIPModel",
        ),
    ],
    ids=["text-encoder-decoder", "swin-image", "segformer-image", "whisper-audio", "clip-multimodal"],
)
def test_declared_native_architecture_routes_and_constructs_on_meta(data, auto_name, model_name):
    config = load_native_config(data)

    auto_class, warnings = select_auto_class(config)
    assert auto_class.__name__ == auto_name
    assert warnings == []

    model, selection = instantiate_native(config)
    assert type(model).__name__ == model_name
    assert selection["auto_class"] == auto_name
    assert selection["device"] == "meta"
    assert all(tensor.device.type == "meta" for tensor in (*model.parameters(), *model.buffers()))


def test_encoder_decoder_hint_routes_when_architecture_is_not_declared():
    config = load_native_config(
        {
            "model_type": "t5",
            "vocab_size": 64,
            "d_model": 16,
            "d_kv": 8,
            "d_ff": 32,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "num_heads": 2,
        }
    )

    auto_class, warnings = select_auto_class(config)

    assert auto_class.__name__ == "AutoModelForSeq2SeqLM"
    assert warnings == [
        "No architecture declared; selected AutoModelForSeq2SeqLM from native config task metadata"
    ]


def test_unavailable_declared_architecture_never_falls_back_to_backbone():
    config = load_native_config({"model_type": "bert", "architectures": ["BertForUnavailableTask"]})

    with pytest.raises(UnsupportedModelError, match="No native AutoModel mapping for declared architecture"):
        select_auto_class(config)


@pytest.mark.parametrize("architectures", ["BertModel", [None], [""]])
def test_malformed_architecture_declarations_are_rejected(architectures):
    config = load_native_config({"model_type": "bert", "architectures": architectures})

    with pytest.raises(ArtifactError, match="Invalid native configuration architectures"):
        select_auto_class(config)


def test_selection_does_not_resolve_lazy_model_implementations(monkeypatch):
    from transformers.models.auto.auto_factory import _LazyAutoMapping

    config = load_native_config({"model_type": "bert", "architectures": ["BertForMaskedLM"]})

    def fail_if_model_imported(*_args, **_kwargs):
        raise AssertionError("AutoModel discovery imported a concrete model implementation")

    monkeypatch.setattr(_LazyAutoMapping, "_load_attr_from_module", fail_if_model_imported)

    auto_class, _ = select_auto_class(config)

    assert auto_class.__name__ == "AutoModelForMaskedLM"


def test_selection_does_not_probe_top_level_transformers_autoclasses(monkeypatch):
    import transformers

    config = load_native_config({"model_type": "bert", "architectures": ["BertModel"]})
    original_getattr = transformers.__getattr__

    def poison_autoclass_lookup(name):
        if name.startswith("AutoModel"):
            raise AssertionError(f"Unexpected top-level AutoModel lookup: {name}")
        return original_getattr(name)

    monkeypatch.setattr(transformers, "__getattr__", poison_autoclass_lookup)

    auto_class, _ = select_auto_class(config)

    assert auto_class.__name__ == "AutoModel"


def _phi3_with_publisher_auto_map(**overrides):
    data = {
        "model_type": "phi3",
        "architectures": ["Phi3ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
        },
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "pad_token_id": 0,
    }
    data.update(overrides)
    return data


def test_stale_root_auto_map_uses_only_verified_native_mapping_on_meta(monkeypatch):
    from transformers.models.auto import auto_factory

    def reject_dynamic_import(*_args, **_kwargs):
        raise AssertionError("Publisher code must never be imported")

    monkeypatch.setattr(auto_factory, "get_class_from_dynamic_module", reject_dynamic_import)
    config = load_native_config(_phi3_with_publisher_auto_map())

    assert type(config).__name__ == "Phi3Config"
    assert not hasattr(config, "auto_map")
    auto_class, warnings = select_auto_class(config)
    assert auto_class.__name__ == "AutoModelForCausalLM"
    assert warnings == [
        (
            "Publisher root auto_map was ignored because every declared architecture matched an installed native "
            "Transformers mapping; equivalence to publisher custom code is unverified, and remote code remains disabled"
        )
    ]

    model, selection = instantiate_native(config)

    assert type(model).__name__ == "Phi3ForCausalLM"
    assert selection["trust_remote_code"] is False
    assert selection["publisher_auto_map"] == "ignored_native_mapping_selected"
    assert selection["warnings"] == warnings
    assert all(tensor.device.type == "meta" for tensor in (*model.parameters(), *model.buffers()))


@pytest.mark.parametrize(
    "overrides",
    [
        {"architectures": []},
        {"architectures": ["Phi3ForCausalLM", "PublisherOnlyModel"]},
        {"architectures": ["PublisherOnlyModel"]},
        {"model_type": "publisher_only"},
        {"architectures": "Phi3ForCausalLM"},
        {"tokenizer_config": {"auto_map": {"AutoTokenizer": "publisher.Tokenizer"}}},
    ],
    ids=[
        "missing-declaration",
        "mixed-declarations",
        "remote-class",
        "unknown-config",
        "malformed",
        "nested-auto-map",
    ],
)
def test_publisher_auto_map_is_rejected_without_exact_native_architecture_match(overrides):
    with pytest.raises(UnsupportedModelError):
        load_native_config(_phi3_with_publisher_auto_map(**overrides))


def test_remote_only_config_remains_unsupported():
    with pytest.raises(UnsupportedModelError, match="No native Transformers config"):
        load_native_config(
            {
                "model_type": "publisher_only",
                "architectures": ["PublisherModel"],
                "auto_map": {"AutoModel": "publisher.model.PublisherModel"},
            }
        )


@pytest.mark.parametrize("auto_map", ["publisher.model.Model", []])
def test_malformed_non_object_root_auto_map_is_rejected(auto_map):
    with pytest.raises(ArtifactError, match="Invalid publisher auto_map"):
        load_native_config(_phi3_with_publisher_auto_map(auto_map=auto_map))


@pytest.mark.parametrize("auto_map", [{}, None])
def test_empty_root_auto_map_is_inert_and_cannot_forge_selection_provenance(auto_map):
    config = load_native_config(
        {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "auto_map": auto_map,
            "_model_anatomy_ignored_publisher_auto_map": True,
        }
    )

    _, warnings = select_auto_class(config)

    assert warnings == []
    assert not getattr(config, "_model_anatomy_ignored_publisher_auto_map", False)


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("MAX_CONFIG_CONTAINER_NODES", 1, "node limit"),
        ("MAX_CONFIG_NESTING_DEPTH", 0, "depth limit"),
    ],
)
def test_root_auto_map_still_counts_against_configuration_budgets(monkeypatch, limit_name, limit, message):
    from model_anatomy.src import limits

    monkeypatch.setattr(limits, limit_name, limit)

    with pytest.raises(ArtifactError, match=message):
        load_native_config(_phi3_with_publisher_auto_map())


def test_unsupported_architecture_still_obeys_original_config_budget(monkeypatch):
    from model_anatomy.src import limits

    monkeypatch.setattr(limits, "MAX_CONFIG_NESTING_DEPTH", 1)
    with pytest.raises(ArtifactError, match="depth limit"):
        load_native_config(_phi3_with_publisher_auto_map(architectures=[], nested={"deeper": {}}))


def test_foreign_config_does_not_inherit_native_auto_map_schema():
    with pytest.raises(UnsupportedModelError, match="native Transformers config"):
        load_native_config({"_class_name": "ForeignComponent", "auto_map": [], "num_attention_heads": None})
