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
    ids=["text-encoder-decoder", "swin-image", "whisper-audio", "clip-multimodal"],
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
