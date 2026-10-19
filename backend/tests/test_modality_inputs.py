"""Symbolic input axes follow native source rather than a task-name guess."""

from types import SimpleNamespace

import torch
from transformers import CLIPConfig, SiglipConfig, T5Config, VideoMAEConfig, WhisperConfig

from model_anatomy.src.inputs import InputResolver
from model_anatomy.src.registry import instantiate_native


def test_videomae_has_frame_axis_without_executing_forward():
    config = VideoMAEConfig(
        image_size=16,
        patch_size=8,
        num_channels=3,
        num_frames=4,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        architectures=["VideoMAEForVideoClassification"],
    )
    model, _ = instantiate_native(config)
    description = InputResolver(model).describe()
    assert description["input_specs"]["pixel_values"]["shape"] == ["B", 4, 3, 16, 16]
    assert description["execution"] == "disabled"


def test_native_clip_uses_nested_vision_config_not_default_image_size():
    config = CLIPConfig(
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        vision_config={
            "image_size": 16,
            "patch_size": 8,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        architectures=["CLIPModel"],
    )
    model, _ = instantiate_native(config)
    assert InputResolver(model).spec("pixel_values").shape == ["B", 3, 16, 16]


def test_native_siglip_records_configured_pixel_axes_from_native_vision_source():
    config = SiglipConfig(
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        architectures=["SiglipModel"],
    )
    model, _ = instantiate_native(config)

    description = InputResolver(model).describe()
    assert description["input_specs"]["pixel_values"]["shape"] == ["B", 1, 32, 32]
    assert "native config" in description["shape_basis"]
    assert description["execution"] == "disabled"


def test_native_t5_input_and_cross_attention_axes_keep_encoder_and_decoder_lengths_distinct():
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        architectures=["T5ForConditionalGeneration"],
    )
    model, _ = instantiate_native(config)
    resolver = InputResolver(model)

    assert resolver.describe()["input_specs"]["input_ids"]["shape"] == ["B", "S"]
    assert resolver.describe()["input_specs"]["decoder_input_ids"]["shape"] == ["B", "T"]
    cross_path = "decoder.block.0.layer.1.EncDecAttention"
    assert resolver.spec("hidden_states", cross_path).shape == ["B", "T", 16]
    assert resolver.spec("key_value_states", cross_path).shape == ["B", "S", 16]
    assert resolver.spec("position_bias", cross_path).shape == [1, 2, "T", "S"]
    assert resolver.spec("position_bias", "decoder.block.0.layer.0.SelfAttention").shape == [1, 2, "T", "T"]
    assert resolver.spec("position_bias", "encoder.block.0.layer.0.SelfAttention").shape == [1, 2, "S", "S"]


def test_native_whisper_describes_mel_features_without_running_forward():
    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        pad_token_id=0,
        architectures=["WhisperForConditionalGeneration"],
    )
    model, _ = instantiate_native(config)

    description = InputResolver(model).describe()
    assert description["input_specs"]["input_features"]["shape"] == ["B", 8, "Frames"]
    assert description["input_specs"]["decoder_input_ids"]["shape"] == ["B", "T"]
    assert description["execution"] == "disabled"


class CompositeModel(torch.nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type, is_encoder_decoder=False, vision_config=SimpleNamespace(image_size=224)
        )

    def forward(self, pixel_values, pixel_values_videos=None):
        raise AssertionError("Forward must never execute")


def test_unknown_composite_image_video_packing_stays_unknown():
    description = InputResolver(CompositeModel("custom_multimodal")).describe()
    assert description["input_specs"]["pixel_values"]["shape"] is None
    assert description["input_specs"]["pixel_values_videos"]["shape"] is None


def test_other_video_families_are_not_assigned_videomae_axis_order():
    assert InputResolver(CompositeModel("timesformer")).spec("pixel_values").shape is None
