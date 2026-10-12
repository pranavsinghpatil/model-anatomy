"""Symbolic input axes follow native source rather than a task-name guess."""

from types import SimpleNamespace

import torch
from transformers import CLIPConfig, VideoMAEConfig

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
