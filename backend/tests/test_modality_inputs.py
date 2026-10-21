"""Declared interfaces do not imply unobserved activation shapes."""

from types import SimpleNamespace

import torch
from transformers import CLIPConfig, VideoMAEConfig

from model_anatomy.src.inputs import describe_inputs
from model_anatomy.src.registry import instantiate_native


def test_videomae_interface_does_not_infer_frame_axes():
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
    description = describe_inputs(model, ["pixel_values"])
    assert description["input_specs"]["pixel_values"]["shape"] is None
    assert description["execution"] == "disabled"


def test_native_clip_interface_does_not_guess_image_size():
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
    assert describe_inputs(model, ["pixel_values"])["input_specs"]["pixel_values"]["shape"] is None


class CompositeModel(torch.nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type, is_encoder_decoder=False, vision_config=SimpleNamespace(image_size=224)
        )

    def forward(self, pixel_values, pixel_values_videos=None):
        raise AssertionError("Forward must never execute")


def test_unknown_composite_image_video_packing_stays_unknown():
    description = describe_inputs(CompositeModel("custom_multimodal"), ["pixel_values", "pixel_values_videos"])
    assert description["input_specs"]["pixel_values"]["shape"] is None
    assert description["input_specs"]["pixel_values_videos"]["shape"] is None


def test_other_video_families_are_not_assigned_videomae_axis_order():
    assert describe_inputs(CompositeModel("timesformer"), ["pixel_values"])["input_specs"]["pixel_values"]["shape"] is None
