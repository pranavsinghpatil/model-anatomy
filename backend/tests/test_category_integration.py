"""Pinned cross-category evidence, not a claim of complete modality semantics."""

import os

import pytest

from model_anatomy.service.views import artifact_view
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests"
    ),
]


@pytest.mark.parametrize(
    "model,revision,architecture,tensors,task,input_name,input_shape",
    [
        (
            "google-t5/t5-small",
            "df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
            "T5ForConditionalGeneration",
            132,
            "translation",
            "input_ids",
            ["B", "S"],
        ),
        (
            "microsoft/swin-tiny-patch4-window7-224",
            "d00d478bfaf1f417d34a1186673ad4b4be264c51",
            "SwinForImageClassification",
            233,
            "image-classification",
            "pixel_values",
            ["B", 3, 224, 224],
        ),
        (
            "openai/whisper-tiny",
            "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
            "WhisperForConditionalGeneration",
            167,
            "automatic-speech-recognition",
            "input_features",
            ["B", 80, "Frames"],
        ),
        (
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "488eb9a0565f257b32866000305c8178965eb9f6",
            "VideoMAEForVideoClassification",
            186,
            "video-classification",
            "pixel_values",
            ["B", 16, 3, 224, 224],
        ),
        (
            "google/siglip-base-patch16-224",
            "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
            "SiglipModel",
            408,
            "zero-shot-image-classification",
            "pixel_values",
            None,
        ),
    ],
)
def test_pinned_native_category(model, revision, architecture, tensors, task, input_name, input_shape):
    ir = reconstruct(model, revision=revision, allow_artifact_inspection=True)
    assert ir.metadata["revision"] == revision
    assert ir.architecture == architecture
    assert len(ir.tensors) == tensors
    assert ir.metadata["model_profile"]["declared_task"] == task
    assert ir.metadata["inputs"]["input_specs"][input_name]["shape"] is None
    assert ir.metadata["analysis_report"]["module_inventory_coverage"] == 1
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    if architecture == "SwinForImageClassification":
        assert ir.metadata["dimensions"]["heads"] == [3, 6, 12, 24]
        assert ir.metadata["dimensions"]["layers"] == 12
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.root_graph.nodes
    assert ModelIR.from_json(ir.to_json()).to_dict() == ir.to_dict()
    ir.root_graph.validate()


def test_real_diffusion_pipeline_and_component_are_not_combined():
    model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    inventory = reconstruct(model, revision=revision, allow_artifact_inspection=True)
    assert inventory.metadata["inspection"]["reason_code"] == "pipeline_components_require_selection"
    components = {item["path"]: item for item in inventory.metadata["model_profile"]["components"]}
    assert components["unet"]["declared_class"] == "UNet2DConditionModel"
    assert components["text_encoder"]["declared_class"] == "CLIPTextModel"
    assert inventory.tensors == {} and inventory.root_graph.nodes == []
    assert any(item.source == "hf_pipeline_manifest" for item in inventory.evidence)
    url = f"https://huggingface.co/{model}/blob/{revision}/unet/diffusion_pytorch_model.safetensors"
    component = reconstruct(url, revision=revision, allow_artifact_inspection=True)
    assert component.metadata["inspection"]["reason_code"] == "native_architecture_unavailable"
    assert component.root_graph.nodes == []
    assert component.tensors
    assert all(
        tensor["artifact"].endswith("/unet/diffusion_pytorch_model.safetensors")
        for tensor in component.tensors.values()
    )
    assert artifact_view(component)["model_profile"]["selected_config"] == "unet/config.json"
    assert component.metadata["artifacts"]["weight_payload_bytes_read"] == 0
