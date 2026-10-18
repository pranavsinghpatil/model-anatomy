"""Repository layout routing must never mix independent components."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_artifact_inspection import GPT2_CONFIG, SHA, InspectionClient, checkpoint_with_tensor

from model_anatomy.api.app import create_app
from model_anatomy.api.settings import Settings
from model_anatomy.service.jobs import cache_key
from model_anatomy.service.views import artifact_view
from model_anatomy.src.artifacts import MetadataClient, parse_reference
from model_anatomy.src.errors import ArtifactError, UnsupportedModelError
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.profile import config_filename, native_dimensions, repository_profile
from model_anatomy.src.registry import load_native_config


def test_profile_preserves_task_declarations_without_architecture_claims():
    ref = parse_reference("org/model")
    files = {"config.json", "model.safetensors", "model.onnx", "pytorch_model.bin"}
    profile = repository_profile(ref, files, GPT2_CONFIG, {"pipeline_tag": "text-generation"}, None)
    assert profile["declared_task"] == "text-generation"
    assert profile["repository_kind"] == "transformers"
    assert profile["formats"] == ["onnx", "pytorch_binary", "safetensors"]
    assert "not verified" in profile["metadata_basis"]


def test_artifact_pipeline_import_does_not_load_numerical_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import model_anatomy.src.pipeline; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


class PipelineClient(InspectionClient):
    def __init__(self, files, manifest, config=None, checkpoint=None):
        super().__init__(files, config, checkpoint)
        self.manifest = manifest
        self.config_reads = []
        self.checkpoint_reads = 0

    def json(self, url, max_bytes=1024 * 1024):
        self.config_reads.append(url)
        if url.endswith("/model_index.json"):
            return self.manifest
        return super().json(url, max_bytes)

    def checkpoint(self, reference, revision):
        self.checkpoint_reads += 1
        return super().checkpoint(reference, revision)


def test_pipeline_inventory_requires_component_selection_and_retains_manifest():
    files = {"model_index.json", "unet/config.json", "unet/model.safetensors", "vae/model.safetensors"}
    client = PipelineClient(
        files, {"_class_name": "DiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}
    )
    ir = reconstruct("org/model", client=client, allow_artifact_inspection=True)
    assert ir.metadata["inspection"]["reason_code"] == "pipeline_components_require_selection"
    assert client.checkpoint_reads == 0
    assert ir.root_graph.nodes == [] and ir.tensors == {}
    profile = artifact_view(ir)["model_profile"]
    assert profile["repository_kind"] == "diffusers_pipeline"
    assert profile["components"] == [
        {
            "path": "unet",
            "library": "diffusers",
            "declared_class": "UNet2DConditionModel",
            "checkpoint_files": ["unet/model.safetensors"],
        }
    ]


def test_explicit_component_does_not_use_unrelated_root_configuration():
    files = {"config.json", "unet/model.safetensors"}
    client = InspectionClient(files, GPT2_CONFIG, checkpoint_with_tensor())
    url = f"https://huggingface.co/org/model/blob/{SHA}/unet/model.safetensors"
    ir = reconstruct(url, client=client, allow_artifact_inspection=True)
    assert ir.metadata["inspection"]["reason_code"] == "architecture_config_unavailable"
    assert "source_config" not in ir.metadata
    assert ir.metadata["model_profile"]["selected_config"] is None
    assert ir.tensors and not ir.root_graph.nodes


def test_explicit_component_reads_sibling_config_and_records_native_blocker():
    files = {"model_index.json", "unet/config.json", "unet/model.safetensors"}
    client = PipelineClient(
        files,
        {"unet": ["diffusers", "UNet2DConditionModel"]},
        {"_class_name": "UNet2DConditionModel"},
        checkpoint_with_tensor(),
    )
    url = f"https://huggingface.co/org/model/blob/{SHA}/unet/model.safetensors"
    ir = reconstruct(url, client=client, allow_artifact_inspection=True)
    assert ir.metadata["inspection"]["reason_code"] == "native_architecture_unavailable"
    assert ir.metadata["model_profile"]["selected_config"] == "unet/config.json"
    assert ir.metadata["model_profile"]["architecture_blockers"]
    assert ir.metadata["artifacts"]["config_url"].endswith("/unet/config.json")
    assert client.checkpoint_reads == 1


def test_nested_index_resolves_relative_shards_and_keeps_raw_index():
    client = MetadataClient()
    ref = parse_reference(
        f"https://huggingface.co/org/model/blob/{SHA}/unet/diffusion_pytorch_model.safetensors.index.json"
    )
    files = {ref.filename, "unet/part.safetensors", "part.safetensors"}
    client.repository_files[(ref.model_id, SHA)] = files
    index = {"weight_map": {"layers.0.weight": "part.safetensors"}}
    client.json = lambda *args, **kwargs: index
    urls = []

    def read(url):
        urls.append(url)
        return checkpoint_with_tensor().shards["model.safetensors"]

    client.safetensors = read
    result = client.checkpoint(ref, SHA)
    assert result.tensor_files == {"layers.0.weight": "unet/part.safetensors"}
    assert urls == [ref.url("unet/part.safetensors", SHA)]
    assert result.index == index


@pytest.mark.parametrize(
    "filename",
    [
        "../part.safetensors",
        "/part.safetensors",
        "folder\\part.safetensors",
        "./part.safetensors",
        "folder//part.safetensors",
    ],
)
def test_component_index_rejects_escaping_or_ambiguous_paths(filename):
    client = MetadataClient()
    ref = parse_reference(f"https://huggingface.co/org/model/blob/{SHA}/unet/model.safetensors.index.json")
    client.repository_files[(ref.model_id, SHA)] = {ref.filename, filename, "unet/" + filename}
    client.json = lambda *args, **kwargs: {"weight_map": {"layers.0.weight": filename}}
    with pytest.raises(ArtifactError, match="invalid shard paths"):
        client.checkpoint(ref, SHA)


@pytest.mark.parametrize("path", ["../unet", "/unet", "unet\\nested", ".", ".."])
def test_manifest_paths_are_not_executed_or_followed(path):
    with pytest.raises(ArtifactError, match="component declaration"):
        repository_profile(parse_reference("org/model"), set(), None, {}, {path: ["custom", "ArbitraryCode"]})


def test_config_selection_is_directory_scoped():
    assert config_filename(parse_reference("org/model")) == "config.json"
    assert (
        config_filename(
            parse_reference(f"https://huggingface.co/org/model/blob/{SHA}/text_encoder/model.safetensors")
        )
        == "text_encoder/config.json"
    )


def test_swin_dimensions_preserve_stages_and_count_actual_blocks():
    config = load_native_config({"model_type": "swin", "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]})
    assert native_dimensions(config)["layers"] == 12
    assert native_dimensions(config)["heads"] == [3, 6, 12, 24]


@pytest.mark.parametrize("framework", ["mlx", "np", "tf", "flax"])
def test_non_pytorch_safetensors_still_have_verified_artifacts(framework):
    checkpoint = checkpoint_with_tensor()
    checkpoint.shards["model.safetensors"].metadata["format"] = framework
    files = {"config.json", "model.safetensors"}
    ir = reconstruct(
        "org/model", client=InspectionClient(files, GPT2_CONFIG, checkpoint), allow_artifact_inspection=True
    )
    assert ir.metadata["inspection"]["reason_code"] == "checkpoint_framework_unavailable"
    assert ir.tensors and not ir.root_graph.nodes
    assert framework in ir.metadata["model_profile"]["architecture_blockers"][0]
    with pytest.raises(UnsupportedModelError, match="non-PyTorch"):
        reconstruct("org/model", client=InspectionClient(files, GPT2_CONFIG, checkpoint))


def test_api_serves_staged_dimensions_instead_of_returning_503(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path)), base_url="http://127.0.0.1") as client:
        service = client.app.state.jobs
        service.acquire_worker("worker")
        created = client.post("/api/v1/analyses", json={"model": "org/staged-model"}).json()
        assert service.claim("worker") == created["id"]
        report = json.loads((Path(__file__).resolve().parents[2] / "output/gpt2/model.json").read_bytes())
        report["model_id"] = "org/staged-model"
        report["metadata"]["dimensions"]["heads"] = [3, 6, 12, 24]
        service.resolved(created["id"], "worker", report["model_id"], report["metadata"]["revision"])
        key = cache_key(
            report["model_id"], report["metadata"]["revision"], service.settings.engine_version, "1.1"
        )
        service.finish(created["id"], "worker", json.dumps(report).encode(), key, cached=False)
        response = client.get(f"/api/v1/analyses/{created['id']}")
        assert response.status_code == 200
        assert response.json()["dimensions"]["heads"] == [3, 6, 12, 24]
        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        assert response.json()["analyses"][0]["dimensions"]["heads"] == [3, 6, 12, 24]
