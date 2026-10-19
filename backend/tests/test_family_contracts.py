"""Offline family contracts using native meta models and header-only checkpoint metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from model_anatomy.api.schemas import AnalysisDetail, GraphResponse, NodeInspection
from model_anatomy.service.jobs import ir_summary, job_detail
from model_anatomy.service.views import graph_view, node_view
from model_anatomy.src.artifacts import (
    DTYPE_BYTES,
    CheckpointMetadata,
    MetadataClient,
    parse_safetensors_header,
)
from model_anatomy.src.ir import ModelIR
from model_anatomy.src.pipeline import reconstruct
from model_anatomy.src.registry import instantiate_native, load_native_config

REVISION = "a" * 40

# These small configurations exercise distinct upstream model implementations and
# public input conventions without loading checkpoints or running forward passes.
FAMILIES = [
    pytest.param(
        {
            "model_type": "bert",
            "architectures": ["BertForMaskedLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
            "type_vocab_size": 2,
        },
        "BertForMaskedLM",
        3392,
        {"input_ids": ["B", "S"]},
        id="bert-encoder",
    ),
    pytest.param(
        {
            "model_type": "distilbert",
            "architectures": ["DistilBertForMaskedLM"],
            "vocab_size": 32,
            "max_position_embeddings": 16,
            "n_layers": 1,
            "n_heads": 2,
            "dim": 16,
            "hidden_dim": 32,
        },
        "DistilBertForMaskedLM",
        3360,
        {"input_ids": ["B", "S"]},
        id="distilbert-encoder",
    ),
    pytest.param(
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 32,
            "n_positions": 16,
            "n_embd": 16,
            "n_head": 2,
            "n_layer": 1,
        },
        "GPT2LMHeadModel",
        4080,
        {"input_ids": ["B", "S"]},
        id="gpt2-causal-decoder",
    ),
    pytest.param(
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 16,
        },
        "LlamaForCausalLM",
        3632,
        {"input_ids": ["B", "S"]},
        id="llama-causal-decoder",
    ),
    pytest.param(
        {
            "model_type": "t5",
            "architectures": ["T5ForConditionalGeneration"],
            "vocab_size": 32,
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
        "T5ForConditionalGeneration",
        5872,
        {"input_ids": ["B", "S"], "decoder_input_ids": ["B", "T"]},
        id="t5-encoder-decoder",
    ),
    pytest.param(
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
        "WhisperForConditionalGeneration",
        8688,
        {"input_features": ["B", 16, "Frames"], "decoder_input_ids": ["B", "T"]},
        id="whisper-audio-encoder-decoder",
    ),
    pytest.param(
        {
            "model_type": "videomae",
            "architectures": ["VideoMAEForVideoClassification"],
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "num_frames": 4,
            "tubelet_size": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_labels": 2,
        },
        "VideoMAEForVideoClassification",
        8434,
        {"pixel_values": ["B", 4, 3, 32, 32]},
        id="videomae-video",
    ),
    pytest.param(
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
        "CLIPModel",
        17953,
        {"input_ids": ["B", "S"], "pixel_values": ["B", 3, 32, 32]},
        id="clip-text-image-multimodal",
    ),
]


class HeaderFixtureClient(MetadataClient):
    def __init__(self, config: dict, checkpoint: CheckpointMetadata):
        super().__init__()
        self.config = config
        self.fixture = checkpoint

    def resolve(self, reference):
        self.repository_files[(reference.model_id, REVISION)] = {"config.json", "model.safetensors"}
        return REVISION

    def json(self, url, max_bytes=1024 * 1024):
        self.audit.append({"url": url, "bytes_read": 1, "status": 200})
        return dict(self.config)

    def checkpoint(self, reference, revision):
        assert revision == REVISION
        return self.fixture


def header_only_checkpoint(model) -> CheckpointMetadata:
    """Make matching Safetensors metadata while retaining no parameter payload."""
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


@pytest.mark.parametrize(("config", "architecture", "unique_parameters", "expected_inputs"), FAMILIES)
def test_native_family_reconstructs_with_trustworthy_graph_and_api_views(
    config, architecture, unique_parameters, expected_inputs
):
    model, selection = instantiate_native(load_native_config(config))
    assert type(model).__name__ == architecture
    assert selection["device"] == "meta"
    assert all(parameter.device.type == "meta" for parameter in model.parameters())

    model_id = f"fixtures/{config['model_type']}"
    ir = reconstruct(model_id, client=HeaderFixtureClient(config, header_only_checkpoint(model)))

    assert ir.model_id == model_id
    assert ir.architecture == architecture
    assert ir.metadata["revision"] == REVISION
    assert ir.metadata["unique_parameter_count"] == unique_parameters
    assert ir.metadata["inputs"]["input_specs"] == {
        name: {
            "shape": shape,
            "dtype": "I64" if name in {"input_ids", "decoder_input_ids"} else "F32",
        }
        for name, shape in expected_inputs.items()
    }
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    assert ir.metadata["inputs"]["synthetic_inputs_created"] is False
    assert ir.root_graph.nodes
    for graph in ir.root_graph.walk():
        graph.validate()

    restored = ModelIR.from_json(ir.to_json())
    assert restored.to_dict() == ir.to_dict()

    summary = ir_summary(ir.to_dict())
    detail = job_detail(
        SimpleNamespace(
            id=f"fixture-{config['model_type']}",
            model_id=model_id,
            revision=REVISION,
            status=summary["result_status"],
            stage=summary["result_status"],
            error=None,
            cached=False,
            created_at=0,
            expires_at=3600,
            engine_version=ir.metadata.get("engine_version"),
        ),
        summary,
    )
    parsed_detail = AnalysisDetail.model_validate(detail)
    assert parsed_detail.architecture == architecture
    assert parsed_detail.stage == parsed_detail.status
    assert parsed_detail.parameter_count == unique_parameters
    assert parsed_detail.input_names == sorted(expected_inputs)

    parsed_graph = GraphResponse.model_validate(graph_view(ir, ir.root_graph.id))
    assert parsed_graph.id == ir.root_graph.id
    assert len(parsed_graph.nodes) == len(ir.root_graph.nodes)
    assert len(parsed_graph.edges) == len(ir.root_graph.edges)
    selected_node = next(node for node in ir.root_graph.nodes if node.parameter_count > 0)
    inspection = NodeInspection.model_validate(node_view(ir, ir.root_graph.id, selected_node.id))
    assert inspection.node.id == selected_node.id
    assert inspection.node.parameter_count > 0
    assert inspection.tensor_count > 0
    assert all(tensor.name in ir.tensors for tensor in inspection.tensors)
