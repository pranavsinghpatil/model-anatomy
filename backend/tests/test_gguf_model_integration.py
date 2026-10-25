"""A second publisher/model guards against a Bonsai-specific GGUF implementation."""

import os

import pytest

from model_anatomy.src.pipeline import reconstruct


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests"
)
def test_pinned_smollm_quantized_gguf():
    revision = "09816acd5d99df7be770d85ea30822623dab342c"
    model = "bartowski/SmolLM2-135M-Instruct-GGUF"
    filename = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
    ir = reconstruct(
        f"https://huggingface.co/{model}/blob/{revision}/{filename}",
        allow_artifact_inspection=True,
    )
    assert ir.metadata["revision"] == revision
    assert ir.metadata["inspection"]["reason_code"] == "gguf_header_only"
    assert ir.metadata["inspection"]["selected_checkpoint_files"] == [filename]
    assert len(ir.tensors) > 100
    assert all(item["shape_convention"] == "ggml_dimensions" for item in ir.tensors.values())
    assert ir.root_graph.nodes == [] and ir.root_graph.edges == []
    artifacts = ir.metadata["artifacts"]
    assert artifacts["checkpoint_format"] == "gguf"
    assert artifacts["weight_payload_bytes_read"] == 0
    shard = artifacts["shards"][filename]
    assert len(ir.tensors) == shard["tensor_count"]
    assert all(item["range"][1] < shard["data_start"] for item in artifacts["requests"] if "range" in item)
