"""Opt-in real pinned-model checks; no checkpoint weight downloads."""

import os

import pytest

from model_anatomy.src.pipeline import reconstruct


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_HF_INTEGRATION") != "1", reason="Set RUN_HF_INTEGRATION=1 for network tests")
@pytest.mark.parametrize(
    ("model_id", "revision", "parameters", "layers", "hidden"),
    [
        ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", 124439808, 12, 768),
        ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", 134515008, 30, 576),
    ],
)
def test_real_pinned_model(model_id, revision, parameters, layers, hidden):
    ir = reconstruct(model_id, revision=revision)
    assert ir.model_id == model_id
    assert ir.metadata["revision"] == revision
    assert ir.metadata["unique_parameter_count"] == parameters
    assert ir.metadata["dimensions"]["layers"] == layers
    assert ir.metadata["dimensions"]["hidden"] == hidden
    assert ir.metadata["analysis_report"]["checkpoint_parameter_coverage"] == 1
    assert ir.metadata["artifacts"]["weight_payload_bytes_read"] == 0
    assert ir.metadata["analysis_report"]["forward_executed"] is False
    ir.root_graph.validate()
