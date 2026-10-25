"""Stale native mask and output contracts must not certify tensor facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from model_anatomy.src import native_output_contracts as contracts
from model_anatomy.src import static_flow
from model_anatomy.src.ir import TensorSpec
from model_anatomy.src.registry import instantiate_native, load_native_config
from model_anatomy.src.static_flow import StaticFlowAnalyzer


def impostor_mask_function(**kwargs):
    raise AssertionError("Source analysis must not execute mask helpers")


class MaskAlias(torch.nn.Module):
    def __init__(self, sliding_window):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False, sliding_window=sliding_window)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    def forward(self, x):
        mask_function = (
            create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        )
        mask_function(
            config=self.config,
            input_embeds=x,
            attention_mask=None,
            cache_position=None,
            past_key_values=None,
        )
        return self.layer(x)


class ShadowedMask(MaskAlias):
    def forward(self, x):
        mask_function = impostor_mask_function
        mask_function(input_embeds=x)
        return self.layer(x)


class InvalidMaskCall(MaskAlias):
    def forward(self, x):
        mask_function = create_causal_mask
        mask_function(input_embeds=x, unexpected=x)
        return self.layer(x)


def analyze(model, path=""):
    paths = {name: SimpleNamespace(kind="backbone") for name, _ in model.named_modules()}
    return StaticFlowAnalyzer(model, path, paths).analyze()


@pytest.mark.parametrize("window", [None, 8])
def test_unreviewed_mask_alias_remains_opaque_without_execution(window):
    flow = analyze(MaskAlias(window))
    assert any("Opaque call mask_function" in warning for warning in flow.warnings)
    assert not any(item["code"] == "reviewed_native_mask_alias" for item in flow.diagnostics)


@pytest.mark.parametrize("model", [ShadowedMask(None), InvalidMaskCall(None)])
def test_shadowed_or_invalid_mask_alias_remains_opaque(model):
    flow = analyze(model)
    assert any("Opaque call mask_function" in warning for warning in flow.warnings)
    assert not any(item["code"] == "reviewed_native_mask_alias" for item in flow.diagnostics)


def test_changed_mask_source_digest_remains_opaque(monkeypatch):
    monkeypatch.setitem(static_flow._REVIEWED_MASK_FACTORY_SHA256, create_causal_mask, "0" * 64)
    flow = analyze(MaskAlias(None))
    assert any("Opaque call mask_function" in warning for warning in flow.warnings)


def test_pinned_mistral_source_drift_does_not_claim_mask_or_activation_shapes():
    config = load_native_config(
        {
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "sliding_window": 8,
        }
    )
    model, _ = instantiate_native(config)
    flow = analyze(model, "model")
    assert any("Opaque call mask_function" in warning for warning in flow.warnings)
    assert not any(item["code"] == "reviewed_native_mask_alias" for item in flow.diagnostics)

    layer = model.model.layers[0]
    spec = TensorSpec(["B", "S", 16], "F32", "hidden_states")
    for module in (layer.self_attn, layer.mlp, layer.input_layernorm, layer):
        assert contracts.native_module_outputs(module, spec) is None
