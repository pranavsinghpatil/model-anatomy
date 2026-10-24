"""A native mask helper selected through a local alias needs the same identity evidence."""

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


@pytest.mark.parametrize(
    ("window", "expected"),
    [(None, "create_causal_mask"), (8, "create_sliding_window_causal_mask")],
)
def test_reviewed_native_alias_is_identified_without_executing_it(window, expected):
    flow = analyze(MaskAlias(window))
    assert not any("Opaque call mask_function" in warning for warning in flow.warnings)
    assert [event.module_path for event in flow.events] == ["layer"]
    diagnostics = [item for item in flow.diagnostics if item["code"] == "reviewed_native_mask_alias"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["callable"].endswith(expected)
    assert len(diagnostics[0]["source_sha256"]) == 64
    assert "mask values are outside" in diagnostics[0]["basis"]


@pytest.mark.parametrize("model", [ShadowedMask(None), InvalidMaskCall(None)])
def test_shadowed_or_invalid_mask_alias_remains_opaque(model):
    flow = analyze(model)
    assert any("Opaque call mask_function" in warning for warning in flow.warnings)
    assert not any(item["code"] == "reviewed_native_mask_alias" for item in flow.diagnostics)


def test_changed_native_source_digest_remains_opaque(monkeypatch):
    monkeypatch.setitem(static_flow._REVIEWED_MASK_FACTORY_SHA256, create_causal_mask, "0" * 64)
    flow = analyze(MaskAlias(None))
    assert any("Opaque call mask_function" in warning for warning in flow.warnings)


@pytest.fixture(scope="module")
def tiny_mistral():
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
    return model


def test_real_native_mistral_source_uses_reviewed_alias(tiny_mistral):
    model = tiny_mistral
    flow = analyze(model, "model")
    assert not any("Opaque call mask_function" in warning for warning in flow.warnings)
    assert any(item["code"] == "reviewed_native_mask_alias" for item in flow.diagnostics)


def test_reviewed_mistral_modules_prove_hidden_state_shapes(tiny_mistral):
    layer = tiny_mistral.model.layers[0]
    spec = TensorSpec(["B", "S", 16], "F32", "hidden_states")
    for module, port in (
        (layer.self_attn, "0"),
        (layer.mlp, "output"),
        (layer.input_layernorm, "output"),
        (layer, "output"),
    ):
        assert contracts.native_module_outputs(module, spec) == {
            port: TensorSpec(["B", "S", 16], "F32", "hidden_states")
        }


def test_changed_mistral_source_or_projection_fails_closed(monkeypatch, tiny_mistral):
    spec = TensorSpec(["B", "S", 16], "F32", "hidden_states")
    layer = tiny_mistral.model.layers[0]
    family = type(layer).__module__
    monkeypatch.setitem(contracts.CONTRACT_SOURCE_SHA256, family, "0" * 64)
    assert contracts.native_module_outputs(layer, spec) is None
    monkeypatch.undo()
    replacement = torch.nn.Linear(15, 16, bias=False, device="meta")
    monkeypatch.setattr(layer.self_attn, "o_proj", replacement)
    assert contracts.native_module_outputs(layer.self_attn, spec) is None
    assert contracts.native_module_outputs(layer, spec) is None
